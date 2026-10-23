"""GPU detection backend built on RetinaFace and ONNX Runtime.

This is the alternative to SCRFD. It expects the RetinaFace ONNX exports
that return a box tensor, a score tensor and a landmark tensor. The outputs
are matched by their shape, so the order the model returns them in does not
matter. The five landmarks use the same order as SCRFD and YuNet, so the
SFace alignment works with them as well.
"""

import cv2
import numpy as np

from ..config import DetectorConfig
from ..device import CUDA_PROVIDER
from ..postprocess import clip_box, corners_to_box, landmark_row, letterbox, suppress
from ..types import Detection
from .base import BaseDetector


class RetinaFaceDetector(BaseDetector):
    """RetinaFace (ResNet-50 backbone) executed by ONNX Runtime."""

    frame_interval = 1
    supports_fallback = True

    def __init__(self, model_path: str, config: DetectorConfig, providers) -> None:
        import onnxruntime

        self.model_path = model_path
        self.config = config
        self.session = onnxruntime.InferenceSession(
            model_path, providers=list(providers)
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self._read_output_layout()

        active = self.session.get_providers()
        if CUDA_PROVIDER in active:
            self.name = "RetinaFace (ONNX Runtime, CUDA)"
            self.uses_gpu = True
        else:
            self.name = "RetinaFace (ONNX Runtime, CPU)"

    def _read_output_layout(self) -> None:
        """Match the model outputs to their role by the width of the tensor."""
        self.box_output = None
        self.score_output = None
        self.landmark_output = None

        for output in self.session.get_outputs():
            width = output.shape[-1] if output.shape else 0
            if width == 4 and self.box_output is None:
                self.box_output = output.name
            elif width == 10 and self.landmark_output is None:
                self.landmark_output = output.name
            elif width in (1, 2) and self.score_output is None:
                self.score_output = output.name

        if self.box_output is None or self.score_output is None:
            raise RuntimeError(
                "Unsupported RetinaFace model: expected a box and a score output."
            )

    def detect(self, image: "cv2.Mat") -> list[Detection]:
        height, width = image.shape[:2]
        canvas, scale = letterbox(image, self.config.input_size)
        blob = cv2.dnn.blobFromImage(
            canvas,
            1.0 / 128.0,
            self.config.input_size,
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        results = dict(zip(self.output_names, outputs))

        boxes = np.asarray(results[self.box_output]).reshape(-1, 4) / scale
        scores = self._read_scores(results[self.score_output])
        landmarks = None
        if self.landmark_output is not None:
            landmarks = (
                np.asarray(results[self.landmark_output]).reshape(-1, 5, 2) / scale
            )

        detections = []
        for index, (box, score) in enumerate(zip(boxes, scores)):
            clipped = clip_box(corners_to_box(*box), width, height)
            if clipped is None:
                continue
            points = None
            row = None
            if landmarks is not None and index < len(landmarks):
                points = tuple(
                    (float(point[0]), float(point[1])) for point in landmarks[index]
                )
                row = landmark_row(clipped, points, float(score))
            detections.append(
                Detection(
                    box=clipped,
                    row=row,
                    confidence=float(score),
                    landmarks=points,
                )
            )

        return suppress(
            detections, self.config.score_threshold, self.config.nms_threshold
        )

    def _read_scores(self, raw) -> np.ndarray:
        """Return the face probability of every prediction."""
        values = np.asarray(raw)
        if values.ndim >= 2 and values.shape[-1] == 2:
            # The score tensor holds a background and a face probability.
            return values.reshape(-1, 2)[:, 1]
        return values.reshape(-1)
