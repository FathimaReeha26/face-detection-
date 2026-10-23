"""GPU detection backend built on SCRFD and ONNX Runtime.

The model is the insightface SCRFD export. Its scores, boxes and landmarks
are decoded the same way the reference implementation does it, and the five
landmarks come out in the order OpenCV's YuNet uses, so a face found here
can be aligned by SFace exactly like a face found by the CPU backend.
"""

import cv2
import numpy as np

from ..config import DetectorConfig
from ..device import CUDA_PROVIDER
from ..postprocess import clip_box, corners_to_box, landmark_row, letterbox, suppress
from ..types import Detection
from .base import BaseDetector

# The SCRFD export uses three feature maps for the small models and five for
# the larger ones, each with its own stride.
STRIDES = {
    3: (8, 16, 32),
    5: (8, 16, 32, 64, 128),
}


class ScrfdDetector(BaseDetector):
    """SCRFD (insightface) executed by ONNX Runtime.

    The session is created with the CUDA provider, so the model runs on the
    GPU. If the provider is not usable the session still starts on the CPU,
    which keeps the application running on a machine without CUDA.
    """

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
        self._center_cache: dict[tuple[int, int, int], np.ndarray] = {}

        active = self.session.get_providers()
        if CUDA_PROVIDER in active:
            self.name = "SCRFD (ONNX Runtime, CUDA)"
            self.uses_gpu = True
        else:
            self.name = "SCRFD (ONNX Runtime, CPU)"

    def _read_output_layout(self) -> None:
        """Work out how the model lays out its scores, boxes and landmarks."""
        count = len(self.output_names)
        if count in (6, 9):
            self.feature_maps = 3
            self.anchors = 2
        elif count in (10, 15):
            self.feature_maps = 5
            self.anchors = 1
        else:
            raise RuntimeError(
                f"Unsupported SCRFD model: {count} outputs, expected 6, 9, 10 or 15."
            )
        self.strides = STRIDES[self.feature_maps]
        self.has_landmarks = count in (9, 15)
        self.batched = len(self.session.get_outputs()[0].shape) == 3

    def detect(self, image: "cv2.Mat") -> list[Detection]:
        height, width = image.shape[:2]
        canvas, scale = self._prepare(image)
        blob = cv2.dnn.blobFromImage(
            canvas,
            1.0 / 128.0,
            self.config.input_size,
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        boxes, scores, landmarks = self._decode(
            outputs, blob.shape[2], blob.shape[3], scale
        )

        detections = []
        for index, (box, score) in enumerate(zip(boxes, scores)):
            clipped = clip_box(corners_to_box(*box), width, height)
            if clipped is None:
                continue
            points = landmarks[index] if landmarks is not None else None
            row = None
            if points is not None:
                row = landmark_row(clipped, points, score)
            detections.append(
                Detection(
                    box=clipped,
                    row=row,
                    confidence=float(score),
                    landmarks=tuple(points) if points is not None else None,
                )
            )

        return suppress(
            detections, self.config.score_threshold, self.config.nms_threshold
        )

    def _prepare(self, image: "cv2.Mat") -> tuple["cv2.Mat", float]:
        """Scale the image into the model input and pad the rest."""
        return letterbox(image, self.config.input_size)

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        """Return the centre of every anchor of one feature map."""
        key = (height, width, stride)
        if key not in self._center_cache:
            grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (grid * stride).reshape((-1, 2))
            if self.anchors > 1:
                centers = np.stack([centers] * self.anchors, axis=1).reshape((-1, 2))
            self._center_cache[key] = centers
        return self._center_cache[key]

    def _decode(self, outputs, input_height: int, input_width: int, scale: float):
        """Turn the raw model output into boxes, scores and landmarks."""
        all_boxes = []
        all_scores = []
        all_landmarks = []
        maps = self.feature_maps

        for index, stride in enumerate(self.strides):
            if self.batched:
                scores = outputs[index][0]
                box_predictions = outputs[index + maps][0] * stride
                landmark_predictions = (
                    outputs[index + maps * 2][0] * stride if self.has_landmarks else None
                )
            else:
                scores = outputs[index]
                box_predictions = outputs[index + maps] * stride
                landmark_predictions = (
                    outputs[index + maps * 2] * stride if self.has_landmarks else None
                )

            centers = self._anchor_centers(
                input_height // stride, input_width // stride, stride
            )
            keep = np.where(scores >= self.config.score_threshold)[0]

            all_boxes.append(_distance_to_box(centers, box_predictions)[keep] / scale)
            all_scores.append(scores[keep].reshape(-1))

            if landmark_predictions is not None:
                points = _distance_to_landmarks(centers, landmark_predictions)
                points = points.reshape((points.shape[0], -1, 2))[keep] / scale
            else:
                points = np.empty((0, 5, 2), dtype=np.float32)
            all_landmarks.append(points)

        if not all_boxes:
            return [], [], []

        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        order = np.argsort(-scores, kind="stable")
        boxes = boxes[order]
        scores = scores[order]

        if self.has_landmarks:
            landmarks = np.vstack(all_landmarks)[order]
        else:
            landmarks = None
        return boxes, scores, landmarks


def _distance_to_box(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Decode the box a prediction describes around its anchor."""
    x1 = centers[:, 0] - distances[:, 0]
    y1 = centers[:, 1] - distances[:, 1]
    x2 = centers[:, 0] + distances[:, 2]
    y2 = centers[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance_to_landmarks(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Decode the five landmark points a prediction describes."""
    points = []
    for index in range(0, distances.shape[1], 2):
        points.append(centers[:, index % 2] + distances[:, index])
        points.append(centers[:, index % 2 + 1] + distances[:, index + 1])
    return np.stack(points, axis=-1)
