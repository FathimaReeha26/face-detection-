"""CPU detection backend built on OpenCV's YuNet detector."""

import cv2
import numpy as np

from ..config import DetectorConfig
from ..postprocess import clip_box
from ..types import Detection
from .base import BaseDetector


class YuNetDetector(BaseDetector):
    """YuNet through cv2.FaceDetectorYN.

    This is the backend the project has always used. It runs on the CPU and
    keeps the thresholds the project was tuned with, so switching between the
    backends does not change how YuNet behaves.
    """

    name = "YuNet (DNN, CPU)"
    frame_interval = 4
    supports_fallback = False

    def __init__(self, model_path: str, config: DetectorConfig) -> None:
        self.model_path = model_path
        self.config = config
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            config.input_size,
            config.score_threshold,
            config.nms_threshold,
            config.top_k,
        )

    def detect(self, image: "cv2.Mat") -> list[Detection]:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return []

        detections = []
        for row in faces:
            box = clip_box(
                (int(row[0]), int(row[1]), int(row[2]), int(row[3])), width, height
            )
            if box is None:
                continue
            landmarks = tuple(
                (float(row[4 + 2 * index]), float(row[5 + 2 * index]))
                for index in range(5)
            )
            detections.append(
                Detection(
                    box=box,
                    row=np.asarray(row, dtype=np.float32),
                    confidence=float(row[14]),
                    landmarks=landmarks,
                )
            )
        return detections
