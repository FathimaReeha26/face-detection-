"""Small DNN face detector used by the liveness webcam script.

Wraps OpenCV's ResNet-10 SSD detector so the liveness check does not depend
on any other module in the project. Returns a list of (x, y, w, h) boxes.
"""

import os

import cv2

PROTO_PATH = os.path.join("models", "deploy.prototxt")
WEIGHTS_PATH = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")
CONFIDENCE_THRESHOLD = 0.6


class DnnDetector:
    """ResNet-10 SSD face detector."""

    def __init__(self, proto_path=PROTO_PATH, weights_path=WEIGHTS_PATH):
        if not os.path.isfile(proto_path):
            raise RuntimeError(f"Missing model file: {proto_path}")
        if not os.path.isfile(weights_path):
            raise RuntimeError(f"Missing model file: {weights_path}")
        self.net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)

    def detect(self, image):
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        output = self.net.forward()

        boxes = []
        for index in range(output.shape[2]):
            confidence = float(output[0, 0, index, 2])
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            x1 = max(0, int(output[0, 0, index, 3] * width))
            y1 = max(0, int(output[0, 0, index, 4] * height))
            x2 = min(width - 1, int(output[0, 0, index, 5] * width))
            y2 = min(height - 1, int(output[0, 0, index, 6] * height))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
