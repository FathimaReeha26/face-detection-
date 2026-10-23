"""Anti-spoofing check using a MiniFASNet model.

The model weights are not shipped with the project. Download them manually
from the Silent-Face-Anti-Spoofing project (MiniFASNet) and place an ONNX
export in models/antispoof/ (see models/antispoof/README.md).

The crop is resized to the model's input size and passed through as a float32
NCHW tensor without division by 255, matching the MiniFASNet training and
the yakhyo/face-anti-spoofing inference code.
"""

import glob
import os

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional dependency
    ort = None

SPOOF_THRESHOLD = 0.5
DEFAULT_MODEL_DIR = os.path.join("models", "antispoof")
DEFAULT_SCALE = 2.7


class AntiSpoofChecker:
    """Scores a face crop as real (1.0) or spoof (0.0)."""

    def __init__(self, model_path=None, scale=DEFAULT_SCALE):
        if model_path is None:
            candidates = sorted(glob.glob(os.path.join(DEFAULT_MODEL_DIR, "*.onnx")))
            if not candidates:
                raise RuntimeError(
                    "Anti-spoofing model weights are missing. Download the "
                    "MiniFASNet weights from "
                    "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing "
                    "and place an ONNX export in models/antispoof/ "
                    "(for example models/antispoof/model.onnx)."
                )
            model_path = candidates[0]

        if not os.path.isfile(model_path):
            raise RuntimeError(
                f"Anti-spoofing model weights not found: {model_path}. "
                "Place the ONNX file in models/antispoof/."
            )
        if ort is None:
            raise RuntimeError(
                "onnxruntime is required for anti-spoofing. Install it with "
                "python -m pip install onnxruntime."
            )

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.scale = scale
        input_config = self.session.get_inputs()[0]
        self.input_name = input_config.name
        self.input_size = tuple(input_config.shape[2:])
        self.output_name = self.session.get_outputs()[0].name

    def _crop_face(self, image, box):
        height, width = image.shape[:2]
        x, y, box_w, box_h = (int(value) for value in box)
        scale = min((height - 1) / max(1, box_h), (width - 1) / max(1, box_w), self.scale)
        new_w = box_w * scale
        new_h = box_h * scale
        center_x = x + box_w / 2.0
        center_y = y + box_h / 2.0
        x1 = max(0, int(center_x - new_w / 2.0))
        y1 = max(0, int(center_y - new_h / 2.0))
        x2 = min(width - 1, int(center_x + new_w / 2.0))
        y2 = min(height - 1, int(center_y + new_h / 2.0))
        cropped = image[y1 : y2 + 1, x1 : x2 + 1]
        return cv2.resize(cropped, self.input_size[::-1])

    def _preprocess(self, image, box):
        face = self._crop_face(image, box).astype(np.float32)
        face = np.transpose(face, (2, 0, 1))
        return np.expand_dims(face, axis=0)

    def score(self, image, box):
        """Return the probability that the face is real."""
        input_tensor = self._preprocess(image, box)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        # MiniFASNet uses index 1 for the real class.
        return float(probabilities[0, 1])
