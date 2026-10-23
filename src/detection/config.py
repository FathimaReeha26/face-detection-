"""Configuration for the detection layer.

The backend is picked from environment variables so the plain scripts and the
packaged launcher behave the same way without any extra command line
plumbing:

    FACE_DETECTOR_MODE        auto (default) | gpu | cpu
    FACE_DETECTOR_MODEL       auto (default) | scrfd | retinaface | yunet
    FACE_DETECTOR_MODELS_DIR  folder holding the model files (default "models")

The thresholds live here as well, so both the GPU and the CPU backend work
with the same shared configuration instead of repeating their own values.
"""

import os
from dataclasses import dataclass

from .logging_setup import get_logger

MODE_VARIABLE = "FACE_DETECTOR_MODE"
MODEL_VARIABLE = "FACE_DETECTOR_MODEL"
MODELS_DIR_VARIABLE = "FACE_DETECTOR_MODELS_DIR"

MODES = ("auto", "gpu", "cpu")
MODELS = ("auto", "scrfd", "retinaface", "yunet")

DEFAULT_MODELS_DIR = "models"
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SCRFD_FILENAME = "scrfd_10g.onnx"
RETINAFACE_FILENAME = "retinaface_resnet50.onnx"


@dataclass(frozen=True)
class DetectorConfig:
    """Thresholds and input size a backend works with."""

    score_threshold: float
    nms_threshold: float
    input_size: tuple[int, int]
    top_k: int = 5000


# YuNet keeps the thresholds this project has always been tuned with. SCRFD
# and RetinaFace use the thresholds of their reference implementations.
BACKEND_CONFIGS: dict[str, DetectorConfig] = {
    "yunet": DetectorConfig(
        score_threshold=0.9, nms_threshold=0.3, input_size=(320, 320)
    ),
    "scrfd": DetectorConfig(
        score_threshold=0.5, nms_threshold=0.4, input_size=(640, 640)
    ),
    "retinaface": DetectorConfig(
        score_threshold=0.5, nms_threshold=0.4, input_size=(640, 640)
    ),
}


@dataclass(frozen=True)
class DetectorSettings:
    """Which backend to build and where the model files live."""

    mode: str = "auto"
    model: str = "auto"
    models_dir: str = DEFAULT_MODELS_DIR

    @classmethod
    def from_env(cls) -> "DetectorSettings":
        logger = get_logger()
        mode = _read_choice(MODE_VARIABLE, MODES, "auto", logger)
        model = _read_choice(MODEL_VARIABLE, MODELS, "auto", logger)
        models_dir = (
            os.environ.get(MODELS_DIR_VARIABLE, "").strip() or DEFAULT_MODELS_DIR
        )
        return cls(mode=mode, model=model, models_dir=models_dir)

    def model_path(self, filename: str) -> str:
        """Return the full path of a model file inside the models folder."""
        return os.path.join(self.models_dir, filename)


def _read_choice(variable: str, allowed: tuple[str, ...], default: str, logger) -> str:
    """Read an environment variable and fall back to the default value."""
    value = os.environ.get(variable, "").strip().lower()
    if not value:
        return default
    if value not in allowed:
        logger.warning(
            "Ignoring %s=%s; expected one of %s.",
            variable,
            value,
            ", ".join(allowed),
        )
        return default
    return value
