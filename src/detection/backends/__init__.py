"""The detection backends the project can use."""

from .base import BaseDetector
from .retinaface import RetinaFaceDetector
from .scrfd import ScrfdDetector
from .yunet import YuNetDetector

__all__ = [
    "BaseDetector",
    "RetinaFaceDetector",
    "ScrfdDetector",
    "YuNetDetector",
]
