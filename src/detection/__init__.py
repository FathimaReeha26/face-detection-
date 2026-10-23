"""The hybrid face detection layer.

The package holds the shared types, the device check, the detection
backends and the factory that picks the right one for the machine.
"""

from .config import DetectorConfig, DetectorSettings
from .device import DeviceInfo, detect_device
from .factory import create_cpu_detector, create_detector
from .logging_setup import configure_logging, get_logger
from .tracking import TrackedFaceDetector
from .types import Box, Detection, Landmark

__all__ = [
    "Box",
    "Detection",
    "DetectorConfig",
    "DetectorSettings",
    "DeviceInfo",
    "Landmark",
    "TrackedFaceDetector",
    "configure_logging",
    "create_cpu_detector",
    "create_detector",
    "detect_device",
    "get_logger",
]
