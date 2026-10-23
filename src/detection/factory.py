"""Builds the detection backend that fits the machine and the settings.

The rule is simple: when a CUDA device is available the GPU backend is used,
otherwise the light CPU backend takes over. Both are picked without any
manual configuration; the environment variables in detection.config can
override the choice for testing.
"""

import os

import cv2

from .backends.base import BaseDetector
from .backends.retinaface import RetinaFaceDetector
from .backends.scrfd import ScrfdDetector
from .backends.yunet import YuNetDetector
from .config import (
    BACKEND_CONFIGS,
    RETINAFACE_FILENAME,
    SCRFD_FILENAME,
    DetectorSettings,
    YUNET_FILENAME,
)
from .device import CUDA_PROVIDER, CPU_PROVIDER, detect_device
from .logging_setup import configure_logging


def create_detector(settings=None, device=None) -> BaseDetector:
    """Return the backend to use for this session."""
    logger = configure_logging()
    settings = settings or DetectorSettings.from_env()
    device = device or detect_device()

    if settings.model == "yunet":
        logger.info("YuNet requested explicitly; using the CPU backend.")
        return create_cpu_detector(settings)

    if settings.model in ("scrfd", "retinaface"):
        detector = _create_gpu_detector(
            settings.model, settings, device, logger, allow_cpu=True
        )
        if detector is not None:
            return detector
        logger.warning(
            "The %s backend is not available; using the YuNet CPU backend instead.",
            settings.model,
        )
        return create_cpu_detector(settings)

    if settings.mode == "cpu":
        logger.info("CPU mode is forced; using the YuNet CPU backend.")
        return create_cpu_detector(settings)

    if settings.mode == "gpu":
        detector = _create_gpu_detector("scrfd", settings, device, logger)
        if detector is None:
            detector = _create_gpu_detector("retinaface", settings, device, logger)
        if detector is None:
            logger.warning(
                "GPU mode is forced but no GPU backend could be started; "
                "using the YuNet CPU backend instead."
            )
            return create_cpu_detector(settings)
        return detector

    if device.cuda:
        detector = _create_gpu_detector("scrfd", settings, device, logger)
        if detector is not None:
            return detector
    else:
        logger.info(
            "No CUDA device available (%s); using the YuNet CPU backend.",
            device.describe(),
        )
    return create_cpu_detector(settings)


def create_cpu_detector(settings=None) -> YuNetDetector:
    """Build the CPU backend. Raises when the model file is missing."""
    logger = configure_logging()
    settings = settings or DetectorSettings.from_env()
    path = settings.model_path(YUNET_FILENAME)
    if not os.path.isfile(path):
        raise RuntimeError(
            f"No face detector available: {path} is missing. Download "
            "face_detection_yunet_2023mar.onnx from "
            "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet "
            "and put it in the models folder."
        )
    try:
        detector = YuNetDetector(path, BACKEND_CONFIGS["yunet"])
    except cv2.error as error:
        raise RuntimeError(f"Could not load the YuNet model {path}: {error}") from error
    logger.info("Detector: %s (%s).", detector.name, path)
    return detector


def _create_gpu_detector(
    kind: str,
    settings: DetectorSettings,
    device,
    logger,
    allow_cpu: bool = False,
) -> BaseDetector | None:
    """Build a GPU backend, or return None when it cannot be used.

    With allow_cpu the backend is also started without a CUDA provider, which
    keeps an explicitly requested model working on a machine without a GPU.
    """
    if not device.onnxruntime:
        logger.warning(
            "ONNX Runtime is not installed, so the %s backend cannot be used. "
            "Install onnxruntime-gpu for GPU detection.",
            kind,
        )
        return None

    providers = [CUDA_PROVIDER, CPU_PROVIDER] if device.cuda else [CPU_PROVIDER]
    if not device.cuda and not allow_cpu:
        logger.warning(
            "ONNX Runtime has no CUDA provider, so the %s backend cannot be used.",
            kind,
        )
        return None
    if not device.cuda:
        logger.warning(
            "No CUDA provider is available; running the %s backend on the CPU.",
            kind,
        )

    filename = SCRFD_FILENAME if kind == "scrfd" else RETINAFACE_FILENAME
    path = settings.model_path(filename)
    if not os.path.isfile(path):
        logger.warning(
            "The %s model %s is missing, so the GPU backend cannot be used.",
            kind,
            path,
        )
        return None

    try:
        if kind == "scrfd":
            detector = ScrfdDetector(path, BACKEND_CONFIGS["scrfd"], providers)
        else:
            detector = RetinaFaceDetector(path, BACKEND_CONFIGS["retinaface"], providers)
    except Exception as error:
        logger.warning(
            "The %s backend could not be started (%s); using the CPU backend.",
            kind,
            error,
        )
        return None

    logger.info("Detector: %s (%s).", detector.name, path)
    return detector
