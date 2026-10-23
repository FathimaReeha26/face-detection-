"""Face detection for the scripts.

The detection layer is hybrid: a GPU backend (SCRFD through ONNX Runtime with
the CUDA provider) when the machine has a CUDA device, and a light CPU
backend (YuNet) when it does not. The backend is picked automatically, so
nothing has to be configured to run the scripts.

This module keeps the interface the rest of the project already uses:
``FaceDetector().detect(image)`` returns a list of Detection objects with a
pixel box and the landmark row the SFace recognizer aligns on.
"""

from detection.backends.base import BaseDetector
from detection.factory import create_cpu_detector, create_detector
from detection.logging_setup import configure_logging
from detection.types import Detection

__all__ = ["Detection", "FaceDetector"]


class FaceDetector:
    """Detects faces with the backend that fits the machine.

    If the GPU backend fails while it runs, for example after a CUDA out of
    memory error or a driver problem, the detector logs the reason, switches
    to the CPU backend and keeps working for the rest of the session.
    """

    def __init__(self) -> None:
        logger = configure_logging()
        self._backend: BaseDetector = create_detector()
        self._fallback: BaseDetector | None = None
        self.name = self._backend.name
        logger.info("Face detector ready: %s.", self.name)

    @property
    def backend(self) -> BaseDetector:
        """The backend that is currently in use."""
        return self._backend

    @property
    def frame_interval(self) -> int:
        """How many frames of a live feed pass between two detections."""
        return self._backend.frame_interval

    @property
    def uses_gpu(self) -> bool:
        """True while the GPU backend is in use."""
        return self._backend.uses_gpu

    def detect(self, image: "cv2.Mat") -> list[Detection]:
        """Return every face found in the image."""
        try:
            return self._backend.detect(image)
        except Exception as error:
            if not self._backend.supports_fallback:
                raise
            logger = configure_logging()
            logger.warning(
                "The %s backend failed (%s: %s). Falling back to the CPU backend "
                "for the rest of this session.",
                self._backend.name,
                type(error).__name__,
                error,
            )
            self._backend = self._cpu_backend()
            self.name = self._backend.name
            return self._backend.detect(image)

    def _cpu_backend(self) -> BaseDetector:
        """Build the CPU backend, reusing it if it is already running."""
        if self._fallback is None:
            self._fallback = create_cpu_detector()
        return self._fallback
