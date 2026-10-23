"""One-off detection of the compute devices the project can use.

The check runs once per process and is cached, so the frame loop never pays
for it. ONNX Runtime lists the providers it was built with; a CUDA provider
in that list means a GPU backend can be built.
"""

from dataclasses import dataclass
from functools import lru_cache

from .logging_setup import get_logger

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


@dataclass(frozen=True)
class DeviceInfo:
    """What the machine offers for face detection."""

    onnxruntime: bool
    version: str
    providers: tuple[str, ...]
    cuda: bool
    opencv_cuda_devices: int

    @property
    def provider(self) -> str:
        """The provider a GPU backend should ask ONNX Runtime for."""
        return CUDA_PROVIDER if self.cuda else CPU_PROVIDER

    def describe(self) -> str:
        """A short summary for the startup log."""
        providers = ", ".join(self.providers) if self.providers else "none"
        runtime = self.version if self.onnxruntime else "not installed"
        return (
            f"onnxruntime {runtime}; providers: {providers}; "
            f"OpenCV CUDA devices: {self.opencv_cuda_devices}"
        )


@lru_cache(maxsize=1)
def detect_device() -> DeviceInfo:
    """Return the device information, computing it only on the first call."""
    logger = get_logger()

    providers: tuple[str, ...] = ()
    version = ""
    available = False
    try:
        import onnxruntime

        available = True
        version = onnxruntime.__version__
        providers = tuple(onnxruntime.get_available_providers())
    except Exception as error:  # pragma: no cover - depends on the machine
        logger.info("ONNX Runtime is unavailable (%s).", error)

    opencv_cuda_devices = 0
    try:
        import cv2

        if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
            opencv_cuda_devices = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:  # pragma: no cover - depends on the machine
        opencv_cuda_devices = 0

    info = DeviceInfo(
        onnxruntime=available,
        version=version,
        providers=providers,
        cuda=CUDA_PROVIDER in providers,
        opencv_cuda_devices=opencv_cuda_devices,
    )
    logger.info("Device check: %s", info.describe())
    return info
