"""The interface every detection backend implements."""

import abc

import cv2

from ..types import Detection


class BaseDetector(abc.ABC):
    """Common interface of the detection backends."""

    #: Human readable name, shown by the scripts when they start.
    name: str = ""
    #: How many frames of a live feed pass between two detections.
    frame_interval: int = 1
    #: True when the backend runs on the GPU.
    uses_gpu: bool = False
    #: True when another backend can take over if this one fails.
    supports_fallback: bool = False

    @abc.abstractmethod
    def detect(self, image: "cv2.Mat") -> list[Detection]:
        """Return every face found in the image."""
