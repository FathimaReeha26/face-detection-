"""Data types shared by the detection layer."""

from dataclasses import dataclass

import numpy as np

Box = tuple[int, int, int, int]
Landmark = tuple[float, float]


@dataclass
class Detection:
    """A detected face.

    box holds the face in absolute pixel coordinates as (x, y, w, h). The
    five landmarks are ordered the way OpenCV's SFace alignment expects them:
    right eye, left eye, nose, right mouth corner, left mouth corner.

    row is the matching 15 value YuNet row. It is what
    cv2.FaceRecognizerSF.alignCrop consumes, so it is kept for the
    recognizer; row is None when the backend does not produce landmarks.
    """

    box: Box
    row: "np.ndarray | None" = None
    confidence: float = 0.0
    landmarks: tuple[Landmark, ...] | None = None
    track_id: int | None = None
    tracked: bool = False
