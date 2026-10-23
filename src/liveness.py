"""Presentation-attack (liveness) check built on MediaPipe Face Landmarker.

The check is a heuristic: a real face blinks and moves its head, while a
photo or a still video replay does neither. It is not a security guarantee -
a video replay of a blinking person can fool it, and a person who simply
stares at the camera for ten seconds is flagged as a suspected spoof.
"""

import os
import time
from collections import deque

import cv2
import numpy as np

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
EAR_THRESHOLD = 0.21
BLINK_MIN_CONSEC_FRAMES = 2
HEAD_MOVEMENT_PIXELS = 15
HEAD_MOVEMENT_WINDOW_SECONDS = 3.0
LIVENESS_WINDOW_SECONDS = 10.0
CHECKING_PERIOD_SECONDS = 10.0
# A face shown on a phone/monitor produces strong periodic peaks in the
# Fourier spectrum (moiré / pixel grid). Real faces do not, so a high
# peak-to-mean ratio flags a screen replay as a suspected spoof.
SCREEN_PEAK_RATIO_THRESHOLD = 100.0

# Precomputed window and center mask for FFT screen replay analysis (128x128)
_HAMMING_WINDOW_128 = np.outer(np.hamming(128), np.hamming(128)).astype(np.float32)
_yy_128, _xx_128 = np.ogrid[:128, :128]
_CENTER_MASK_128 = ((_xx_128 - 64) ** 2 + (_yy_128 - 64) ** 2) <= 10 ** 2

# MediaPipe Face Landmarker indices (478-point model).
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
NOSE_TIP_INDEX = 1

DEFAULT_MODEL_PATH = os.path.join("models", "face_landmarker.task")


def eye_aspect_ratio(landmarks, eye_indices):
    """Compute the eye aspect ratio from six eye landmark points.

    landmarks is any sequence of points, each either an (x, y) pair or an
    object with .x and .y attributes. eye_indices holds six indices into
    landmarks in the order outer, top-outer, top-inner, inner, bottom-inner,
    bottom-outer.
    """
    points = []
    for index in eye_indices:
        landmark = landmarks[index]
        if hasattr(landmark, "x"):
            points.append((float(landmark.x), float(landmark.y)))
        else:
            points.append((float(landmark[0]), float(landmark[1])))

    p1, p2, p3, p4, p5, p6 = points
    vertical_a = _distance(p2, p6)
    vertical_b = _distance(p3, p5)
    horizontal = _distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def _distance(first, second):
    return float(np.hypot(first[0] - second[0], first[1] - second[1]))


class BlinkMonitor:
    """Counts blinks from a stream of eye aspect ratios."""

    def __init__(self, threshold=EAR_THRESHOLD, min_consec_frames=BLINK_MIN_CONSEC_FRAMES):
        self.threshold = threshold
        self.min_consec_frames = min_consec_frames
        self.blinks = 0
        self.last_blink_time = None
        self._consec_below = 0

    def update(self, ear, now=None):
        """Feed one EAR value. Returns True when a blink just finished."""
        if now is None:
            now = time.monotonic()
        if ear is None:
            self._consec_below = 0
            return False
        if ear < self.threshold:
            self._consec_below += 1
            return False
        if self._consec_below >= self.min_consec_frames:
            self.blinks += 1
            self.last_blink_time = now
            self._consec_below = 0
            return True
        self._consec_below = 0
        return False


class LivenessChecker:
    """Runs MediaPipe Face Landmarker and reports liveness signals."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        if not os.path.isfile(model_path):
            raise RuntimeError(
                f"Face landmarker model not found: {model_path}. Download "
                "face_landmarker.task from "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task and put it "
                "in the models folder."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.blink_monitor = BlinkMonitor()
        self.nose_history = deque()
        self.start_time = None
        self.screen_like = False

    def check(self, frame):
        """Analyse one frame and return the liveness dict."""
        now = time.monotonic()
        if self.start_time is None:
            self.start_time = now

        landmarks = self._detect(frame)
        if landmarks is not None:
            height, width = frame.shape[:2]
            points = [(point.x * width, point.y * height) for point in landmarks]
            left_ear = eye_aspect_ratio(points, LEFT_EYE)
            right_ear = eye_aspect_ratio(points, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            self.blink_monitor.update(ear, now)
            nose_x, nose_y = points[NOSE_TIP_INDEX]
            self.nose_history.append((now, nose_x, nose_y))

        self.screen_like = self._is_screen(frame)

        cutoff = now - HEAD_MOVEMENT_WINDOW_SECONDS
        while self.nose_history and self.nose_history[0][0] < cutoff:
            self.nose_history.popleft()

        head_moved = False
        if len(self.nose_history) >= 2:
            xs = [entry[1] for entry in self.nose_history]
            ys = [entry[2] for entry in self.nose_history]
            head_moved = (
                (max(xs) - min(xs) > HEAD_MOVEMENT_PIXELS)
                or (max(ys) - min(ys) > HEAD_MOVEMENT_PIXELS)
            )

        elapsed = now - self.start_time
        if self.blink_monitor.last_blink_time is None:
            seconds_since_blink = float("inf")
        else:
            seconds_since_blink = now - self.blink_monitor.last_blink_time

        if elapsed < CHECKING_PERIOD_SECONDS:
            status = "CHECKING"
        elif self.screen_like:
            status = "SUSPECTED SPOOF"
        elif (
            seconds_since_blink <= LIVENESS_WINDOW_SECONDS or head_moved
        ):
            status = "LIVE"
        else:
            status = "SUSPECTED SPOOF"

        return {
            "blinks": self.blink_monitor.blinks,
            "seconds_since_blink": seconds_since_blink,
            "head_moved": head_moved,
            "screen_like": self.screen_like,
            "status": status,
        }

    def _detect(self, frame):
        if frame is None or frame.size == 0:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def _is_screen(self, frame):
        """Heuristic screen-replay detection using the Fourier spectrum."""
        if frame is None or frame.size == 0:
            return False
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return False
        gray = cv2.resize(gray, (128, 128)).astype(np.float32)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray * _HAMMING_WINDOW_128)))
        masked = spectrum.copy()
        masked[_CENTER_MASK_128] = 0.0
        mean_value = float(masked.mean())
        if mean_value <= 0:
            return False
        return float(masked.max()) / mean_value > SCREEN_PEAK_RATIO_THRESHOLD

    def close(self):
        self.landmarker.close()
