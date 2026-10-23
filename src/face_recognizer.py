import json
import os

import cv2
import numpy as np

from pathlib import Path

from face_detector import Detection

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODELS_DIR = str(_ROOT / "models") if not os.path.isdir("models") and (_ROOT / "models").is_dir() else "models"
MODEL_PATH = os.environ.get("FACE_RECOGNIZER_MODEL", os.path.join(_DEFAULT_MODELS_DIR, "face_recognition_sface_2021dec.onnx"))
DB_PATH = os.environ.get("FACE_DB_PATH", os.path.join(_DEFAULT_MODELS_DIR, "known_faces.json"))
MATCH_THRESHOLD = 0.363
UNKNOWN_LABEL = "Unknown"
FEATURE_SIZE = (112, 112)
KNOWN_COLOR = (0, 170, 0)
UNKNOWN_COLOR = (0, 100, 220)
TEXT_COLOR = (255, 255, 255)


class FaceRecognizer:
    """SFace embeddings compared against the known-faces database.

    Each known person is stored as the centroid of the enrolled samples in
    models/known_faces.json. A face is labelled with the best matching name
    when the cosine similarity clears MATCH_THRESHOLD, otherwise "Unknown".
    Faces aligned with YuNet landmarks give the most reliable matches.
    """

    def __init__(self) -> None:
        self.model = None
        self.names: list[str] = []
        self.centroids: list[np.ndarray] = []

        if os.path.isfile(MODEL_PATH):
            self.model = cv2.FaceRecognizerSF.create(MODEL_PATH, "")
        self.reload()

    def reload(self) -> None:
        """Reload the known faces database from disk."""
        self.names.clear()
        self.centroids.clear()
        self._load()

    @property
    def enabled(self) -> bool:
        return self.model is not None

    @property
    def has_known_faces(self) -> bool:
        return bool(self.names)

    @property
    def people(self) -> list[str]:
        return list(self.names)

    def _load(self) -> None:
        if not os.path.isfile(DB_PATH):
            return
        try:
            with open(DB_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        for name, vectors in data.items():
            if not vectors:
                continue
            array = np.asarray(vectors, dtype=np.float32).reshape(-1, 128)
            self.names.append(name)
            self.centroids.append(array.mean(axis=0, keepdims=True))

    def feature(self, image: "cv2.Mat", detection: Detection):
        if self.model is None:
            return None
        if detection.row is not None:
            aligned = self.model.alignCrop(image, detection.row)
            if aligned is not None and aligned.size:
                return self.model.feature(aligned)

        x, y, w, h = detection.box
        height, width = image.shape[:2]
        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return self.model.feature(cv2.resize(crop, FEATURE_SIZE))

    def identify(self, image: "cv2.Mat", detection: Detection) -> tuple[str, float]:
        if self.model is None or not self.centroids:
            return UNKNOWN_LABEL, 0.0
        embedding = self.feature(image, detection)
        if embedding is None:
            return UNKNOWN_LABEL, 0.0
        best_name = UNKNOWN_LABEL
        best_score = -1.0
        for name, centroid in zip(self.names, self.centroids):
            score = float(self.model.match(embedding, centroid, cv2.FaceRecognizerSF_FR_COSINE))
            if score > best_score:
                best_score = score
                best_name = name
        if best_score >= MATCH_THRESHOLD:
            return best_name, best_score
        return UNKNOWN_LABEL, best_score


def draw_label(
    frame: "cv2.Mat",
    box: tuple[int, int, int, int],
    label: str,
) -> None:
    """Draw the box plus an inline label bar attached to its top edge."""
    x, y, w, h = box
    known = label != UNKNOWN_LABEL
    color = KNOWN_COLOR if known else UNKNOWN_COLOR
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    text = label
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    bar_h = text_h + baseline + 8
    bar_y1 = y - bar_h if y - bar_h >= 0 else y + h
    bar_y2 = bar_y1 + bar_h
    cv2.rectangle(frame, (x, bar_y1), (x + text_w + 12, bar_y2), color, cv2.FILLED)
    cv2.putText(
        frame,
        text,
        (x + 6, bar_y2 - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
