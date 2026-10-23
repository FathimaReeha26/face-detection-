"""Frame sampling and tracking for the live camera path.

The GPU backend is fast enough to detect on every frame. The CPU backend is
not, so it detects every few frames and a lightweight tracker keeps the
boxes alive in between. The tracker is used for the camera only; single
images are always detected completely.
"""

import cv2

from .logging_setup import get_logger
from .postprocess import intersection_over_union
from .types import Box, Detection

# CSRT is the best of the three, KCF is a good second and MIL is the one
# that ships with the plain opencv-python build.
TRACKER_FACTORIES = ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create")
MATCH_IOU_THRESHOLD = 0.3


class MotionTracker:
    """Last resort tracker used when OpenCV has no tracker at all.

    It keeps the last box and repeats the movement it saw last, which keeps
    the box roughly in place without any extra dependency.
    """

    def __init__(self) -> None:
        self.box: Box | None = None
        self.velocity = (0.0, 0.0)

    def init(self, frame, box) -> None:
        self.box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        self.velocity = (0.0, 0.0)

    def update(self, frame):
        if self.box is None:
            return False, (0, 0, 0, 0)
        x, y, w, h = self.box
        velocity_x, velocity_y = self.velocity
        self.box = (
            int(round(x + velocity_x)),
            int(round(y + velocity_y)),
            w,
            h,
        )
        return True, self.box

    def move_to(self, box) -> None:
        """Move the tracker onto a new box and remember the movement."""
        new_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        if self.box is not None:
            self.velocity = (
                (new_box[0] - self.box[0]) / 2,
                (new_box[1] - self.box[1]) / 2,
            )
        self.box = new_box


class Track:
    """One face followed over time."""

    def __init__(self, track_id: int, box: Box, tracker, frame) -> None:
        self.track_id = track_id
        self.box = box
        self.tracker = tracker
        self.confidence = 0.0
        self.tracker.init(frame, box)

    def move_to(self, box: Box, frame) -> None:
        """Follow a new detection and remember its confidence."""
        self.box = box
        if isinstance(self.tracker, MotionTracker):
            self.tracker.move_to(box)
        else:
            self.tracker.init(frame, box)

    def update(self, frame) -> Box | None:
        """Move the box along with the face, or give up when it is lost."""
        found, box = self.tracker.update(frame)
        if not found:
            return None
        x, y, w, h = (int(round(value)) for value in box)
        if w <= 0 or h <= 0:
            return None
        self.box = (x, y, w, h)
        return self.box


class TrackedFaceDetector:
    """Runs the detector every few frames and tracks the faces in between.

    The detector decides how often a real detection happens: the GPU backend
    detects on every frame, the CPU backend every few frames. Frames in
    between are filled with the tracked boxes.
    """

    def __init__(self, detector, interval: int | None = None) -> None:
        self.detector = detector
        self.interval = interval
        self._frame_index = 0
        self._tracks: list[Track] = []
        self._next_track_id = 1
        self._tracker_factory = _find_tracker_factory()

    def detect(self, frame: "cv2.Mat") -> list[Detection]:
        self._frame_index += 1
        interval = self.interval or getattr(self.detector, "frame_interval", 1)
        if interval <= 1 or self._frame_index % interval == 1:
            detections = self.detector.detect(frame)
            self._sync_tracks(detections, frame)
            return detections
        return self._advance_tracks(frame)

    def _sync_tracks(self, detections: list[Detection], frame) -> None:
        """Match the fresh detections with the tracks and drop the rest."""
        tracks = []
        for detection in detections:
            track = self._match_track(detection.box)
            if track is None:
                track = Track(
                    self._next_track_id, detection.box, self._new_tracker(), frame
                )
                self._next_track_id += 1
            else:
                track.move_to(detection.box, frame)
            track.confidence = detection.confidence
            detection.track_id = track.track_id
            tracks.append(track)
        self._tracks = tracks

    def _match_track(self, box: Box) -> Track | None:
        """Return the track that overlaps the box the most."""
        best = None
        best_score = MATCH_IOU_THRESHOLD
        for track in self._tracks:
            score = intersection_over_union(track.box, box)
            if score > best_score:
                best = track
                best_score = score
        return best

    def _advance_tracks(self, frame) -> list[Detection]:
        """Move every track along with the frame and report the boxes."""
        detections = []
        for track in list(self._tracks):
            box = track.update(frame)
            if box is None:
                get_logger().info("Lost the track of face %s.", track.track_id)
                self._tracks.remove(track)
                continue
            detections.append(
                Detection(
                    box=box,
                    row=None,
                    confidence=track.confidence,
                    landmarks=None,
                    track_id=track.track_id,
                    tracked=True,
                )
            )
        return detections

    def _new_tracker(self):
        return self._tracker_factory() if self._tracker_factory else MotionTracker()


def _find_tracker_factory():
    """Return a factory for the best tracker this OpenCV build offers."""
    logger = get_logger()
    for name in TRACKER_FACTORIES:
        factory = getattr(cv2, name, None)
        if factory is None:
            continue
        try:
            factory()
        except cv2.error:
            continue
        logger.info("Tracking with %s.", name.replace("_create", ""))
        return factory
    logger.warning(
        "No OpenCV tracker is available; using the built-in motion tracker instead."
    )
    return None
