"""Webcam demo for the presentation-attack (liveness) check.

Detects the largest face with the project's DNN detector, crops it, and runs
the blink and head-movement liveness check on the crop. The box is green for
LIVE, yellow for CHECKING and red for SUSPECTED SPOOF.
"""

import os
import sys

import cv2

try:
    from face_detector import FaceDetector as ProjectDetector
except Exception:  # pragma: no cover - fallback for older project layouts
    ProjectDetector = None

from antispoof import AntiSpoofChecker, SPOOF_THRESHOLD
from dnn_detector import DnnDetector
from liveness import LivenessChecker


def _make_detector():
    if ProjectDetector is not None:
        try:
            return ProjectDetector()
        except Exception:
            pass
    return DnnDetector()


def main() -> None:
    proto = os.path.join("models", "deploy.prototxt")
    weights = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")
    if not os.path.isfile(proto) or not os.path.isfile(weights):
        print(
            "Missing model files: models/deploy.prototxt and/or "
            "models/res10_300x300_ssd_iter_140000.caffemodel."
        )
        sys.exit(1)

    try:
        detector = _make_detector()
    except RuntimeError as error:
        print(error)
        sys.exit(1)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print(
            "Could not open the webcam. Check that a camera is connected and "
            "not used by another app."
        )
        sys.exit(1)

    try:
        checker = LivenessChecker()
    except RuntimeError as error:
        print(error)
        camera.release()
        sys.exit(1)

    try:
        antispoof = AntiSpoofChecker()
    except RuntimeError as error:
        print(error)
        checker.close()
        camera.release()
        sys.exit(1)

    window = "Liveness check - press q to quit"
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame. The camera may have been disconnected.")
                break

            detections = _boxes(detector, frame)
            if detections:
                x, y, w, h = max(detections, key=lambda box: box[2] * box[3])
                height, width = frame.shape[:2]
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(1, min(w, width - x))
                h = max(1, min(h, height - y))
                crop = frame[y : y + h, x : x + w]
                result = checker.check(crop if crop.size else frame)
                status = result["status"]
                antispoof_score = antispoof.score(frame, (x, y, w, h))
                if antispoof_score < SPOOF_THRESHOLD:
                    status = "SUSPECTED SPOOF"
                color = _color(status)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{status}  blinks={result['blinks']}  antispoof={antispoof_score:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        checker.close()
        camera.release()
        cv2.destroyAllWindows()


def _boxes(detector, frame):
    """Return a list of (x, y, w, h) boxes from any detector."""
    detections = detector.detect(frame)
    boxes = []
    for detection in detections:
        if hasattr(detection, "box"):
            boxes.append(tuple(int(value) for value in detection.box))
        else:
            boxes.append(tuple(int(value) for value in detection))
    return boxes


def _color(status):
    if status == "LIVE":
        return (0, 200, 0)
    if status == "CHECKING":
        return (0, 200, 200)
    return (0, 0, 200)


if __name__ == "__main__":
    main()
