"""Combined live camera window: face detection + recognition + liveness check.

One window shows the webcam feed with detected/recognised faces and, for the
largest face, the blink and head-movement liveness status. The largest face's
box is green for LIVE, yellow for CHECKING and red for SUSPECTED SPOOF; other
faces use the usual green (known) / orange (unknown) colours.
"""

import os
import sys
import time

import cv2

from detection import TrackedFaceDetector
from face_detector import FaceDetector
from face_recognizer import UNKNOWN_LABEL, FaceRecognizer, draw_label
from liveness import LivenessChecker

CAMERA_INDEX = 0
OUTPUT_DIR = "outputs"
WINDOW_TITLE = "Face detection + liveness - press S to save, Q to quit"


def main() -> None:
    try:
        detector = FaceDetector()
        recognizer = FaceRecognizer()
        checker = LivenessChecker()
    except RuntimeError as error:
        print(error)
        sys.exit(1)

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(
            f"Could not open camera {CAMERA_INDEX}. "
            "Check that a camera is connected and not used by another app."
        )
        sys.exit(1)

    print(f"Camera started. Detector: {detector.name}.")
    if not recognizer.enabled:
        print(
            "Recognition model models/face_recognition_sface_2021dec.onnx is missing; "
            "every face will be labelled Unknown."
        )
    elif not recognizer.has_known_faces:
        print(
            "No known faces enrolled; every face will be labelled Unknown. "
            "Run src/enroll.py to add people."
        )
    else:
        print(f"Known people: {', '.join(recognizer.people)}")
    print("Press S to save a frame to outputs/, Q or ESC to quit.")

    runner = TrackedFaceDetector(detector)
    labels = {}

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame. The camera may have been disconnected.")
                break

            faces = runner.detect(frame)
            current = {}
            known = 0
            largest = None
            for detection in faces:
                label = None
                if detection.tracked and detection.track_id is not None:
                    label = labels.get(detection.track_id)
                if label is None:
                    label, _ = recognizer.identify(frame, detection)
                if detection.track_id is not None:
                    current[detection.track_id] = label
                if label != UNKNOWN_LABEL:
                    known += 1
                if largest is None or detection.box[2] * detection.box[3] > largest.box[2] * largest.box[3]:
                    largest = detection
                draw_label(frame, detection.box, label)
            labels = current

            if largest is not None:
                x, y, w, h = largest.box
                height, width = frame.shape[:2]
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(1, min(w, width - x))
                h = max(1, min(h, height - y))
                crop = frame[y : y + h, x : x + w]
                result = checker.check(crop if crop.size else frame)
                status = result["status"]
                color = (0, 200, 0) if status == "LIVE" else (0, 200, 200) if status == "CHECKING" else (0, 0, 200)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                text = f"{status}  blinks={result['blinks']}"
                cv2.putText(
                    frame,
                    text,
                    (x, max(20, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            status_line = f"Faces: {len(faces)}"
            if faces:
                status_line += f" (known: {known})"
            cv2.putText(
                frame,
                status_line,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 170, 0),
                2,
            )

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                destination = os.path.join(OUTPUT_DIR, f"combined_{int(time.time())}.jpg")
                if cv2.imwrite(destination, frame):
                    print(f"Saved {destination}")
                else:
                    print(f"Could not save {destination}")
    finally:
        checker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()