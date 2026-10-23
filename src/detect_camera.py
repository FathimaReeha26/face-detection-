import os
import sys
import time

import cv2

from detection import TrackedFaceDetector
from face_detector import FaceDetector
from face_recognizer import UNKNOWN_LABEL, FaceRecognizer, draw_label

CAMERA_INDEX = 0
OUTPUT_DIR = "outputs"
WINDOW_TITLE = "Face detection - press S to save, Q to quit"


def main() -> None:
    try:
        detector = FaceDetector()
        recognizer = FaceRecognizer()
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

    # The GPU backend detects on every frame. The CPU backend is slower, so
    # it detects every few frames and a tracker keeps the faces alive in
    # between; the name of a tracked face is reused instead of recognised
    # again on every frame.
    runner = TrackedFaceDetector(detector)
    labels: dict[int, str] = {}

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame. The camera may have been disconnected.")
                break

            faces = runner.detect(frame)
            current: dict[int, str] = {}
            known = 0
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
                draw_label(frame, detection.box, label)
            labels = current

            status = f"Faces: {len(faces)}"
            if faces:
                status += f" (known: {known})"
            cv2.putText(
                frame,
                status,
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
                destination = os.path.join(OUTPUT_DIR, f"camera_{int(time.time())}.jpg")
                if cv2.imwrite(destination, frame):
                    print(f"Saved {destination}")
                else:
                    print(f"Could not save {destination}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
