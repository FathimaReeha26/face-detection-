import os
import sys
import time

import cv2

CAMERA_INDEX = 0
OUTPUT_DIR = "outputs"
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
WINDOW_TITLE = "Face detection - press S to save, Q to quit"


def main() -> None:
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print(f"Could not load the Haar cascade from {CASCADE_PATH}.")
        sys.exit(1)

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(
            f"Could not open camera {CAMERA_INDEX}. "
            "Check that a camera is connected and not used by another app."
        )
        sys.exit(1)

    print("Camera started. Press S to save a frame to outputs/, Q or ESC to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame. The camera may have been disconnected.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(30, 30),
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
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
