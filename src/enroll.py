import argparse
import json
import os
import sys

import cv2

from face_detector import FaceDetector
from face_recognizer import DB_PATH, FaceRecognizer

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_SAMPLES_PER_PERSON = 100
CAMERA_INDEX = 0
WINDOW_TITLE = "Enroll face - SPACE to capture, Q to finish"


def load_db() -> dict:
    if not os.path.isfile(DB_PATH):
        return {}
    try:
        with open(DB_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        print(f"Could not read {DB_PATH}. Starting a new database.")
        return {}


def save_db(db: dict) -> None:
    dirname = os.path.dirname(DB_PATH)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(DB_PATH, "w", encoding="utf-8") as handle:
        json.dump(db, handle, indent=2)


def largest_face(detector: FaceDetector, image: "cv2.Mat"):
    faces = detector.detect(image)
    if not faces:
        return None
    return max(faces, key=lambda detection: detection.box[2] * detection.box[3])


def collect_from_images(recognizer: FaceRecognizer, detector: FaceDetector, folder: str) -> list[list[float]]:
    if not os.path.isdir(folder):
        print(f"Missing image folder {folder}.")
        sys.exit(1)
    names = sorted(
        name for name in os.listdir(folder) if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not names:
        print(f"No images found in {folder}.")
        sys.exit(1)

    features = []
    for name in names:
        path = os.path.join(folder, name)
        image = cv2.imread(path)
        if image is None:
            print(f"Skipped {path}: could not read the file as an image.")
            continue
        detection = largest_face(detector, image)
        if detection is None:
            print(f"Skipped {path}: no face detected.")
            continue
        embedding = recognizer.feature(image, detection)
        if embedding is None:
            continue
        features.append(embedding.flatten().tolist())
        print(f"Captured {len(features)} sample(s) from {path}")
    return features


def collect_from_camera(recognizer: FaceRecognizer, detector: FaceDetector) -> list[list[float]]:
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(
            f"Could not open camera {CAMERA_INDEX}. "
            "Check that a camera is connected and not used by another app."
        )
        sys.exit(1)

    print("Look at the camera, then press SPACE to capture a sample. Press Q or ESC to finish.")
    features = []
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame. The camera may have been disconnected.")
                break

            detection = largest_face(detector, frame)
            if detection is not None:
                x, y, w, h = detection.box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 170, 0), 2)
            cv2.putText(
                frame,
                f"Samples: {len(features)}",
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
            if key == ord(" ") and detection is not None:
                embedding = recognizer.feature(frame, detection)
                if embedding is not None:
                    features.append(embedding.flatten().tolist())
                    print(f"Captured sample {len(features)}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a known face for recognition.")
    parser.add_argument("--name", help="Name of the person to enroll.")
    parser.add_argument("--images", help="Folder of face images to enroll instead of the camera.")
    parser.add_argument("--list", action="store_true", help="List known people and exit.")
    args = parser.parse_args()

    if args.list:
        db = load_db()
        if not db:
            print("No known faces yet. Run this script to enroll someone.")
            return
        for name, vectors in sorted(db.items()):
            print(f"{name}: {len(vectors)} sample(s)")
        return

    try:
        detector = FaceDetector()
        recognizer = FaceRecognizer()
    except RuntimeError as error:
        print(error)
        sys.exit(1)

    if not recognizer.enabled:
        print(
            "Recognition model models/face_recognition_sface_2021dec.onnx is missing. "
            "Download it from "
            "https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface"
        )
        sys.exit(1)

    name = args.name or input("Name for this person: ").strip()
    if not name:
        print("A name is required to enroll a face.")
        sys.exit(1)

    print(f"Detector: {detector.name}")
    if args.images:
        features = collect_from_images(recognizer, detector, args.images)
    else:
        features = collect_from_camera(recognizer, detector)

    if not features:
        print(f"No samples captured, so {name} was not enrolled.")
        sys.exit(1)

    db = load_db()
    existing = db.get(name, [])
    combined = (existing + features)[-MAX_SAMPLES_PER_PERSON:]
    db[name] = combined
    save_db(db)

    print(f"Enrolled {name} with {len(features)} new sample(s), {len(combined)} total.")
    print(f"Known people: {', '.join(sorted(db))}")


if __name__ == "__main__":
    main()
