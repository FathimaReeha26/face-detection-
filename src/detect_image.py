import os
import sys

import cv2

from face_detector import FaceDetector
from face_recognizer import UNKNOWN_LABEL, FaceRecognizer, draw_label

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTENSIONS)


def main() -> None:
    if not os.path.isdir(DATA_DIR):
        print(f"Missing {DATA_DIR}/ folder. Create it and add your images.")
        sys.exit(1)

    entries = sorted(os.listdir(DATA_DIR))
    images = [name for name in entries if is_image(name) and os.path.isfile(os.path.join(DATA_DIR, name))]
    others = [name for name in entries if name not in images]

    for name in others:
        print(f"Ignored {os.path.join(DATA_DIR, name)}: only images inside {DATA_DIR}/ are processed.")

    if not images:
        print(
            f"No images found in {DATA_DIR}/. "
            f"Supported types: {', '.join(IMAGE_EXTENSIONS)}. "
            f"Add an image to {DATA_DIR}/ and run again."
        )
        sys.exit(1)

    try:
        detector = FaceDetector()
        recognizer = FaceRecognizer()
    except RuntimeError as error:
        print(error)
        sys.exit(1)

    print(f"Detector: {detector.name}")
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_faces = 0
    for name in images:
        image_path = os.path.join(DATA_DIR, name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipped {image_path}: could not read the file as an image.")
            continue

        faces = detector.detect(image)
        labels = []
        for detection in faces:
            label, score = recognizer.identify(image, detection)
            draw_label(image, detection.box, label)
            labels.append(label if label == UNKNOWN_LABEL else f"{label} {score:.2f}")

        destination = os.path.join(OUTPUT_DIR, f"result_{os.path.splitext(name)[0]}.jpg")
        if not cv2.imwrite(destination, image):
            print(f"Skipped {image_path}: could not save to {destination}.")
            continue

        summary = ", ".join(labels) if labels else "none"
        print(f"{image_path}: {len(faces)} face(s) [{summary}] -> {destination}")
        total_faces += len(faces)

    print(f"Processed {len(images)} image(s), {total_faces} face(s) found.")


if __name__ == "__main__":
    main()
