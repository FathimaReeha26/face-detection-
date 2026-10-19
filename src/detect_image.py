import os
import sys

import cv2

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 8
MIN_SIZE = (30, 30)


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

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print(f"Could not load the Haar cascade from {CASCADE_PATH}.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_faces = 0
    for name in images:
        image_path = os.path.join(DATA_DIR, name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipped {image_path}: could not read the file as an image.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE,
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        destination = os.path.join(OUTPUT_DIR, f"result_{os.path.splitext(name)[0]}.jpg")
        if not cv2.imwrite(destination, image):
            print(f"Skipped {image_path}: could not save to {destination}.")
            continue

        print(f"{image_path}: {len(faces)} face(s) -> {destination}")
        total_faces += len(faces)

    print(f"Processed {len(images)} image(s), {total_faces} face(s) found.")


if __name__ == "__main__":
    main()
