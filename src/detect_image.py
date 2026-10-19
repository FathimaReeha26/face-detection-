import os
import sys

import cv2

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images() -> list[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        os.path.join(DATA_DIR, name)
        for name in os.listdir(DATA_DIR)
        if name.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(DATA_DIR, name))
    )


def output_path(image_path: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(OUTPUT_DIR, f"result_{stem}.jpg")


def main() -> None:
    images = list_images()
    if not images:
        print(
            f"No images found in {DATA_DIR}/. "
            f"Add images to the {DATA_DIR}/ folder and run again."
        )
        sys.exit(1)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print(f"Could not load the Haar cascade from {CASCADE_PATH}.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_faces = 0
    for image_path in images:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipped {image_path}: could not read the file as an image.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        destination = output_path(image_path)
        if not cv2.imwrite(destination, image):
            print(f"Skipped {image_path}: could not save to {destination}.")
            continue

        print(f"{image_path}: {len(faces)} face(s) -> {destination}")
        total_faces += len(faces)

    print(f"Processed {len(images)} image(s), {total_faces} face(s) found.")


if __name__ == "__main__":
    main()
