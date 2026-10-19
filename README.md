# Face Detection

A Python face detection project that uses OpenCV's built-in Haar cascade to find faces in images, draw green rectangles around them, and save annotated copies of the results. It is a starting point for a larger face detection and recognition project, with space for test images and DNN model files as the project grows.

## Setup

1. Create a virtual environment named `face_env`:

   ```powershell
   python -m venv face_env
   ```

2. Activate it.

   PowerShell:

   ```powershell
   .\face_env\Scripts\Activate.ps1
   ```

   Command Prompt:

   ```bat
   face_env\Scripts\activate.bat
   ```

3. Install the dependencies:

   ```powershell
   python -m pip install "opencv-python<5" numpy matplotlib
   ```

   OpenCV 4.x is required because it bundles the Haar cascade XML files under `cv2.data.haarcascades`. OpenCV 5 no longer ships them.

## Usage

1. Add test images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) to `data/`.
2. From the project root, run:

   ```powershell
   python src/detect_image.py
   ```

3. The script prints the face count per image and a total, then saves an annotated copy of each image to `outputs/result_<image-name>.jpg`.

If `data/` contains no images, the script prints a clear message telling you to add images to `data/`.

## Live camera

From the project root, run:

```powershell
python src/detect_camera.py
```

A window opens showing the webcam feed with a green box and a live face count for every detected face.

- Press `S` to save the current frame to `outputs/camera_<timestamp>.jpg`.
- Press `Q` or `ESC` to quit.
- If the camera cannot be opened, the script prints a clear message about checking the camera.

## Project layout

- `data/` — test images (for example, `group.jpg`).
- `models/` — DNN model files (for example, `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`).
- `src/` — Python source files.
- `outputs/` — generated results (ignored by Git).

DNN face detection model files can be downloaded from the OpenCV GitHub repository:
<https://github.com/opencv/opencv/tree/4.x/samples/dnn/face_detector>
