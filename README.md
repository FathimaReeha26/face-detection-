# Face Detection

A Python face detection and recognition project that finds faces with OpenCV's YuNet DNN detector and tells known people from unknown ones with the SFace recognizer. Every detected face gets an inline label bar: a green box with the person's name when they are recognized, or an orange box labelled "Unknown" when they are not.

The detection layer is hybrid. When the machine has a CUDA device, the accurate SCRFD model runs on the GPU through ONNX Runtime; when it does not, the light YuNet model runs on the CPU. The backend is chosen automatically, so nothing has to be configured to run the scripts.

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

   OpenCV 4.x is required for the YuNet detector, the SFace recognizer and the trackers.

4. Optional, only needed for GPU detection. Install ONNX Runtime with the CUDA provider:

   ```powershell
   python -m pip uninstall -y onnxruntime
   python -m pip install onnxruntime-gpu
   ```

   `onnxruntime-gpu` needs a matching CUDA and cuDNN runtime; ONNX Runtime 1.17 and newer ask for CUDA 12.x with cuDNN 9.x. Without it the project keeps working on the CPU backend. The plain `onnxruntime` package can be installed instead if you only want the SCRFD model on the CPU.

   If you install `opencv-contrib-python` instead of `opencv-python`, the tracker automatically switches from MIL to CSRT, which tracks better. The two OpenCV packages cannot be installed side by side.

## Model files

Model files live in `models/`:

- `face_detection_yunet_2023mar.onnx` — YuNet face detector, the CPU backend.
- `scrfd_10g.onnx` — SCRFD face detector, the GPU backend.
- `face_recognition_sface_2021dec.onnx` — SFace face recognizer.

`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` are the old ResNet-10 SSD files. The detector does not use them any more; they are kept in the folder so the old weights stay available.

Where to get the files again:

- <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface>
- <https://huggingface.co/deepghs/insightface/blob/main/buffalo_l/det_10g.onnx> (SCRFD 10G, also part of the [buffalo_l model pack](https://github.com/deepinsight/insightface/releases))

### Licenses

| File | License | Notes |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | MIT (opencv_zoo) | Free for any use. |
| `face_recognition_sface_2021dec.onnx` | MIT (opencv_zoo) | Free for any use. |
| `scrfd_10g.onnx` | **Non-commercial research use only** | The SCRFD *code* is MIT, but the pretrained weights come from the InsightFace model zoo and are published under a non-commercial license. Do not use them in a commercial product. Train your own SCRFD (MIT code) if you need a commercial model. |
| RetinaFace weights, if you use them | **Non-commercial research use only** | The RetinaFace code is MIT, but the pretrained weights are trained on WIDER FACE, which is licensed for non-commercial research only. |

## Detection backends

The backend is picked at startup, after a single device check:

1. ONNX Runtime reports a CUDA provider and `models/scrfd_10g.onnx` is present, so SCRFD runs on the GPU.
2. Otherwise YuNet runs on the CPU.

If the GPU backend fails while it runs, for example after a CUDA out of memory error or a driver problem, the detector logs the reason, switches to the CPU backend and keeps going for the rest of the session.

Two environment variables override the automatic choice, which is useful for testing:

| Variable | Values | Meaning |
| --- | --- | --- |
| `FACE_DETECTOR_MODE` | `auto` (default), `gpu`, `cpu` | Force the GPU or the CPU path even when the other one would be picked. |
| `FACE_DETECTOR_MODEL` | `auto` (default), `scrfd`, `retinaface`, `yunet` | Force one specific model. A model that is requested explicitly also runs on the CPU when no CUDA provider is available. |
| `FACE_DETECTOR_MODELS_DIR` | folder path (default `models`) | Where the model files live. |

```powershell
$env:FACE_DETECTOR_MODE = "cpu"
python src/detect_image.py
```

The chosen backend and every fallback are written to the log, so the scripts and the launcher show which detector is running.

## Face recognition

Recognition compares faces against `models/known_faces.json`. Enroll people before running the detectors:

Camera (look at the camera, press `SPACE` to capture each sample, `Q` or `ESC` to finish and save):

```powershell
python src/enroll.py --name "Your name"
```

From a folder of face images (one face per image):

```powershell
python src/enroll.py --name "Alice" --images C:\path\to\images
```

List enrolled people:

```powershell
python src/enroll.py --list
```

Capture 10–20 samples from slightly different angles for reliable matching. `known_faces.json` stays on your machine — it is ignored by Git.

## Detect faces in images

1. Add test images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) to `data/`.
2. From the project root, run:

   ```powershell
   python src/detect_image.py
   ```

3. The script prints which detector it loaded, the enrolled people, the face count and labels per image, and a total, then saves an annotated copy of each image to `outputs/result_<image-name>.jpg`.

If `data/` contains no images, the script prints a clear message telling you to add images to `data/`.

## Live camera

From the project root, run:

```powershell
python src/detect_camera.py
```

A window opens showing the webcam feed with a green box and a name label for known faces, an orange "Unknown" label for everyone else, and a live count (`Faces: N (known: K)`).

- Press `S` to save the current frame to `outputs/camera_<timestamp>.jpg`.
- Press `Q` or `ESC` to quit.
- If the camera cannot be opened, the script prints a clear message about checking the camera.

On the GPU backend every frame is detected. On the CPU backend the detector runs every fourth frame and a tracker (CSRT, KCF or MIL) keeps the faces alive in between, so the camera keeps a steady frame rate on a slower machine. A tracked face keeps the name it was given, so the labels do not flicker.

## Project layout

- `data/` — test images (for example, `group.jpg`).
- `models/` — detector and recognizer models plus the local `known_faces.json` database.
## Presentation attack check

`src/detect_liveness_webcam.py` runs a heuristic liveness check on top of the
DNN face detector. It looks at two signals:

- **Blinks.** MediaPipe Face Landmarker gives the eye landmarks, from which
  the eye aspect ratio (EAR) is computed. A blink is counted when the EAR
  drops below 0.21 for at least two consecutive frames and then rises again.
- **Head movement.** The nose tip position is tracked over the last three
  seconds; the check passes when it moved more than 15 pixels in that window.
- **Screen replay.** A Fourier-spectrum check flags faces shown on a phone or
  monitor as a suspected spoof.
- **Anti-spoof model.** A MiniFASNet model scores each face crop as real or
  fake. Faces scoring below `SPOOF_THRESHOLD` (default 0.5) are flagged as
  suspected spoofs. The model weights are not shipped; place an ONNX export
  in `models/antispoof/` (see `models/antispoof/README.md`).

This is a heuristic and can still be fooled by high-quality video or
deepfakes. Do not rely on it as a security control.

The status is `CHECKING` during the first 10 seconds, `LIVE` when a blink was
seen in the last 10 seconds or head movement was detected, and
`SUSPECTED SPOOF` when neither happened for 10 seconds after the check
started. The box is green for `LIVE`, yellow for `CHECKING` and red for
`SUSPECTED SPOOF`, with the status and blink count drawn on screen.

This is a heuristic. It can produce false alarms (a person who simply stares
at the camera is flagged) and it can be fooled by a video replay of a
blinking, moving person. Do not rely on it as a security control.

Run it with:

```powershell
python src/detect_liveness_webcam.py
```

Press `q` to quit. The script needs the ResNet-10 SSD model files
(`models/deploy.prototxt`, `models/res10_300x300_ssd_iter_140000.caffemodel`)
and the MediaPipe model (`models/face_landmarker.task`), plus `mediapipe`
installed in the environment:

```powershell
python -m pip install mediapipe
```

- `src/detection/` — the hybrid detection layer: device check, backends, factory and tracker.
- `outputs/` — generated results (ignored by Git).

## HTTP API

The project also exposes a REST API for other applications:

```powershell
python src/api_server.py
```

See `docs/API.md` for endpoints, authentication, and client examples.
