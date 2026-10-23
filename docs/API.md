# Face Detection HTTP API

The API exposes face detection and recognition over HTTP so other
applications can integrate with the project. It is implemented in
`src/api_server.py` using only the Python standard library.

## Run

```powershell
python src/api_server.py
python src/api_server.py --host 0.0.0.0 --port 9000
```

Environment variables:

- `FACE_API_HOST`, `FACE_API_PORT` — bind address.
- `FACE_API_KEY` — if set, every request must include `X-API-Key: <key>`.
- `FACE_DETECTOR_MODE`, `FACE_DETECTOR_MODEL`, `FACE_DETECTOR_MODELS_DIR` — detector selection.

Interactive docs: `http://localhost:8000/docs`, OpenAPI spec: `http://localhost:8000/openapi.json`.

## Authentication

When `FACE_API_KEY` is set, send:

```
X-API-Key: <your-key>
```

Otherwise no authentication is required.

## Endpoints

### GET /health

```json
{"status":"ok","service":"face-detection-api","version":"1.0.0"}
```

### GET /info

```json
{
  "detector": "YuNet (DNN, CPU)",
  "recognition_enabled": true,
  "known_people": ["Alice"],
  "api_version": "1.0.0"
}
```

### GET /people

```json
{"people": ["Alice"]}
```

### POST /detect

Detect faces in one or more images.

Input options:

- Multipart form-data with one or more `image` file fields.
- JSON body: `{"image": "<base64>"}` or `{"images": ["<base64>", ...]}`.
- Raw image bytes with an image content type.

Query parameters:

- `annotate=1` — include an `annotated_image` (base64 JPEG) with boxes and labels.

Response (single image):

```json
{
  "filename": "photo.jpg",
  "width": 1280,
  "height": 720,
  "count": 1,
  "known_count": 1,
  "detector": "YuNet (DNN, CPU)",
  "faces": [
    {
      "box": {"x": 10, "y": 20, "w": 100, "h": 120},
      "confidence": 0.93,
      "label": "Alice",
      "score": 0.84,
      "known": true,
      "landmarks": [[...]],
      "track_id": null
    }
  ]
}
```

Multiple images return `{"results": [...], "count": N}`.

### POST /enroll?name=<name>

Enroll a person from one or more images. Multipart `image` fields or JSON
`{"name": "...", "images": ["<base64>"]}`.

Response:

```json
{"name": "Alice", "added": 2, "total": 2, "people": ["Alice"]}
```

## Client examples

Python:

```python
import base64, json, urllib.request

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

req = urllib.request.Request(
    "http://localhost:8000/detect",
    data=json.dumps({"image": b64}).encode(),
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req).read().decode())
```

curl:

```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@photo.jpg"
```

## Errors

Errors are JSON objects: `{"error": "...", "detail": "..."}` with appropriate HTTP status codes (400, 401, 404, 500, 503).
