"""HTTP API exposing the face detection and recognition features.

Run with:
    python src/api_server.py            # listens on 0.0.0.0:8000
    python src/api_server.py --port 9000

Other apps can call the REST endpoints documented in docs/API.md and
/openapi.json. Set FACE_API_KEY to require an X-API-Key header.
"""

import argparse
import base64
import email.parser
import email.policy
import hmac
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from face_detector import FaceDetector  # noqa: E402
from face_recognizer import (  # noqa: E402
    DB_PATH,
    UNKNOWN_LABEL,
    FaceRecognizer,
    draw_label,
)
from enroll import largest_face, load_db, save_db  # noqa: E402

API_VERSION = "1.0.0"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

_lock = threading.Lock()
_detector = None
_recognizer = None
_db_lock = threading.Lock()
_process_lock = threading.Lock()


def get_detector():
    global _detector
    with _lock:
        if _detector is None:
            _detector = FaceDetector()
        return _detector


def get_recognizer():
    global _recognizer
    with _lock:
        if _recognizer is None:
            _recognizer = FaceRecognizer()
        return _recognizer


API_KEY = os.environ.get("FACE_API_KEY", "").strip()


def check_auth(headers):
    if not API_KEY:
        return True
    req_key = headers.get("X-API-Key", "")
    return hmac.compare_digest(req_key, API_KEY)


def decode_image_bytes(data):
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


def b64_encode_image(image, quality=85):
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def detection_to_dict(det, image, recognizer):
    label, score = recognizer.identify(image, det)
    box = det.box
    landmarks = None
    if det.landmarks:
        landmarks = [[float(x), float(y)] for (x, y) in det.landmarks]
    return {
        "box": {"x": int(box[0]), "y": int(box[1]), "w": int(box[2]), "h": int(box[3])},
        "confidence": float(det.confidence),
        "label": label,
        "score": float(score),
        "known": label != UNKNOWN_LABEL,
        "landmarks": landmarks,
        "track_id": det.track_id,
    }


def process_image(image, filename, annotate=False):
    detector = get_detector()
    recognizer = get_recognizer()
    faces = detector.detect(image)
    face_dicts = [detection_to_dict(d, image, recognizer) for d in faces]
    result = {
        "filename": filename,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "count": len(faces),
        "known_count": sum(1 for f in face_dicts if f["known"]),
        "detector": detector.name,
        "faces": face_dicts,
    }
    if annotate:
        annotated = image.copy()
        for det, fd in zip(faces, face_dicts):
            draw_label(annotated, det.box, fd["label"])
        result["annotated_image"] = b64_encode_image(annotated)
    return result


def read_multipart(handler):
    """Return (files: list[(name, bytes, filename)], fields: dict)."""
    files = []
    fields = {}
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        return files, fields

    try:
        length = int(handler.headers.get("Content-Length", "0") or 0)
    except (ValueError, TypeError):
        length = 0

    if length <= 0:
        return files, fields

    raw_body = handler.rfile.read(length)
    header_bytes = f"Content-Type: {content_type}\r\n\r\n".encode("latin1", errors="replace")
    msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(header_bytes + raw_body)

    if msg.is_multipart():
        for part in msg.iter_parts():
            filename = part.get_filename()
            name = part.get_param("name", header="content-disposition") or "file"
            payload = part.get_payload(decode=True)
            if filename:
                files.append((name, payload or b"", filename))
            else:
                if payload is not None:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        fields[name] = payload.decode(charset, errors="replace")
                    except Exception:
                        fields[name] = str(payload)
    return files, fields


def read_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def images_from_request(handler):
    """Collect (filename, image) pairs from multipart or JSON body."""
    collected = []
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" in content_type:
        files, fields = read_multipart(handler)
        for _name, data, filename in files:
            image = decode_image_bytes(data)
            if image is not None:
                collected.append((filename or "image", image))
        return collected, fields
    if "application/json" in content_type:
        body = read_json_body(handler)
        if body is None:
            return None, {"_error": "invalid json body"}
        fields = {k: v for k, v in body.items() if not isinstance(v, (list, dict)) or k not in ("image", "images")}
        images_payload = []
        if "image" in body:
            images_payload.append(("image", body["image"]))
        if "images" in body and isinstance(body["images"], list):
            for idx, value in enumerate(body["images"]):
                images_payload.append((f"image_{idx}", value))
        for name, value in images_payload:
            if isinstance(value, str):
                try:
                    data = base64.b64decode(value, validate=True)
                except Exception:
                    continue
                image = decode_image_bytes(data)
                if image is not None:
                    collected.append((name, image))
        return collected, fields
    # raw binary
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length > 0:
        data = handler.rfile.read(length)
        image = decode_image_bytes(data)
        if image is not None:
            collected.append(("image", image))
    return collected, {}


OPENAPI = {
    "openapi": "3.0.3",
    "info": {
        "title": "Face Detection API",
        "version": API_VERSION,
        "description": "REST API for face detection and recognition.",
    },
    "servers": [{"url": "/"}],
    "paths": {
        "/health": {"get": {"summary": "Health check", "responses": {"200": {"description": "OK"}}}},
        "/info": {"get": {"summary": "Service information"}},
        "/people": {"get": {"summary": "List enrolled people"}},
        "/detect": {
            "post": {
                "summary": "Detect faces in an image",
                "parameters": [
                    {"name": "annotate", "in": "query", "schema": {"type": "boolean"}},
                ],
                "requestBody": {
                    "content": {
                        "multipart/form-data": {
                            "schema": {"type": "object", "properties": {"image": {"type": "string", "format": "binary"}}}
                        },
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"image": {"type": "string", "description": "base64-encoded image"}},
                            }
                        },
                    }
                },
                "responses": {"200": {"description": "Detection results"}},
            }
        },
        "/enroll": {
            "post": {
                "summary": "Enroll a person from image(s)",
                "parameters": [{"name": "name", "in": "query", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "Enrollment result"}},
            }
        },
    },
}


DOCS_HTML = """<!doctype html>
<html><head><title>Face Detection API</title></head>
<body>
<h1>Face Detection API</h1>
<p>See <a href="/openapi.json">OpenAPI spec</a> and docs/API.md.</p>
<ul>
<li>GET /health</li>
<li>GET /info</li>
<li>GET /people</li>
<li>POST /detect (multipart image or JSON base64; ?annotate=1)</li>
<li>POST /enroll?name=... (multipart image(s) or JSON)</li>
</ul>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # quiet by default
        pass

    def _send_json(self, status, payload):
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, status, text, content_type="text/plain"):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _auth_ok(self):
        if check_auth(self.headers):
            return True
        self._send_json(401, {"error": "unauthorized", "detail": "invalid or missing X-API-Key"})
        return False

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/health":
            self._send_json(200, {"status": "ok", "service": "face-detection-api", "version": API_VERSION})
            return
        if path == "/info":
            if not self._auth_ok():
                return
            recognizer = get_recognizer()
            detector = get_detector()
            self._send_json(200, {
                "detector": detector.name,
                "recognition_enabled": recognizer.enabled,
                "known_people": recognizer.people,
                "api_version": API_VERSION,
            })
            return
        if path == "/people":
            if not self._auth_ok():
                return
            recognizer = get_recognizer()
            self._send_json(200, {"people": recognizer.people})
            return
        if path == "/openapi.json":
            self._send_json(200, OPENAPI)
            return
        if path == "/docs":
            self._send_text(200, DOCS_HTML, "text/html")
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if not self._auth_ok():
            return
        if path == "/detect":
            self.handle_detect(query)
            return
        if path == "/enroll":
            self.handle_enroll(query)
            return
        self._send_json(404, {"error": "not found"})

    def handle_detect(self, query):
        annotate = str(query.get("annotate", [""])[0]).lower() in ("1", "true", "yes")
        try:
            collected, fields = images_from_request(self)
        except Exception as error:
            self._send_json(400, {"error": "bad request", "detail": str(error)})
            return
        if collected is None:
            self._send_json(400, {"error": "bad request", "detail": fields.get("_error", "invalid request")})
            return
        if not collected:
            self._send_json(400, {"error": "no image provided", "detail": "send multipart 'image' or JSON base64 'image'"})
            return
        try:
            with _process_lock:
                results = [process_image(img, name, annotate=annotate) for name, img in collected]
        except Exception as error:
            self._send_json(500, {"error": "detection failed", "detail": str(error)})
            return
        if len(results) == 1:
            self._send_json(200, results[0])
        else:
            self._send_json(200, {"results": results, "count": len(results)})

    def handle_enroll(self, query):
        name = query.get("name", [None])[0]
        try:
            collected, fields = images_from_request(self)
        except Exception as error:
            self._send_json(400, {"error": "bad request", "detail": str(error)})
            return
        if name is None:
            name = fields.get("name")
        if not name:
            self._send_json(400, {"error": "name required", "detail": "pass ?name=... or form field name"})
            return
        if not collected:
            self._send_json(400, {"error": "no image provided"})
            return
        recognizer = get_recognizer()
        detector = get_detector()
        if not recognizer.enabled:
            self._send_json(503, {"error": "recognition disabled", "detail": "SFace model missing"})
            return
        new_embeddings = []
        with _process_lock:
            for _fname, image in collected:
                detection = largest_face(detector, image)
                if detection is None:
                    continue
                embedding = recognizer.feature(image, detection)
                if embedding is None:
                    continue
                new_embeddings.append(embedding.flatten().tolist())

        with _db_lock:
            db = load_db()
            existing = list(db.get(name, []))
            existing.extend(new_embeddings)
            added = len(new_embeddings)
            existing = existing[-100:]
            db[name] = existing
            save_db(db)
            recognizer.reload()
        self._send_json(200, {
            "name": name,
            "added": added,
            "total": len(db[name]),
            "people": sorted(db.keys()),
        })


def main():
    parser = argparse.ArgumentParser(description="Face Detection HTTP API server")
    parser.add_argument("--host", default=os.environ.get("FACE_API_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FACE_API_PORT", DEFAULT_PORT)))
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Face Detection API listening on http://{args.host}:{args.port}")
    print("Docs: /docs  OpenAPI: /openapi.json")
    if API_KEY:
        print("API key authentication enabled (X-API-Key).")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
