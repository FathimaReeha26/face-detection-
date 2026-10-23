"""Post-processing shared by every detection backend.

The confidence threshold, the non-maximum suppression, the clipping to the
image and the conversion into the landmark row used by the SFace recognizer
live here, so the GPU and the CPU backend produce the same kind of result
instead of each rolling its own version.
"""

import cv2
import numpy as np

from .types import Box, Detection, Landmark

# The order OpenCV's SFace alignment expects and OpenCV's YuNet produces:
# right eye, left eye, nose, right mouth corner, left mouth corner.
LANDMARK_ORDER = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")


def clip_box(box: Box, width: int, height: int) -> Box | None:
    """Clamp a box to the image and drop it when it becomes empty."""
    x, y, w, h = box
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + int(w), y1 + int(h)

    cx1 = max(0, min(x1, width))
    cy1 = max(0, min(y1, height))
    cx2 = max(0, min(x2, width))
    cy2 = max(0, min(y2, height))

    nw = cx2 - cx1
    nh = cy2 - cy1
    if nw <= 0 or nh <= 0:
        return None
    return (cx1, cy1, nw, nh)


def corners_to_box(x1: float, y1: float, x2: float, y2: float) -> Box:
    """Convert corner coordinates into the (x, y, w, h) box the app uses."""
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def letterbox(image: "cv2.Mat", input_size: tuple[int, int]) -> tuple["cv2.Mat", float]:
    """Scale an image into the model input and pad the rest.

    The aspect ratio is kept and the empty space is added on the right and
    at the bottom, so a coordinate found in the model input maps back to the
    original image by dividing it with the returned scale.
    """
    input_width, input_height = input_size
    ratio = float(image.shape[0]) / image.shape[1]
    model_ratio = float(input_height) / input_width
    if ratio > model_ratio:
        new_height = input_height
        new_width = int(new_height / ratio)
    else:
        new_width = input_width
        new_height = int(new_width * ratio)

    scale = float(new_height) / image.shape[0]
    resized = cv2.resize(image, (new_width, new_height))
    canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    canvas[:new_height, :new_width, :] = resized
    return canvas, scale


def suppress(
    detections: list[Detection], score_threshold: float, nms_threshold: float
) -> list[Detection]:
    """Drop low scoring boxes and keep one box per overlapping group."""
    if not detections:
        return []
    boxes = [detection.box for detection in detections]
    scores = [float(detection.confidence) for detection in detections]
    keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
    indices = np.asarray(keep).reshape(-1).tolist()
    return [detections[int(index)] for index in indices]


def landmark_row(box: Box, landmarks: tuple[Landmark, ...], confidence: float):
    """Build the 15 value row cv2.FaceRecognizerSF.alignCrop expects.

    The row keeps the YuNet layout, so a face found by the GPU backend can be
    aligned exactly like a face found by the CPU backend.
    """
    values = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    for landmark in landmarks:
        values.append(float(landmark[0]))
        values.append(float(landmark[1]))
    values.append(float(confidence))
    return np.array(values, dtype=np.float32)


def intersection_over_union(first: Box, second: Box) -> float:
    """Return the overlap of two boxes as a number between 0 and 1."""
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    overlap_width = min(ax2, bx2) - max(ax1, bx1)
    overlap_height = min(ay2, by2) - max(ay1, by1)
    if overlap_width <= 0 or overlap_height <= 0:
        return 0.0
    intersection = float(overlap_width * overlap_height)
    union = float(aw * ah + bw * bh) - intersection
    return (intersection / union) if union > 0 else 0.0
