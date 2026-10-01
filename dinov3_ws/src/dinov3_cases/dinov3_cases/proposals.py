"""Optional candidate adapter; reuses the existing DINOv3 patch encoder and bank."""
from copy import deepcopy
import math

import cv2
import numpy as np


def crop_bounds(bbox, width, height):
    values = (bbox.center.position.x, bbox.center.position.y, bbox.size_x, bbox.size_y, bbox.center.theta)
    if not all(math.isfinite(v) for v in values) or bbox.size_x <= 0 or bbox.size_y <= 0 or abs(bbox.center.theta) > 1e-8:
        return None
    cx, cy = bbox.center.position.x, bbox.center.position.y
    x0, y0 = max(0, math.floor(cx - bbox.size_x / 2)), max(0, math.floor(cy - bbox.size_y / 2))
    x1, y1 = min(width, math.ceil(cx + bbox.size_x / 2)), min(height, math.ceil(cy + bbox.size_y / 2))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def recognize_proposals(encoder, image, bank, proposals, name, threshold):
    from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
    results = Detection2DArray()
    results.header = proposals.header
    scores = np.full((image.height, image.width), -1.0, dtype=np.float32)
    best_match = None
    for detection in proposals.detections:
        bounds = crop_bounds(detection.bbox, image.width, image.height)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        crop_scores = encoder.heatmap(image.crop(bounds), bank)
        # Mean of per-pixel best reference-patch cosine scores, not a probability.
        similarity = float(np.mean(crop_scores))
        scores[y0:y1, x0:x1] = np.maximum(scores[y0:y1, x0:x1], crop_scores)
        if similarity >= threshold and (best_match is None or similarity > best_match[0]):
            best_match = similarity, detection, bounds
    if best_match is not None:
        similarity, detection, (x0, y0, x1, y1) = best_match
        result = deepcopy(detection)
        result.header = proposals.header
        result.bbox.center.position.x = (x0 + x1) / 2.0
        result.bbox.center.position.y = (y0 + y1) / 2.0
        result.bbox.size_x, result.bbox.size_y = float(x1 - x0), float(y1 - y0)
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = name
        hypothesis.hypothesis.score = similarity
        result.results = [hypothesis]
        results.detections.append(result)
    return scores, results


def draw_proposals(image, proposals):
    for detection in proposals.detections:
        bounds = crop_bounds(detection.bbox, image.shape[1], image.shape[0])
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        color = (60, 220, 80)
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        label = f'{detection.id} ({x0},{y0})-({x1},{y1})'
        cv2.putText(image, label, (x0, max(16, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)


def draw_recognition(image, results):
    color = (0, 0, 255)
    for detection in results.detections:
        bounds = crop_bounds(detection.bbox, image.shape[1], image.shape[0])
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        label = f'{detection.id}: {detection.results[0].hypothesis.score:.2f}'
        cv2.putText(image, label, (x0, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
