"""2D rectangles and projected 3D box wireframes on a matching RGB frame."""

import cv2
import numpy as np
from .depth_bbox import bbox_vertices


def draw_bbox3d(image, data, info, color):
    points = bbox_vertices(data)
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 1e-6):
        return False
    k = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(info.d, dtype=np.float64)
    zero = np.zeros(3, dtype=np.float64)
    try:
        if info.distortion_model == "equidistant" and distortion.size == 4:
            pixels, _ = cv2.fisheye.projectPoints(points.reshape(-1, 1, 3), zero, zero, k, distortion)
        elif info.distortion_model in ("plumb_bob", "rational_polynomial", ""):
            pixels, _ = cv2.projectPoints(points, zero, zero, k,
                                          distortion if distortion.size else None)
        else:
            return False
    except cv2.error:
        return False
    pixels = pixels.reshape(-1, 2)
    if not np.isfinite(pixels).all() or np.max(np.abs(pixels)) > 1e7:
        return False
    pixels = np.rint(pixels).astype(int)
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]
    h, w = image.shape[:2]
    drawn = False
    for start, end in edges:
        visible, p0, p1 = cv2.clipLine((0, 0, w, h), tuple(pixels[start]), tuple(pixels[end]))
        if visible:
            cv2.line(image, p0, p1, color, 2, cv2.LINE_AA)
            drawn = True
    return drawn


def render_bbox_overlay(image, detections, info=None, mode="2d"):
    if mode not in ("2d", "3d"):
        raise ValueError("Invalid bbox visualization mode.")
    overlay = image.copy()
    for item in detections:
        if not item["bbox_visible"]:
            continue
        x0, y0, x1, y1 = item["bbox"]
        color = (0, 190, 255) if item["label"] == "unknown" else (60, 220, 80)
        if mode == "3d":
            if item["label"] == "unknown" or "bbox3d" not in item or info is None:
                continue
            if not draw_bbox3d(overlay, item["bbox3d"], info, color):
                continue
        else:
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        label = f"ID {item['track_id']} {item['label']} | {item['detection_label']}"
        bbox3d = item.get("bbox3d")
        if bbox3d is not None:
            px, py, pz = bbox3d["position"]
            position_label = f"position (m): X={px:.3f} Y={py:.3f} Z={pz:.3f}"
            distance_label = f"distance: {np.linalg.norm((px, py, pz)):.3f} m"
        else:
            position_label = "position: N/A"
            distance_label = "distance: N/A"
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (_, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        line_height = text_height + baseline + 8
        label_lines = (label, position_label, distance_label)
        label_y = max(
            text_height + 6,
            min(y0 - 8, overlay.shape[0] - (len(label_lines) - 1) * line_height - baseline - 6),
        )
        for line_index, text in enumerate(label_lines):
            (text_width, _), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            origin = (
                max(4, min(x0, overlay.shape[1] - text_width - 4)),
                label_y + line_index * line_height,
            )
            cv2.putText(
                overlay, text, origin, font, font_scale, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                overlay, text, origin, font, font_scale, color, thickness, cv2.LINE_AA
            )
    return overlay
