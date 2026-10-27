"""2D rectangles and projected 3D box wireframes on a matching RGB frame."""

from types import SimpleNamespace

import cv2
import numpy as np
from .depth_bbox import bbox_vertices, project_box_points


def draw_bbox3d(image, data, info, color):
    points = bbox_vertices(data)
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 1e-6):
        return False
    try:
        pixels = project_box_points(points, info)
    except (cv2.error, ValueError):
        return False
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
            if "bbox3d" not in item or info is None:
                continue
            if not draw_bbox3d(overlay, item["bbox3d"], info, color):
                continue
        else:
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        label = f"{item['label']} | {item['detection_label']}"
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        origin = (
            max(4, min(x0, overlay.shape[1] - text_width - 4)),
            max(text_height + 6, min(y0 - 8, overlay.shape[0] - baseline - 6)),
        )
        cv2.putText(
            overlay, label, origin, font, font_scale, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            overlay, label, origin, font, font_scale, color, thickness, cv2.LINE_AA
        )
    return overlay


def render_display_overlay(rgb, detections, info=None, mode="3d"):
    """Render browser pixels without changing source boxes or metric 3D data."""
    height, width = rgb.shape[:2]
    sx, sy = 960 / width, 540 / height
    pixels = (cv2.resize(rgb, (960, 540), interpolation=cv2.INTER_AREA)
              if (width, height) != (960, 540) else rgb)
    bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    # Scale the inclusive drawn endpoints, then restore exclusive bbox maxima.
    scaled_detections = [
        dict(item, bbox=[round((value - (1 if i >= 2 else 0)) *
                              (sx if i % 2 == 0 else sy)) + (1 if i >= 2 else 0)
                         for i, value in enumerate(item["bbox"])])
        for item in detections
    ]
    scaled_info = None
    if info is not None:
        k = np.asarray(info.k, dtype=np.float64).reshape(3, 3).copy()
        k[0, :] *= sx
        k[1, :] *= sy
        # Distortion and metric geometry do not change when resizing the image.
        scaled_info = SimpleNamespace(k=k.reshape(-1).tolist(), d=info.d,
                                      distortion_model=info.distortion_model)
    return render_bbox_overlay(bgr, scaled_detections, scaled_info, mode)
