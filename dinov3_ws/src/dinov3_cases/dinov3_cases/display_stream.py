"""Choose the main page's single live image and its encoding demand.

Call while holding node.lock so the JPEG and its metadata agree.
"""
import time


OVERLAY_MAX_AGE = 3.0


def has_display_boxes(detections, mode):
    return any(item.get("bbox_visible", True)
               and (mode == "2d" or "bbox3d" in item)
               for item in detections)


def display_overlay_requested(node):
    return node.source == "upload" or has_display_boxes(
        node.detections, node.bbox_visualization_mode)


def display_uses_overlay(node):
    if node.jpeg is None or not display_overlay_requested(node):
        return False
    if node.source == "upload":
        return True  # A static uploaded result does not expire.
    return (node.overlay_has_boxes and node.overlay_at is not None
            and time.monotonic() - node.overlay_at < OVERLAY_MAX_AGE)
