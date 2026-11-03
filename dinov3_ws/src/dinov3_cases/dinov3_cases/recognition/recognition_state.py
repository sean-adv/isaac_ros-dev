"""Recognition state transitions shared by HTTP controls.

Callers must hold node.lock. Keep each setting/source update and invalidation in
one critical section so workers cannot publish results from the old generation.
Validation and image decoding must complete before calling these functions.
"""


def invalidate_results(node):
    """Discard output from the previous input revision, including MJPEG/debug data."""
    node.generation += 1
    node.video_at = None
    node.image_stream.reset()
    node.clear_object_debug()
    node.detections = []


def set_grounding_settings(node, settings):
    node.grounding_settings = settings
    invalidate_results(node)
    node.status = "Settings applied: Preparing first detection and recognition…"


def set_region(node, region, previous):
    """Reapplying the same region preserves valid results."""
    if region != previous:
        node.sam3_roi = region
        invalidate_results(node)
        node.status = "SAM3 region applied: Preparing first detection and recognition…"


def gallery_changed(node):
    invalidate_results(node)
    node.status = "Building reference features…"


def select_case(node, name, *, rebuild=False):
    node.selected = name
    if rebuild:
        gallery_changed(node)


def set_source(node, mode, image):
    node.source = mode
    node.frame = image
    node.frame_seq += 1
    invalidate_results(node)
