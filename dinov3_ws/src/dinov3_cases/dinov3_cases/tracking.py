"""Geometric history matching and moving averages for SAM3/KCF boxes."""

from collections import deque
import math


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


class BBoxSmoother:
    """Average raw boxes per matched object without feeding smoothing into KCF.

    Match geometrically across SAM3 refreshes, including detection reordering.
    Missing objects lose their history immediately. A repeated frame with the
    same raw boxes reuses its result; a same-frame correction replaces samples
    so gallery refreshes and detector handoffs cannot bias the average.
    """

    def __init__(self, window=3):
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("bbox_smoothing_frames must be an integer >= 1.")
        self.window = window
        self.context = None
        self.frame_key = None
        self.histories = {}
        self.boxes = []
        self.raw_boxes = None
        self.initialized = False

    def update(self, items, frame_key, context):
        if not self.initialized or context != self.context:
            self.histories.clear()
            self.initialized = False
            self.context = context
        raw_boxes = tuple(tuple(item["bbox"]) for item in items)
        same_frame = self.initialized and frame_key == self.frame_key
        if not same_frame or raw_boxes != self.raw_boxes:
            # A background SAM3 handoff may change count/order on the same frame.
            # Cache by raw boxes too, so zip below can never truncate new objects.
            # Keep matching on raw coordinates; recognition/ID fields stay outside.
            matches = _match_boxes(self.raw_boxes or (), raw_boxes) if self.initialized else {}
            histories, boxes = {}, []
            for index, bbox in enumerate(raw_boxes):
                previous = matches.get(index)
                history = self.histories.get(previous, deque(maxlen=self.window))
                if same_frame and previous in self.histories:
                    # Replace this frame's old sample instead of counting it twice.
                    history.pop()
                history.append(bbox)
                histories[index] = history
                mean = [sum(box[i] for box in history) / len(history) for i in range(4)]
                boxes.append([math.floor(mean[0]), math.floor(mean[1]),
                              math.ceil(mean[2]), math.ceil(mean[3])])
            self.histories, self.boxes = histories, boxes
            self.raw_boxes = raw_boxes
            self.frame_key = frame_key
            self.initialized = True
        return [dict(item, bbox=list(box), area=(box[2] - box[0]) * (box[3] - box[1]))
                for item, box in zip(items, self.boxes)]


def _match_boxes(previous, current):
    """Match only adjacent raw box sets for smoothing, without persistent IDs."""
    def expanded(bbox):
        x0, y0, x1, y1 = bbox
        px, py = (x1 - x0) * 0.5, (y1 - y0) * 0.5
        return (x0 - px, y0 - py, x1 + px, y1 + py)

    candidates = []
    previous_rois = [expanded(bbox) for bbox in previous]
    current_rois = [expanded(bbox) for bbox in current]
    for old_index, old_box in enumerate(previous):
        for index, bbox in enumerate(current):
            overlap = _iou(previous_rois[old_index], current_rois[index])
            if overlap >= 0.2:
                candidates.append((_iou(old_box, bbox), overlap, old_index, index))
    matches, used_previous = {}, set()
    for _, _, old_index, index in sorted(candidates, reverse=True):
        if old_index not in used_previous and index not in matches:
            matches[index] = old_index
            used_previous.add(old_index)
    return matches
