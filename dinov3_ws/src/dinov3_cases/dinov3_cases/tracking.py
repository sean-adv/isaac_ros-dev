"""Small, class-independent ROI tracker; no model or third-party dependencies."""

from dataclasses import dataclass
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
        self.matcher = ROITracker(max_missed=0)
        self.context = None
        self.frame_key = None
        self.histories = {}
        self.boxes = []
        self.raw_boxes = None
        self.initialized = False

    def update(self, items, frame_key, context):
        if not self.initialized or context != self.context:
            self.matcher.reset()
            self.histories.clear()
            self.initialized = False
            self.context = context
        raw_boxes = tuple(tuple(item["bbox"]) for item in items)
        same_frame = self.initialized and frame_key == self.frame_key
        if not same_frame or raw_boxes != self.raw_boxes:
            # A background SAM3 handoff may change count/order on the same frame.
            # Cache by raw boxes too, so zip below can never truncate new objects.
            # Keep matching on raw coordinates; recognition/ID fields stay outside.
            matched = self.matcher.update(
                [dict(bbox=list(item["bbox"])) for item in items], frame_key=frame_key
            )
            histories, boxes = {}, []
            for item in matched:
                track_id = item["track_id"]
                history = self.histories.get(track_id, deque(maxlen=self.window))
                if same_frame and track_id in self.histories:
                    # Replace this frame's old sample instead of counting it twice.
                    history.pop()
                history.append(tuple(item["bbox"]))
                histories[track_id] = history
                mean = [sum(box[i] for box in history) / len(history) for i in range(4)]
                boxes.append([math.floor(mean[0]), math.floor(mean[1]),
                              math.ceil(mean[2]), math.ceil(mean[3])])
            self.histories, self.boxes = histories, boxes
            self.raw_boxes = raw_boxes
            self.frame_key = frame_key
            self.initialized = True
        return [dict(item, bbox=list(box), area=(box[2] - box[0]) * (box[3] - box[1]))
                for item, box in zip(items, self.boxes)]


@dataclass
class _Track:
    bbox: tuple
    missed: int = 0
    detected_frames: int = 1
    total_frames: int = 1
    last_hit_frame: int = 0


class ROITracker:
    def __init__(self, max_missed=5, iou_threshold=0.2, roi_padding=0.5,
                 bbox_min_hit_ratio=0.8, bbox_min_detections=10):
        if isinstance(max_missed, bool) or not isinstance(max_missed, int) or max_missed < 0:
            raise ValueError("tracking_max_missed must be a nonnegative integer.")
        if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
            raise ValueError("tracking_iou_threshold must be in (0, 1].")
        if not math.isfinite(roi_padding) or roi_padding < 0:
            raise ValueError("tracking_roi_padding must be finite and nonnegative.")
        if (isinstance(bbox_min_hit_ratio, bool)
                or not isinstance(bbox_min_hit_ratio, (int, float))
                or not math.isfinite(bbox_min_hit_ratio)
                or not 0 <= bbox_min_hit_ratio <= 1):
            raise ValueError("tracking_bbox_min_hit_ratio must be between 0 and 1.")
        if (isinstance(bbox_min_detections, bool)
                or not isinstance(bbox_min_detections, int) or bbox_min_detections < 0):
            raise ValueError("tracking_bbox_min_detections must be a nonnegative integer.")
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.roi_padding = roi_padding
        self.bbox_min_hit_ratio = bbox_min_hit_ratio
        self.bbox_min_detections = bbox_min_detections
        self.tracks = {}
        self.next_id = 1
        self.frame_key = None
        self.frame_number = 0

    def reset(self):
        """Forget ROIs on input/context changes, but never reuse an allocated ID."""
        self.tracks.clear()
        self.frame_key = None
        self.frame_number = 0

    def _roi(self, bbox):
        x0, y0, x1, y1 = bbox
        px = (x1 - x0) * self.roi_padding
        py = (y1 - y0) * self.roi_padding
        return (x0 - px, y0 - py, x1 + px, y1 + py)

    def update(self, items, frame_key=None):
        """Attach track_id to current detections; retain missing tracks internally.

        Reprocessing the same frame (e.g. after a gallery refresh) does not age
        tracks or inflate hit counts. Lifetime hit/total counts include misses
        and start at each track's first detection. Visibility uses strict >
        thresholds and does not remove provisional tracks or current results.
        """
        advance = frame_key is None or frame_key != self.frame_key
        if advance:
            self.frame_number += 1
            for track in self.tracks.values():
                track.total_frames += 1
        candidates = []
        rois = [self._roi(item["bbox"]) for item in items]
        for track_id, track in self.tracks.items():
            roi = self._roi(track.bbox)
            for index, item in enumerate(items):
                overlap = _iou(roi, rois[index])
                if overlap >= self.iou_threshold:
                    # Prefer the actual box overlap to the expanded search ROI.
                    candidates.append((_iou(track.bbox, item["bbox"]), overlap,
                                       -track.missed, track_id, index))
        used_tracks, used_items = set(), set()
        for _, _, _, track_id, index in sorted(candidates, reverse=True):
            if track_id in used_tracks or index in used_items:
                continue
            track = self.tracks[track_id]
            track.bbox = tuple(items[index]["bbox"])
            track.missed = 0
            if track.last_hit_frame != self.frame_number:
                track.detected_frames += 1
                track.last_hit_frame = self.frame_number
            items[index]["track_id"] = track_id
            used_tracks.add(track_id)
            used_items.add(index)
        for track_id in list(self.tracks):
            if track_id not in used_tracks and advance:
                self.tracks[track_id].missed += 1
                if self.tracks[track_id].missed > self.max_missed:
                    del self.tracks[track_id]
        for index, item in enumerate(items):
            if index not in used_items:
                item["track_id"] = self.next_id
                self.tracks[self.next_id] = _Track(
                    tuple(item["bbox"]), last_hit_frame=self.frame_number
                )
                self.next_id += 1
        for item in items:
            track = self.tracks[item["track_id"]]
            ratio = track.detected_frames / track.total_frames
            item.update(
                track_detected_frames=track.detected_frames,
                track_total_frames=track.total_frames,
                track_hit_ratio=ratio,
                bbox_visible=(ratio > self.bbox_min_hit_ratio
                              and track.detected_frames > self.bbox_min_detections),
            )
        self.frame_key = frame_key
        return items
