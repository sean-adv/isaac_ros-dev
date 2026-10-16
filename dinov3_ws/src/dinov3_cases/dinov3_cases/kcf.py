"""Periodic SAM3 proposals with OpenCV KCF between detection frames."""

import math

import cv2


def validate_detection_interval(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("detection_interval must be an integer >= 1.")


def kcf_factory():
    for namespace in (cv2, getattr(cv2, "legacy", None)):
        factory = getattr(namespace, "TrackerKCF_create", None)
        if factory is not None:
            return factory
    raise ValueError(
        "KCF requires opencv-contrib-python. Remove opencv-python / "
        "opencv-python-headless and install opencv-contrib-python>=4.8,<4.12 "
        "in the DINOv3 main-process environment."
    )


class KCFProposals:
    """Count unique processed frames, never camera callbacks or gallery refreshes.

    Detection replaces all trackers, including when SAM3 returns no objects.
    Any lost/invalid track triggers detection on that same frame. Detection
    scores on tracked proposals remain the last SAM3 score, not KCF confidence.
    """

    def __init__(self, factory=None):
        self.factory = factory
        self.reset()

    def reset(self):
        self.context = None
        self.frame_key = None
        self.interval = None
        self.since_detection = 0
        self.proposals = []
        self.trackers = []
        self.source = "sam3"
        self.initialized = False
        self.retry_detection = False

    @staticmethod
    def _clip(box, pixels):
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            return None
        height, width = pixels.shape[:2]
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return None
        x0, y0 = max(0, math.floor(x)), max(0, math.floor(y))
        x1, y1 = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
        return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None

    def update(self, pixels, frame_key, context, interval, detect):
        validate_detection_interval(interval)
        if context != self.context or interval != self.interval:
            self.reset()
            self.context, self.interval = context, interval
        if self.initialized and frame_key == self.frame_key:
            return [dict(p, bbox=list(p["bbox"])) for p in self.proposals], self.source

        due = (not self.initialized or interval == 1 or self.retry_detection
               or self.since_detection + 1 >= interval)
        # SAM3/shared memory uses RGB; OpenCV KCF's color features expect BGR.
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR) if interval > 1 else None
        proposals = []
        if not due:
            for tracker, previous in zip(self.trackers, self.proposals):
                try:
                    ok, box = tracker.update(bgr)
                    bbox = self._clip(box, pixels) if ok else None
                except cv2.error:
                    bbox = None
                if bbox is None:
                    due = True
                    break
                proposals.append(dict(previous, bbox=bbox,
                                      area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                                      bbox_source="kcf"))
        if due:
            # Retry detection if the detector raises before this frame commits.
            self.retry_detection = True
            proposals = [dict(p, bbox=list(p["bbox"]), bbox_source="sam3") for p in detect()]
            trackers = []
            failed_init = False
            if interval > 1:
                factory = self.factory or kcf_factory()
                for proposal in proposals:
                    tracker = factory()
                    x0, y0, x1, y1 = proposal["bbox"]
                    try:
                        ok = tracker.init(bgr, (int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
                    except cv2.error:
                        ok = False
                    # Modern OpenCV returns None; the legacy API returns bool.
                    if ok is not None and not ok:
                        failed_init = True
                    trackers.append(tracker)
            self.trackers = trackers
            self.retry_detection = failed_init
            self.since_detection = 0
            self.source = "sam3"
        else:
            self.since_detection += 1
            self.source = "kcf"
        self.proposals = proposals
        self.frame_key = frame_key
        self.initialized = True
        return [dict(p, bbox=list(p["bbox"])) for p in proposals], self.source
