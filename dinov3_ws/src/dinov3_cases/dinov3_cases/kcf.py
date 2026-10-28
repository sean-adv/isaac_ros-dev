"""Periodic SAM3 proposals with OpenCV KCF between detection frames."""

import math

import cv2

from .kcf_lab import LabCheckedTracker, validate_lab_settings


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


class _ResizedTracker:
    """Bound KCF working pixels while returning coordinates in the original ROI."""

    def __init__(self, tracker, max_side):
        self.tracker, self.max_side = tracker, max_side
        self.size = None
        self.scale = (1.0, 1.0)

    def _image(self, image):
        if self.size == (image.shape[1], image.shape[0]):
            return image
        return cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)

    def init(self, image, box):
        height, width = image.shape[:2]
        # Keep at least 16 pixels on the target's smaller side, where possible.
        ratio = min(1.0, max(self.max_side / max(width, height),
                             16 / min(box[2], box[3])))
        self.size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        self.scale = (self.size[0] / width, self.size[1] / height)
        sx, sy = self.scale
        x, y, w, h = box
        scaled_box = (round(x * sx), round(y * sy),
                      max(1, round(w * sx)), max(1, round(h * sy)))
        return self.tracker.init(self._image(image), scaled_box)

    def update(self, image):
        ok, box = self.tracker.update(self._image(image))
        if not ok:
            return ok, box
        sx, sy = self.scale
        x, y, w, h = box
        return ok, (x / sx, y / sy, w / sx, h / sy)


class KCFProposals:
    """Count unique processed frames, never camera callbacks or gallery refreshes.

    Detection replaces all trackers, including when SAM3 returns no objects.
    Lost/invalid tracks trigger synchronous detection by default. Async callers
    can retain valid survivors and request a background refresh instead. Detection
    scores on tracked proposals remain the last SAM3 score, not KCF confidence.
    Trackers use fixed crops padded per side until the next detection.
    max_roi_side bounds working pixels only; boxes stay in input-image coordinates.
    Small targets limit downsampling to retain at least 16 pixels where possible.
    Optional Lab checks reject consecutive appearance changes against the SAM3 seed.
    refresh_requested signals a crop boundary or lost/changed track to async callers.
    """

    def __init__(self, factory=None, roi_padding=0.5, max_roi_side=0,
                 lab_similarity_threshold=0.0, lab_change_confirmations=3):
        validate_lab_settings(lab_similarity_threshold, lab_change_confirmations)
        self.lab_similarity_threshold = lab_similarity_threshold
        self.lab_change_confirmations = lab_change_confirmations
        if roi_padding is not None and (
            isinstance(roi_padding, bool) or not isinstance(roi_padding, (int, float))
            or not math.isfinite(roi_padding) or roi_padding < 0
        ):
            raise ValueError("roi_padding must be finite and >= 0, or None for full-frame.")
        if type(max_roi_side) is not int or max_roi_side < 0:
            raise ValueError("max_roi_side must be an integer >= 0 (0 disables resizing).")
        self.max_roi_side = max_roi_side
        self.roi_padding = roi_padding
        self.factory = factory
        self.reset()

    def reset(self):
        self.context = None
        self.frame_key = None
        self.interval = None
        self.since_detection = 0
        self.proposals = []
        self.trackers = []
        self.rois = []
        self.refresh_requested = False
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

    def _roi(self, bbox, pixels):
        if self.roi_padding is None:
            return (0, 0, pixels.shape[1], pixels.shape[0])
        x0, y0, x1, y1 = bbox
        px, py = (x1-x0)*self.roi_padding, (y1-y0)*self.roi_padding
        return (max(0, math.floor(x0-px)), max(0, math.floor(y0-py)),
                min(pixels.shape[1], math.ceil(x1+px)),
                min(pixels.shape[0], math.ceil(y1+py)))

    @staticmethod
    def _crop(pixels, roi):
        x0, y0, x1, y1 = roi
        return cv2.cvtColor(pixels[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)

    def _near_edge(self, bbox, roi, pixels):
        if self.roi_padding is None:
            return False
        x0, y0, x1, y1 = bbox
        rx0, ry0, rx1, ry1 = roi
        mx, my = max(1, (x1-x0)*0.1), max(1, (y1-y0)*0.1)
        # Ignore physical image boundaries, which cannot be expanded further.
        return ((rx0 > 0 and x0-rx0 <= mx) or (ry0 > 0 and y0-ry0 <= my)
                or (rx1 < pixels.shape[1] and rx1-x1 <= mx)
                or (ry1 < pixels.shape[0] and ry1-y1 <= my))

    def update(self, pixels, frame_key, context, interval, detect, *, retain_valid_tracks=False):
        validate_detection_interval(interval)
        if context != self.context or interval != self.interval:
            self.reset()
            self.context, self.interval = context, interval
        if self.initialized and frame_key == self.frame_key:
            return [dict(p, bbox=list(p["bbox"])) for p in self.proposals], self.source

        due = (not self.initialized or interval == 1 or self.retry_detection
               or self.since_detection + 1 >= interval)
        self.refresh_requested = False
        full_bgr = (cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                    if interval > 1 and self.roi_padding is None else None)
        proposals = []
        if not due:
            surviving_trackers, surviving_rois = [], []
            lost = False
            for tracker, previous, roi in zip(self.trackers, self.proposals, self.rois):
                try:
                    bgr = full_bgr if full_bgr is not None else self._crop(pixels, roi)
                    ok, box = tracker.update(bgr)
                    local = self._clip(box, bgr) if ok else None
                    # Exiting an artificial crop edge invalidates the box;
                    # physical image edges retain the normal clipping behavior.
                    if local is not None and (
                            (box[0] < 0 and roi[0] > 0)
                            or (box[1] < 0 and roi[1] > 0)
                            or (box[0]+box[2] > bgr.shape[1] and roi[2] < pixels.shape[1])
                            or (box[1]+box[3] > bgr.shape[0] and roi[3] < pixels.shape[0])):
                        local = None
                    bbox = ([local[0]+roi[0], local[1]+roi[1],
                             local[2]+roi[0], local[3]+roi[1]] if local else None)
                except cv2.error:
                    bbox = None
                if bbox is None:
                    if retain_valid_tracks:
                        lost = True
                        continue
                    due = True
                    break
                surviving_trackers.append(tracker)
                surviving_rois.append(roi)
                self.refresh_requested |= self._near_edge(bbox, roi, pixels)
                lab_metrics = (dict(kcf_lab_similarity=tracker.lab_similarity,
                                    kcf_lab_changed_frames=tracker.lab_changed_frames)
                               if isinstance(tracker, LabCheckedTracker) else {})
                proposals.append(dict(previous, **lab_metrics, bbox=bbox,
                                      area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                                      bbox_source="kcf"))
            if retain_valid_tracks and lost:
                self.trackers, self.rois = surviving_trackers, surviving_rois
                self.refresh_requested = True
                # Keep valid current-frame boxes while async SAM3 refreshes.
                # Total loss clears the stale set through detect() below.
                due = not proposals
        if due:
            # Retry detection if the detector raises before this frame commits.
            self.retry_detection = True
            proposals = [dict(p, bbox=list(p["bbox"]), bbox_source="sam3") for p in detect()]
            trackers, rois, initialized_proposals = [], [], []
            failed_init = False
            if interval > 1:
                factory = self.factory or kcf_factory()
                for proposal in proposals:
                    tracker = factory()
                    if self.max_roi_side:
                        tracker = _ResizedTracker(tracker, self.max_roi_side)
                    if self.lab_similarity_threshold > 0:
                        tracker = LabCheckedTracker(tracker, self.lab_similarity_threshold,
                                                    self.lab_change_confirmations)
                    x0, y0, x1, y1 = proposal["bbox"]
                    roi = self._roi(proposal["bbox"], pixels)
                    try:
                        bgr = full_bgr if full_bgr is not None else self._crop(pixels, roi)
                        ok = tracker.init(bgr, (int(x0-roi[0]), int(y0-roi[1]),
                                               int(x1-x0), int(y1-y0)))
                    except cv2.error:
                        ok = False
                    # Modern OpenCV returns None; the legacy API returns bool.
                    if ok is not None and not ok:
                        failed_init = True
                        if retain_valid_tracks:
                            continue
                    trackers.append(tracker)
                    rois.append(roi)
                    initialized_proposals.append(proposal)
                if retain_valid_tracks:
                    proposals = initialized_proposals
            self.trackers, self.rois = trackers, rois
            self.retry_detection = failed_init and (not retain_valid_tracks or not proposals)
            self.since_detection = 0
            self.source = "sam3"
        else:
            self.since_detection += 1
            self.source = "kcf"
        self.proposals = proposals
        self.frame_key = frame_key
        self.initialized = True
        return [dict(p, bbox=list(p["bbox"])) for p in proposals], self.source
