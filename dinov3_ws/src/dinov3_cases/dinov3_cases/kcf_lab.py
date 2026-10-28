"""Bounded CPU-only Lab appearance checks against a fixed detection-frame crop."""

import math

import cv2
import numpy as np


def validate_lab_settings(threshold, confirmations):
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("kcf_lab_similarity_threshold must be finite and in [0, 1]")
    if type(confirmations) is not int or confirmations < 1:
        raise ValueError("kcf_lab_change_confirmations must be an integer >= 1")


def _histogram(image, box):
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    height, width = image.shape[:2]
    x0, y0 = max(0, math.floor(x)), max(0, math.floor(y))
    x1, y1 = min(width, math.ceil(x+w)), min(height, math.ceil(y+h))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image[y0:y1, x0:x1]
    # Bound conversion/histogram cost independently of the input resolution.
    crop = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [4, 8, 8], [0, 256]*3).ravel()
    return np.sqrt(hist / hist.sum())


class LabCheckedTracker:
    """Reject consecutive appearance changes even if KCF still reports success.

    The reference stays fixed until the next SAM3 seed. This check measures color
    distribution, not object identity. Its state travels with the tracker through
    async handoff and survivor retention. Input is the original-resolution BGR ROI.
    """

    def __init__(self, tracker, threshold, confirmations):
        self.tracker = tracker
        self.threshold = threshold
        self.confirmations = confirmations
        self.lab_similarity = 1.0
        self.lab_changed_frames = 0

    def init(self, image, box):
        self.reference = _histogram(image, box)
        self.lab_similarity = 1.0
        self.lab_changed_frames = 0
        if self.reference is None:
            return False
        return self.tracker.init(image, box)

    def update(self, image):
        ok, box = self.tracker.update(image)
        if not ok:
            return ok, box
        current = _histogram(image, box)
        if current is None:
            return False, box
        self.lab_similarity = float(np.clip(self.reference @ current, 0, 1))
        self.lab_changed_frames = (self.lab_changed_frames + 1
                                   if self.lab_similarity < self.threshold else 0)
        return self.lab_changed_frames < self.confirmations, box
