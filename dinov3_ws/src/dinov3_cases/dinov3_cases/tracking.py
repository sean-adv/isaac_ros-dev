"""Geometric history matching and moving averages for SAM3/KCF boxes."""

from collections import deque
import math
import time


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def match_objects(previous, current):
    """Conservative one-to-one association in a common frame's coordinates.

    Nearby centers allow modest detector corrections, but size and distance
    gates prevent a neighboring object from inheriting an unrelated tracker.
    Recognition labels are deliberately not identities (two items can match).
    """
    candidates = []
    for old_index, old in enumerate(previous):
        a = old["bbox"]
        aw, ah = a[2] - a[0], a[3] - a[1]
        if min(aw, ah) <= 0:
            continue
        for index, item in enumerate(current):
            b = item["bbox"]
            bw, bh = b[2] - b[0], b[3] - b[1]
            if min(bw, bh) <= 0 or not .25 <= bw * bh / (aw * ah) <= 4:
                continue
            distance = math.hypot((a[0]+a[2]-b[0]-b[2]) / (aw+bw),
                                  (a[1]+a[3]-b[1]-b[3]) / (ah+bh))
            overlap = _iou(a, b)
            if overlap >= .2 or distance <= .5:
                candidates.append((overlap, -distance, old_index, index))
    matches, used = {}, set()
    for _, _, old_index, index in sorted(candidates, reverse=True):
        if old_index not in used and index not in matches:
            matches[index] = old_index
            used.add(old_index)
    return matches


class TrackIdentities:
    """Keep IDs across refreshes and briefly remember lost, undisplayed items."""

    def __init__(self, recovery_seconds=1.0):
        self.recovery_seconds = recovery_seconds
        self.next_id = 0
        self.clear()

    def clear(self):
        # IDs must not be recycled when camera/settings context changes.
        self.recent = {}

    def forget(self, identities):
        for identity in identities:
            self.recent.pop(identity, None)

    def observe(self, items):
        now = time.monotonic()
        self.recent = {key: value for key, value in self.recent.items()
                       if now - value[1] <= self.recovery_seconds}
        for item in items:
            self.recent[item["track_id"]] = (dict(item, bbox=list(item["bbox"])), now)

    def assign(self, items, live):
        self.observe(live)
        matches = match_objects(live, items)
        live_ids = {item["track_id"] for item in live}
        lost = [item for key, (item, _) in self.recent.items() if key not in live_ids]
        unmatched = [index for index in range(len(items)) if index not in matches]
        recovered = match_objects(lost, [items[index] for index in unmatched])
        identities = {index: live[old]["track_id"] for index, old in matches.items()}
        identities.update({unmatched[index]: lost[old]["track_id"]
                           for index, old in recovered.items()})
        output = []
        for index, item in enumerate(items):
            identity = identities.get(index)
            if identity is None:
                self.next_id += 1
                identity = self.next_id
            output.append(dict(item, track_id=identity))
        self.observe(output)
        return output, matches


class TrackLifecycle:
    """Require detector support; KCF success alone never renews a track's life."""

    def __init__(self, max_missed_detections=3,
                 max_unconfirmed_seconds=5.0):
        if type(max_missed_detections) is not int or max_missed_detections < 1:
            raise ValueError("kcf_max_missed_detections must be an integer >= 1")
        if (isinstance(max_unconfirmed_seconds, bool)
                or not isinstance(max_unconfirmed_seconds, (int, float))
                or not math.isfinite(max_unconfirmed_seconds) or max_unconfirmed_seconds <= 0):
            raise ValueError("kcf_max_unconfirmed_seconds must be finite and > 0")
        self.max_missed_detections = max_missed_detections
        self.max_unconfirmed_seconds = max_unconfirmed_seconds
        self.clear()

    def clear(self):
        self.states = {}

    def update(self, items, detected_ids=None, detected_at=None, *, now=None):
        """Count one miss per committed SAM3 result, never per KCF/camera frame.

        No detector result (pending/error/discarded) is not a miss. An absolute
        detector age bound still removes stale boxes if detection stops working.
        New and established tracks use the same miss and age limits.
        """
        now = time.monotonic() if now is None else now
        detected_at = now if detected_at is None else detected_at
        states, output, retired = {}, [], set()
        for item in items:
            identity = item["track_id"]
            state = dict(self.states.get(identity,
                                         dict(hits=0, misses=0, confirmed_at=detected_at)))
            if detected_ids is not None:
                if identity in detected_ids:
                    state.update(hits=state["hits"] + 1, misses=0, confirmed_at=detected_at)
                else:
                    state["misses"] += 1
            age = max(0.0, now - state["confirmed_at"])
            if (age >= self.max_unconfirmed_seconds
                    or state["misses"] >= self.max_missed_detections):
                retired.add(identity)
                continue
            states[identity] = state
            output.append(dict(item,
                               track_status="coasting" if state["misses"] else "confirmed",
                               sam3_hits=state["hits"], sam3_misses=state["misses"],
                               sam3_age_seconds=age))
        self.states = states
        return output, retired


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
        self.track_ids = None
        self.initialized = False

    def update(self, items, frame_key, context):
        if not self.initialized or context != self.context:
            self.histories.clear()
            self.initialized = False
            self.context = context
        raw_boxes = tuple(tuple(item["bbox"]) for item in items)
        track_ids = tuple(item.get("track_id") for item in items)
        same_frame = self.initialized and frame_key == self.frame_key
        if not same_frame or raw_boxes != self.raw_boxes or track_ids != self.track_ids:
            # A background SAM3 handoff may change count/order on the same frame.
            # Cache by raw boxes too, so zip below can never truncate new objects.
            # Prefer object identity; legacy callers still match raw coordinates.
            matches = _match_boxes(self.raw_boxes or (), raw_boxes) if self.initialized else {}
            if (self.initialized and all(key is not None for key in track_ids)
                    and all(key is not None for key in self.track_ids)):
                previous = {key: index for index, key in enumerate(self.track_ids)}
                matches = {index: previous[key] for index, key in enumerate(track_ids)
                           if key in previous}
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
            self.track_ids = track_ids
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
