"""Periodic full depth-box measurements with one active and one latest job."""

import copy
import math
import threading
import time

from .tracking import _match_boxes, _iou


def validate_bbox3d_schedule(interval, max_age):
    if type(interval) is not int or interval < 1:
        raise ValueError("bbox3d_interval must be an integer >= 1.")
    if (isinstance(max_age, bool) or not isinstance(max_age, (int, float))
            or not math.isfinite(max_age) or max_age <= 0):
        raise ValueError("bbox3d_max_age must be finite and > 0.")


class AsyncBBox3D:
    """Foreground owns object association; the worker only measures snapshots.

    IDs are private, geometric associations between adjacent sets, independent
    of detector refreshes and list ordering. Lost IDs are never reused. Crossing
    or indistinguishable objects still need a persistent visual tracker ID.
    """

    def __init__(self, measure, *, interval=5, max_age=1.5, wake=None, on_error=None):
        validate_bbox3d_schedule(interval, max_age)
        self.measure = measure
        self.interval, self.max_age = interval, max_age
        self.wake = wake or (lambda: None)
        self.on_error = on_error or (lambda error: None)
        self.condition = threading.Condition()
        self.closed = False
        self.context = None
        self.epoch = 0
        self.frame_key = None
        self.frame_count = 0
        self.next_id = 0
        self.previous = []
        self.ids = []
        self.pending = None
        self.cache = {}
        self.version = self.consumed = 0
        self.last_ms = None
        self.camera_info = None
        self.thread = threading.Thread(target=self._run, name="depth-bbox", daemon=True)
        self.thread.start()

    def has_result(self):
        with self.condition:
            return self.version != self.consumed

    def update(self, detections, inputs, *, header, frame_key, context, captured_at=None):
        """Nonblocking submit/read; first frame then every N unique frames."""
        now = time.monotonic()
        with self.condition:
            if context != self.context:
                self.context = context
                self.epoch += 1
                self.frame_key = None
                self.frame_count = 0
                self.previous, self.ids = [], []
                self.cache.clear()
                self.pending = None
                self.camera_info = None
            boxes = [list(item["bbox"]) for item in detections]
            matches = _match_boxes(self.previous, boxes)
            ids = []
            for index, box in enumerate(boxes):
                previous = matches.get(index)
                # Reject weak matches; proximity alone is not enough for a pose.
                if previous is not None and _iou(self.previous[previous], box) >= .2:
                    ids.append(self.ids[previous])
                else:
                    self.next_id += 1
                    ids.append(self.next_id)
            self.previous, self.ids = boxes, ids
            live = set(ids)
            self.cache = {key: value for key, value in self.cache.items() if key in live}
            changed = frame_key != self.frame_key
            due = changed and self.frame_count % self.interval == 0
            if changed:
                self.frame_count += 1
                self.frame_key = frame_key
            if not ids:
                self.pending = None
            elif due and not self.closed:
                # ROS messages and detections are immutable after submission in
                # the node. Copy small metadata, retain owned depth message bytes.
                proposals = [dict(bbox=list(item["bbox"])) for item in detections]
                source = dict(frame_key=list(frame_key), frame_id=header.frame_id,
                              stamp=dict(sec=header.stamp.sec, nanosec=header.stamp.nanosec))
                self.pending = (self.epoch, ids[:], proposals, inputs, source,
                                now if captured_at is None else captured_at)
                self.condition.notify()
            self.consumed = self.version
            keyed = [dict(item, _bbox3d_key=(self.epoch, identity))
                     for item, identity in zip(detections, ids)]
            output = self._latest_locked(keyed, now)
            metrics = dict(interval=self.interval, measurement_ms=self.last_ms,
                           pending=self.pending is not None)
            return output, self.camera_info, metrics

    def latest(self, detections):
        """Refresh at render/publish time without advancing the frame counter."""
        with self.condition:
            return self._latest_locked(detections, time.monotonic()), self.camera_info

    def _latest_locked(self, detections, now):
        output = []
        for item in detections:
            item = dict(item)
            key = item.get("_bbox3d_key")
            if key is None:
                output.append(item)
                continue
            for field in ("bbox3d", "bbox3d_source", "bbox3d_age_seconds"):
                item.pop(field, None)
            cached = self.cache.get(key[1]) if key[0] == self.epoch else None
            if cached is None:
                item["bbox3d_status"] = "pending_background_measurement"
            elif now - cached[2] > self.max_age:
                item["bbox3d_status"] = "stale_background_measurement"
            else:
                measured, source, submitted = cached
                item.update(copy.deepcopy(measured))
                item["bbox3d_source"] = copy.deepcopy(source)
                item["bbox3d_age_seconds"] = max(0., now - submitted)
            output.append(item)
        return output

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                job, self.pending = self.pending, None
            epoch, ids, proposals, inputs, source, submitted = job
            started = time.monotonic()
            try:
                self.measure(proposals, inputs)
            except Exception as error:
                for item in proposals:
                    item.pop("bbox3d", None)
                    item["bbox3d_status"] = "background_measurement_failed"
                self.on_error(error)
            with self.condition:
                if not self.closed and epoch == self.epoch:
                    live = set(self.ids)
                    for identity, item in zip(ids, proposals):
                        if identity in live:
                            measured = {key: value for key, value in item.items()
                                        if key in ("bbox3d", "bbox3d_status")}
                            self.cache[identity] = (measured, source, submitted)
                    self.camera_info = inputs[1] if inputs is not None else None
                    self.last_ms = (time.monotonic() - started) * 1000
                    self.version += 1
            self.wake()

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify_all()
        self.thread.join(timeout=3)
