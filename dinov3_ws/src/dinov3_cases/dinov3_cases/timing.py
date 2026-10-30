"""Rolling throughput measured from completed inference results."""

from collections import deque
from contextlib import contextmanager
import threading
import time


class ModelTimings:
    """Latest successful stage durations, shared by workers, HTTP and ROS.

    A context revision prevents an old source/settings job from replacing new
    telemetry. Durations are wall time, not GPU kernel time.
    """

    SCOPES = {
        "nvdcf": "live_tracker_update_wall",
        "dinov3": "gallery_recognition_diagnostics_wall",
        "sam3": "worker_detect_wall",
    }

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.context = None
        self.revision = 0
        self.samples = self._empty()

    def _empty(self):
        return {name: dict(last_ms=None, source_frame_id=None, generation=None,
                           completed_at=None, busy=False)
                for name in self.SCOPES}

    def set_context(self, context):
        with self.lock:
            if context != self.context:
                self.context = context
                self.revision += 1
                self.samples = self._empty()
            return self.revision

    @contextmanager
    def measure(self, name, frame_key, revision=None):
        with self.lock:
            revision = self.revision if revision is None else revision
            if revision == self.revision:
                self.samples[name]["busy"] = True
        started = self.clock()
        result = {}
        try:
            yield result
        except BaseException:
            raise
        else:
            completed = self.clock()
            with self.lock:
                if revision == self.revision:
                    self.samples[name].update(
                        last_ms=result.get("duration_ms", (completed - started) * 1000),
                        source_frame_id=frame_key[1], generation=frame_key[0],
                        completed_at=completed,
                    )
        finally:
            with self.lock:
                if revision == self.revision:
                    self.samples[name]["busy"] = False

    def milliseconds(self):
        """Compact payload shared by the ROS topic and web display."""
        with self.lock:
            return {f"{name}_ms": sample["last_ms"]
                    for name, sample in self.samples.items()}

    def snapshot(self):
        with self.lock:
            now = self.clock()
            return {name: dict(
                scope=self.SCOPES[name], last_ms=sample["last_ms"],
                source_frame_id=sample["source_frame_id"], generation=sample["generation"],
                valid=sample["completed_at"] is not None, busy=sample["busy"],
                age_ms=(None if sample["completed_at"] is None
                        else max(0.0, (now - sample["completed_at"]) * 1000)),
            ) for name, sample in self.samples.items()}


class ResultRate:
    """Count completed intervals over roughly the last window of seconds.

    Keep the completion immediately before the window boundary so an interval
    crossing that boundary is measured in full, including slow detection frames.
    """

    def __init__(self, window_seconds=3.0):
        if window_seconds <= 0:
            raise ValueError("FPS window must be positive.")
        self.window_seconds = window_seconds
        self.completed = deque()
        self.context = None

    def update(self, now, context):
        if context != self.context:
            self.completed.clear()
            self.context = context
        if self.completed and now <= self.completed[-1]:
            self.completed.clear()
        self.completed.append(now)
        cutoff = now - self.window_seconds
        while len(self.completed) > 2 and self.completed[1] <= cutoff + 1e-9:
            self.completed.popleft()
        if len(self.completed) < 2:
            return None
        return (len(self.completed) - 1) / (now - self.completed[0])
