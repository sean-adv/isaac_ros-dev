"""Background SAM3 detection with KCF replay to the latest processed frame."""

from collections import deque
import threading
import math
import time

from .kcf import KCFProposals, validate_detection_interval


def validate_replay_limits(max_replay_frames, replay_budget_ms):
    if type(max_replay_frames) is not int or max_replay_frames < 1:
        raise ValueError("max_replay_frames must be an integer >= 1")
    if (isinstance(replay_budget_ms, bool) or not isinstance(replay_budget_ms, (int, float))
            or not math.isfinite(replay_budget_ms) or replay_budget_ms <= 0):
        raise ValueError("replay_budget_ms must be finite and > 0")


class AsyncKCFProposals:
    """One detection job, bounded owned RGB history, and atomic tracker handoff.

    Only the main thread uses the active trackers. A background job owns a new
    tracker set until it has replayed every buffered frame. It publishes that
    set under the same lock used to append frames. During initial detection,
    incoming frames are not admitted to tracking history; the first handoff
    advances the seeded model against the current image. Steady-state replay
    has a frame count and cooperative time budget. Old boxes are never
    initialized directly against a newer image. Interval=1 retains synchronous
    detection without KCF.
    """

    def __init__(self, factory=None, wake=None, max_frames=32, max_bytes=64 << 20,
                 roi_padding=0.5, max_replay_frames=8, replay_budget_ms=150.0):
        validate_replay_limits(max_replay_frames, replay_budget_ms)
        self.max_replay_frames = max_replay_frames
        self.replay_budget_ms = replay_budget_ms
        self.factory = factory
        self.roi_padding = roi_padding
        self.wake = wake or (lambda: None)
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self.lock = threading.Lock()
        self.closed = False
        self.busy = False
        self.thread = None
        self.context = None
        self.version = 0
        self.history = deque()
        self.history_bytes = 0
        self.serial = 0
        self.last_key = None
        self.ready = self.error = None
        self.active = KCFProposals(factory, roi_padding=roi_padding)
        self.frames_since_submit = 0
        self.force_detection = True
        self.last_metrics = {}
        self.last_replay = {}

    def has_result(self):
        with self.lock:
            return (self.ready is not None or self.error is not None
                    or (self.force_detection and not self.busy))

    def waiting_for_initial_detection(self, context, interval):
        with self.lock:
            return (self.context == (context, interval) and self.busy
                    and not self.active.initialized
                    and self.ready is None and self.error is None)

    def scheduler_metrics(self):
        with self.lock:
            return dict(self.last_replay, busy=self.busy,
                        initializing=not self.active.initialized)

    def close(self):
        with self.lock:
            self.closed = True
            self.version += 1
            self.history.clear()
            self.history_bytes = 0
            self.ready = None
            thread = self.thread
        if thread is not None:
            thread.join(timeout=3)

    def _job(self, version, serial, pixels, frame, detect):
        published = False
        replay_frames = 0
        replay_started = None
        discard_reason = "cancelled"
        try:
            try:
                proposals, metrics = detect(frame if frame is not None else pixels)
            finally:
                if frame is not None:
                    frame.release()
            with self.lock:
                if self.closed or version != self.version:
                    return
            candidate = KCFProposals(self.factory, roi_padding=self.roi_padding)
            candidate.update(pixels, serial, version, 2**63, lambda: proposals)
            if candidate.retry_detection:
                discard_reason = "init_failed"
                return
            # Detection and model initialization do not consume replay budget.
            replay_started = time.monotonic()
            while True:
                with self.lock:
                    if self.closed or version != self.version:
                        return
                    # Every processed frame after the detector's input is needed.
                    if self.history and self.history[0][0] > serial + 1:
                        discard_reason = "history_gap"
                        return
                    pending = [(n, image) for n, image in self.history if n > serial]
                    if not pending:
                        self.ready = (candidate, metrics)
                        published = True
                        return
                for next_serial, image in pending:
                    # Check before each KCF call: one non-preemptible update may
                    # exceed the time budget, but no additional call is started.
                    with self.lock:
                        if self.closed or version != self.version:
                            return
                    if replay_frames >= self.max_replay_frames:
                        discard_reason = "frame_budget"
                        return
                    if (time.monotonic() - replay_started)*1000 >= self.replay_budget_ms:
                        discard_reason = "time_budget"
                        return
                    if next_serial != serial + 1:
                        discard_reason = "history_gap"
                        return
                    serial = next_serial
                    replay_frames += 1
                    _, source = candidate.update(image, serial, version, 2**63, lambda: [])
                    if source != "kcf":  # Lost during replay: request a fresh image.
                        discard_reason = "tracking_lost"
                        return
        except Exception as error:
            with self.lock:
                if version == self.version and not self.closed:
                    discard_reason = "error"
                    self.error = error
        finally:
            with self.lock:
                self.busy = False
                if version == self.version and not self.closed:
                    self.last_replay = dict(
                        replay_frames=replay_frames,
                        replay_ms=(0.0 if replay_started is None else
                                   (time.monotonic()-replay_started)*1000),
                        discard_reason=None if published else discard_reason,
                    )
                    if not published and discard_reason not in ("frame_budget", "time_budget"):
                        self.force_detection = True
                    # On overload preserve the active tracker and normal submit
                    # interval, avoiding an immediate detector/replay retry loop.
            self.wake()

    def update(self, pixels, frame_key, context, interval, detect, *, frame=None):
        """Return None until first detection; detect(input) returns (boxes, metrics)."""
        validate_detection_interval(interval)
        with self.lock:
            if self.closed:
                return None
            identity = (context, interval)
            if identity != self.context:
                self.context = identity
                self.version += 1
                self.active = KCFProposals(self.factory, roi_padding=self.roi_padding)
                self.history.clear()
                self.history_bytes = 0
                self.last_key = None
                self.ready = self.error = None
                self.frames_since_submit = 0
                self.force_detection = True
                self.last_metrics = {}
                self.last_replay = {}
            if self.error is not None:
                error, self.error = self.error, None
                raise error
            if interval > 1 and not self.active.initialized and self.busy:
                # Camera callbacks still keep the newest frame in CaseNode.
                # Unprocessed arrivals must not become an unbounded bootstrap
                # replay backlog or consume history/submit interval counters.
                return None
            applied = self.ready is not None
            if applied:
                self.active, self.last_metrics = self.ready
                self.ready = None
            changed = frame_key != self.last_key
            # Only asynchronous detection consumes replay history.
            if changed and interval > 1:
                self.last_key = frame_key
                self.serial += 1
                # Independent storage: shared-memory slots can be reused meanwhile.
                owned = pixels.copy()
                self.history.append((self.serial, owned))
                self.history_bytes += owned.nbytes
                while len(self.history) > 1 and (
                    len(self.history) > self.max_frames or self.history_bytes > self.max_bytes
                ):
                    self.history_bytes -= self.history.popleft()[1].nbytes
                self.frames_since_submit += 1
            serial = self.serial
            version = self.version

        if interval == 1:
            def synchronous():
                proposals, self.last_metrics = detect(frame if frame is not None else pixels)
                return proposals
            return self.active.update(pixels, frame_key, identity, 1, synchronous)

        result = None
        lost = False
        if self.active.initialized:
            # Candidate keys are history serials, matching active updates exactly.
            proposals, source = self.active.update(
                pixels, serial, version, 2**63, lambda: [], retain_valid_tracks=True,
            )
            lost = source == "sam3" and changed
            if applied:
                source = "sam3"
            result = proposals, source

        with self.lock:
            self.force_detection |= lost or self.active.refresh_requested
            due = self.force_detection or self.frames_since_submit >= interval
            if due and not self.busy and not self.closed:
                self.busy = True
                self.force_detection = False
                self.frames_since_submit = 0
                snapshot = self.history[-1][1]
                lease = frame.retain() if frame is not None else None
                self.thread = threading.Thread(
                    target=self._job, args=(version, serial, snapshot, lease, detect),
                    name="sam3-kcf-replay", daemon=True,
                )
                self.thread.start()
        return result
