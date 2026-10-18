"""Background detection, detection-frame KCF seeding, and latest-frame catch-up."""

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
    """Seed candidates on the detection image and repeatedly track the latest frame.

    The main thread keeps active trackers until a candidate catches up and is
    handed off atomically. Each update skips intermediate arrivals; frame and
    time budgets bound catch-up work. Interval=1 disables async KCF.
    """

    def __init__(self, factory=None, wake=None, roi_padding=0.5,
                 max_replay_frames=8, replay_budget_ms=150.0):
        validate_replay_limits(max_replay_frames, replay_budget_ms)
        self.max_replay_frames = max_replay_frames
        self.replay_budget_ms = replay_budget_ms
        self.factory = factory
        self.roi_padding = roi_padding
        self.wake = wake or (lambda: None)
        self.lock = threading.Lock()
        self.closed = False
        self.busy = False
        self.thread = None
        self.context = None
        self.version = 0
        self.latest = None
        self.serial = 0
        self.last_key = None
        self.ready = self.error = None
        self.active = KCFProposals(factory, roi_padding=roi_padding)
        self.frames_since_submit = 0
        self.force_detection = True
        self.last_metrics = {}
        self.last_recognition = None
        self.last_replay = {}

    def has_result(self):
        with self.lock:
            return (self.ready is not None or self.error is not None
                    or (self.force_detection and not self.busy))

    def scheduler_metrics(self):
        with self.lock:
            return dict(self.last_replay, busy=self.busy,
                        initializing=not self.active.initialized)

    def close(self):
        with self.lock:
            self.closed = True
            self.version += 1
            self.latest = None
            self.ready = None
            thread = self.thread
        if thread is not None:
            thread.join(timeout=3)

    def _job(self, version, serial, pixels, frame, detect, seed_frame_key, recognize=None):
        published = False
        replay_frames = 0
        replay_started = None
        init_ms = 0.0
        seed_skipped_frames = 0
        catchup_skipped_frames = 0
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
            recognition = None
            if recognize is not None:
                proposals, recognition = recognize(pixels, proposals, seed_frame_key)
            with self.lock:
                if self.closed or version != self.version:
                    return
            init_started = time.monotonic()
            candidate = KCFProposals(self.factory, roi_padding=self.roi_padding)
            candidate.update(pixels, serial, version, 2**63, lambda: proposals)
            init_ms = (time.monotonic() - init_started) * 1000
            if candidate.retry_detection:
                discard_reason = "init_failed"
                return
            # Detection and model initialization do not consume replay budget.
            replay_started = time.monotonic()
            while True:
                with self.lock:
                    if self.closed or version != self.version:
                        return
                    if self.latest[0] == serial:
                        self.ready = (candidate, metrics, recognition)
                        published = True
                        return
                    next_serial, image = self.latest
                # Take the newest admitted frame on every iteration. The local
                # reference keeps its owned pixels alive while newer ones arrive.
                # Never hold the foreground lock during KCF work.
                if replay_frames >= self.max_replay_frames:
                    discard_reason = "frame_budget"
                    return
                if (time.monotonic() - replay_started)*1000 >= self.replay_budget_ms:
                    discard_reason = "time_budget"
                    return
                catchup_skipped_frames += next_serial - serial - 1
                serial = next_serial
                replay_frames += 1
                _, source = candidate.update(image, serial, version, 2**63, lambda: [])
                if source != "kcf":
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
                        seed_frame_key=seed_frame_key,
                        seed_skipped_frames=seed_skipped_frames,
                        catchup_skipped_frames=catchup_skipped_frames,
                        init_ms=init_ms,
                        replay_frames=replay_frames,
                        replay_ms=(0.0 if replay_started is None else
                                   (time.monotonic()-replay_started)*1000),
                        discard_reason=None if published else discard_reason,
                    )
                    if not published and (not self.active.initialized
                                          or discard_reason not in ("frame_budget", "time_budget")):
                        self.force_detection = True
                    # On overload preserve the active tracker and normal submit
                    # interval, avoiding an immediate detector/replay retry loop.
            self.wake()

    def update(self, pixels, frame_key, context, interval, detect, *, frame=None, recognize=None,
               wait_for_initial_result=False):
        """Run detection and optional recognition in one scheduled job.

        recognize(seed_pixels, boxes, frame_key) returns (classified_boxes, payload).
        Labels travel with each KCF proposal; payload commits only at handoff.
        Interval=1 runs both models in the caller with KCF disabled.
        wait_for_initial_result waits on the first job before admitting more frames,
        so cold-start inference can hand off before admitting more frames.
        Camera capture/preview may continue independently of this caller.
        """
        validate_detection_interval(interval)
        with self.lock:
            if self.closed:
                return None
            identity = (context, interval)
            if identity != self.context:
                self.context = identity
                self.version += 1
                self.active = KCFProposals(self.factory, roi_padding=self.roi_padding)
                self.latest = None
                self.last_key = None
                self.ready = self.error = None
                self.frames_since_submit = 0
                self.force_detection = True
                self.last_metrics = {}
                self.last_recognition = None
                self.last_replay = {}
            if self.error is not None:
                error, self.error = self.error, None
                raise error
            applied = self.ready is not None
            if applied:
                self.active, self.last_metrics, self.last_recognition = self.ready
                self.ready = None
            changed = frame_key != self.last_key
            # Only asynchronous detection needs owned replay images.
            if changed and interval > 1:
                self.last_key = frame_key
                self.serial += 1
                # One owned latest image, independent of shared-memory leases.
                self.latest = (self.serial, pixels.copy())
                if self.active.initialized:
                    self.frames_since_submit += 1
            serial = self.serial
            version = self.version

        if interval == 1:
            def synchronous():
                proposals, self.last_metrics = detect(frame if frame is not None else pixels)
                if recognize is not None:
                    proposals, self.last_recognition = recognize(pixels, proposals, frame_key)
                return proposals
            return self.active.update(pixels, frame_key, identity, 1, synchronous)

        result = None
        lost = False
        if self.active.initialized:
            # Foreground and candidate keys use the same admitted-frame serials.
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
                snapshot = self.latest[1]
                lease = frame.retain() if frame is not None else None
                self.thread = threading.Thread(
                    target=self._job, args=(version, serial, snapshot, lease, detect, frame_key, recognize),
                    name="sam3-kcf-replay", daemon=True,
                )
                self.thread.start()
            initial_job = (self.thread if wait_for_initial_result
                           and not self.active.initialized and self.busy else None)
        if initial_job is not None:
            # Never join under the scheduler lock: the job needs it to publish.
            initial_job.join()
            return self.update(pixels, frame_key, context, interval, detect,
                               frame=frame, recognize=recognize)
        return result
