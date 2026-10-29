"""Background detection, detection-frame KCF seeding, and latest-frame catch-up."""

import math
import threading
import time
from .kcf_workers import PersistentKCFWorkers

from .kcf import (KCFProposals, older_frame, serialized, validate_detection_interval,
                  validate_parallel_settings)
from .tracking import TrackIdentities, TrackLifecycle, contained_track_ids


def validate_replay_limits(max_replay_frames):
    if type(max_replay_frames) is not int or max_replay_frames < 1:
        raise ValueError("max_replay_frames must be an integer >= 1")


class AsyncKCFProposals:
    """Seed candidates on the detection image and repeatedly track the latest frame.

    The main thread keeps active trackers until a candidate catches up. Matched
    objects switch individually; unmatched valid trackers survive until
    the detector miss limit retires them. All tracks use the same limits.
    Each update skips intermediate arrivals; the frame budget bounds catch-up work.
    Persistent owners replay one latest snapshot, then freeze READY. Foreground
    reconciles that snapshot to its current frame before assigning IDs/activation.
    Interval=1 disables async KCF.
    One persistent worker group is shared by foreground and replay; their tracker sets
    are disjoint until ready publication. Worker batches drain before publication.
    _update_lock protects foreground calls/merging/close, not background work;
    self.lock only protects scheduler metadata and never spans KCF computation.
    """

    def __init__(self, factory=None, wake=None, roi_padding=0.5,
                 max_replay_frames=8, max_roi_side=0,
                 max_missed_detections=3, lab_similarity_threshold=0.0,
                 lab_change_confirmations=3, containment_threshold=0.0,
                 parallel_enabled=False, parallel_workers=4):
        validate_parallel_settings(parallel_enabled, parallel_workers)
        self._update_lock = threading.RLock()
        self.workers = None
        self.parallel_settings = dict(parallel_enabled=parallel_enabled,
                                      parallel_workers=parallel_workers, workers=self.workers)
        validate_replay_limits(max_replay_frames)
        if (isinstance(containment_threshold, bool)
                or not isinstance(containment_threshold, (int, float))
                or not math.isfinite(containment_threshold)
                or not 0 <= containment_threshold <= 1):
            raise ValueError("containment_threshold must be finite and between 0 and 1")
        self.containment_threshold = containment_threshold
        self.max_replay_frames = max_replay_frames
        self.factory = factory
        self.roi_padding = roi_padding
        self.max_roi_side = max_roi_side
        self.lab_settings = dict(lab_similarity_threshold=lab_similarity_threshold,
                                 lab_change_confirmations=lab_change_confirmations)
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
        self.active = KCFProposals(factory, roi_padding=roi_padding,
                                   max_roi_side=max_roi_side,
                                   **self.lab_settings, **self.parallel_settings)
        self.frames_since_submit = 0
        self.force_detection = True
        self.last_metrics = {}
        self.last_recognition = None
        self.last_replay = {}
        self.last_handoff = {}
        self.identities = TrackIdentities()
        self.lifecycle = TrackLifecycle(max_missed_detections)
        if parallel_enabled:
            self.workers = PersistentKCFWorkers(parallel_workers)
            self.parallel_settings["workers"] = self.workers
            self.active.workers = self.workers
            self.active._owns_workers = False

    def _remove_contained_tracks(self):
        retired = contained_track_ids(self.active.proposals, self.containment_threshold)
        keep = [i for i, item in enumerate(self.active.proposals)
                if item["track_id"] not in retired]
        self.active.proposals = [self.active.proposals[i] for i in keep]
        # Interval=1 has no KCF trackers or ROIs.
        if self.active.trackers:
            self.active.keep(keep)
        self.identities.forget(retired)
        for identity in retired:
            self.lifecycle.states.pop(identity, None)

    def has_result(self):
        with self.lock:
            return (self.ready is not None or self.error is not None
                    or (self.force_detection and not self.busy))

    def scheduler_metrics(self):
        with self.lock:
            return dict(self.last_replay, workers=(self.workers.metrics() if self.workers else []),
                        handoff=dict(self.last_handoff), busy=self.busy,
                        initializing=not self.active.initialized)

    @serialized
    def close(self):
        with self.lock:
            self.closed = True
            self.version += 1
            self.latest = None
            ready, self.ready = self.ready, None
            thread = self.thread
        if thread is not None:
            thread.join()
        # No new foreground or replay submissions can now race shutdown.
        if ready is not None:
            ready[0].close()
        self.active.close()
        if self.workers is not None:
            self.workers.shutdown()

    def _job(self, version, serial, pixels, frame, detect, seed_frame_key, recognize=None):
        published = False
        candidate = None
        replay_frames = 0
        replay_started = None
        init_ms = 0.0
        seed_skipped_frames = 0
        catchup_skipped_frames = 0
        discard_reason = "cancelled"
        detected_at = time.monotonic()
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
            candidate = KCFProposals(self.factory, roi_padding=self.roi_padding, phase="replay",
                                     max_roi_side=self.max_roi_side,
                                     **self.lab_settings, **self.parallel_settings)
            candidate.detected_at = detected_at
            candidate.set_phase("replay")
            candidate.update(pixels, serial, version, 2**63, lambda: proposals,
                             retain_valid_tracks=True)
            init_ms = (time.monotonic() - init_started) * 1000
            if candidate.retry_detection:
                discard_reason = "init_failed"
                return
            # Record catch-up time separately from detection and initialization.
            replay_started = time.monotonic()
            while True:
                candidate.set_phase("ready")
                with self.lock:
                    if self.closed or version != self.version:
                        return
                    if (self.latest[0] == serial
                            or (self.workers is not None and replay_frames > 0)):
                        # Persistent owners can put replay behind foreground in
                        # the same mailbox indefinitely. Publish a frozen snapshot
                        # after one catch-up instead of chasing a moving target.
                        # Foreground MUST reconcile to its current serial before
                        # association/activation; READY boxes are never committed
                        # directly as current-frame results.
                        # All candidate workers have finished. After publishing,
                        # this job never accesses its trackers again.
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
                catchup_skipped_frames += next_serial - serial - 1
                serial = next_serial
                replay_frames += 1
                candidate.set_phase("replay")
                _, source = candidate.update(image, serial, version, 2**63, lambda: [],
                                             retain_valid_tracks=True)
                if source != "kcf":
                    discard_reason = "tracking_lost"
                    return
        except Exception as error:
            with self.lock:
                if version == self.version and not self.closed:
                    discard_reason = "error"
                    self.error = error
        finally:
            if candidate is not None and not published:
                candidate.close()
            with self.lock:
                self.busy = False
                if version == self.version and not self.closed:
                    self.last_replay = dict(
                        seed_frame_key=seed_frame_key,
                        ready_frame_id=serial if published else None,
                        tracker_context_generation=version,
                        seed_skipped_frames=seed_skipped_frames,
                        catchup_skipped_frames=catchup_skipped_frames,
                        init_ms=init_ms,
                        replay_frames=replay_frames,
                        replay_ms=(0.0 if replay_started is None else
                                   (time.monotonic()-replay_started)*1000),
                        discard_reason=None if published else discard_reason,
                    )
                    if not published and (not self.active.initialized
                                          or discard_reason != "frame_budget"):
                        self.force_detection = True
                    # On overload preserve the active tracker and normal submit
                    # interval, avoiding an immediate detector/replay retry loop.
            self.wake()

    @serialized
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
                self.active.close()
                if self.ready is not None:
                    self.ready[0].close()
                self.active = KCFProposals(self.factory, roi_padding=self.roi_padding,
                                           max_roi_side=self.max_roi_side,
                                     **self.lab_settings, **self.parallel_settings)
                self.latest = None
                self.last_key = None
                self.ready = self.error = None
                self.frames_since_submit = 0
                self.force_detection = True
                self.last_metrics = {}
                self.last_recognition = None
                self.last_replay = {}
                self.last_handoff = {}
                self.identities.clear()
                self.lifecycle.clear()
            # Reject late arrivals before changing latest/replay admission state.
            if older_frame(frame_key, self.last_key):
                # Do not pair current boxes with the caller's stale image.
                return None
            if self.error is not None:
                error, self.error = self.error, None
                raise error
            ready, self.ready = self.ready, None
            applied = ready is not None
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
            previous = self.active.proposals

            def synchronous():
                proposals, self.last_metrics = detect(frame if frame is not None else pixels)
                if recognize is not None:
                    proposals, self.last_recognition = recognize(pixels, proposals, frame_key)
                return proposals
            proposals, source = self.active.update(pixels, frame_key, identity, 1, synchronous)
            proposals, _ = self.identities.assign(proposals, previous)
            retired = self.identities.retire_unknown(proposals)
            proposals = [item for item in proposals if item["track_id"] not in retired]
            self.active.proposals = proposals
            self._remove_contained_tracks()
            return self.active.proposals, source

        result = None
        lost = False
        detected_ids = None
        detected_at = None
        if self.active.initialized:
            # Foreground and candidate keys use the same admitted-frame serials.
            proposals, source = self.active.update(
                pixels, serial, version, 2**63, lambda: [], retain_valid_tracks=True,
            )
            lost = source == "sam3" and changed
            self.identities.observe(proposals)
            result = proposals, source

        if applied:
            candidate, metrics, recognition = ready
            ready_frame_id = candidate.frame_key
            try:
                # Foreground and candidate must refer to this same image before
                # association, including arrivals after the worker published ready.
                candidate.set_phase("replay")
                candidate.update(pixels, serial, version, 2**63, lambda: [],
                                 retain_valid_tracks=True)
                live = self.active.proposals if self.active.initialized else []
                candidate.proposals, matches = self.identities.assign(candidate.proposals, live)
                detected_ids = {item["track_id"] for item in candidate.proposals}
                detected_at = candidate.detected_at
                candidate.activate()
                replaced = set(matches.values())
                for index, (proposal, tracker, roi) in enumerate(zip(
                        live, self.active.trackers, self.active.rois)):
                    if index not in replaced:
                        candidate.proposals.append(proposal)
                        candidate.trackers.append(tracker)
                        candidate.rois.append(roi)
                if len(replaced) < len(live):
                    candidate.refresh_requested |= self.active.refresh_requested
                # Retained handles move to candidate; replaced handles are retired.
                self.active.release_trackers([self.active.trackers[i] for i in replaced])
                self.active.trackers, self.active.rois = [], []
                self.active.close()
                self.active = candidate
                self.last_handoff = dict(ready_frame_id=ready_frame_id, frame_id=serial,
                                         skipped_frames=max(0, serial-ready_frame_id-1),
                                         tracker_context_generation=version)
                self.last_metrics, self.last_recognition = metrics, recognition
                self.identities.observe(candidate.proposals)
                result = [dict(p, bbox=list(p["bbox"])) for p in candidate.proposals], "sam3"
            except Exception:
                candidate.close()
                raise

        if result is not None:
            # One containment pass after candidate/retained-track merging, also
            # covering overlaps introduced by subsequent foreground KCF motion.
            self._remove_contained_tracks()
            # Match unknown candidates before filtering them: otherwise the old
            # known tracker would be retained as an unmatched SAM3 survivor.
            unknown_ids = self.identities.retire_unknown(self.active.proposals)
            proposals, retired = self.lifecycle.update(
                [item for item in self.active.proposals
                 if item["track_id"] not in unknown_ids],
                detected_ids, detected_at, now=time.monotonic())
            retired |= unknown_ids
            keep = [index for index, item in enumerate(self.active.proposals)
                    if item["track_id"] not in retired]
            self.active.keep(keep)
            self.active.proposals = proposals
            self.identities.forget(retired)
            result = [dict(p, bbox=list(p["bbox"])) for p in proposals], result[1]

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
