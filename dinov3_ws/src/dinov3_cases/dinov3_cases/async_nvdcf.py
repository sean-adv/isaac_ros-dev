"""Whole-frame NvDCF handoff, optionally publishing boxes before DINO labels."""
import logging
import queue
import threading
import time

from .nvdcf import NvDCF, DEFAULT_CONFIG, validate_detection_interval
from .tracking import TrackIdentities, TrackLifecycle, contained_track_ids


def validate_missed_frames(value):
    if type(value) is not int or value < 0:
        raise ValueError('nvdcf_max_missed_frames must be an integer >= 0')


class _TrackerReaper:
    """Close detached contexts on one worker; bound outstanding native resources.

    Saturation applies backpressure rather than leaking processes or spawning
    unbounded cleanup threads. Normal handoffs enqueue one context per job.
    The scheduler joins all producers before close() drains this worker.
    """
    def __init__(self):
        self.queue = queue.Queue(maxsize=8)
        self.thread = None
        self.start_lock = threading.Lock()

    def retire(self, tracker):
        with self.start_lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='nvdcf-cleanup', daemon=True)
                self.thread.start()
        self.queue.put(tracker)

    def _run(self):
        while True:
            tracker = self.queue.get()
            try:
                if tracker is None:
                    return
                tracker.close()
            except Exception:
                logging.getLogger(__name__).exception('NvDCF context cleanup failed')
            finally:
                self.queue.task_done()

    def drain(self):
        self.queue.join()

    def close(self):
        if self.thread is not None:
            self.queue.put(None)
            self.thread.join()
            self.thread = None


_CLASSIFICATION_FIELDS = ('label', 'score', 'dino_score', 'lab_score', 'candidate',
                          'margin', 'reason', 'reference')


class _NvDCFTracks:
    """Own disjoint public tracks across native contexts without reseeding them.

    Native IDs are local to each context. Hidden owners retain metadata and their
    last-seen serial until the miss grace expires; they are never rendered.
    """
    def __init__(self, contexts=(), *, max_missed_frames=30, serial=0, last_seen=None, retire=None):
        self.retire = retire or (lambda tracker: tracker.close())
        self.contexts = list(contexts)
        self.proposals = [p for tracker in self.contexts for p in tracker.proposals]
        self.max_missed_frames = max_missed_frames
        self.last_seen = dict(last_seen or {})
        self.last_seen.update({p['track_id']: serial for p in self.proposals})
        self.newly_lost = set()
        self.expired = set()

    @property
    def hidden(self):
        visible = {p['track_id'] for p in self.proposals}
        return [dict(p) for tracker in self.contexts for p in tracker.metadata.values()
                if 'track_id' in p and p['track_id'] not in visible]

    @property
    def owned(self):
        return [*self.proposals, *self.hidden]

    def keep(self, proposals):
        by_id = {p['track_id']: p for p in proposals}
        visible = {p['track_id'] for p in self.proposals}
        retained = []
        for tracker in self.contexts:
            tracker.metadata = {
                key: dict(by_id[value['track_id']])
                for key, value in tracker.metadata.items()
                if value.get('track_id') in by_id
            }
            tracker.proposals = [by_id[p['track_id']] for p in tracker.proposals
                                 if p['track_id'] in by_id]
            if tracker.metadata:
                retained.append(tracker)
            else:
                self.retire(tracker)
        self.contexts = retained
        self.proposals = [p for p in proposals if p['track_id'] in visible]
        self.last_seen = {key: value for key, value in self.last_seen.items() if key in by_id}

    def update(self, pixels, serial):
        previous = {p['track_id'] for p in self.proposals}
        self.proposals = [p for tracker in self.contexts
                          for p in tracker.update(pixels, serial)]
        visible = {p['track_id'] for p in self.proposals}
        self.last_seen.update({key: serial for key in visible})
        self.newly_lost = previous - visible
        hidden = self.hidden
        self.expired = {p['track_id'] for p in hidden
                        if serial - self.last_seen[p['track_id']] > self.max_missed_frames}
        self.keep([*self.proposals,
                   *(p for p in hidden if p['track_id'] not in self.expired)])

    def close(self):
        for tracker in self.contexts:
            self.retire(tracker)
        self.contexts = []
        self.proposals = []
        self.last_seen = {}


class AsyncNvDCFProposals:
    """Keep live NvDCF running while a detector generation is prepared.

    Associate generations only after both reach the same live frame. Matched
    tracks switch to the candidate; unmatched visible tracks keep their original
    context until the detector miss limit retires them. Hidden tracks keep their
    context for the frame grace and pause detector miss counting. Replay uses the
    newest admitted image, without frame history.
    """
    def __init__(self, wake=None, config=DEFAULT_CONFIG, width=640, height=384,
                 containment_threshold=0.0, factory=None, max_missed_detections=3,
                 max_missed_frames=30, defer_recognition=False):
        if not 0 <= containment_threshold <= 1:
            raise ValueError('containment_threshold must be in [0, 1]')
        validate_missed_frames(max_missed_frames)
        self.max_missed_frames = max_missed_frames
        self.defer_recognition = defer_recognition
        self.geometry_ready = threading.Event()
        self.recognized = None
        self.recognition_error = None
        self.factory = factory or (lambda: NvDCF(config, width, height))
        self.wake = wake or (lambda: None)
        self.containment_threshold = containment_threshold
        self.lock = threading.Lock()
        self._update_lock = threading.RLock()
        self._reaper = _TrackerReaper()
        self.context = None
        self.version = 0
        self.serial = 0
        self.latest = None
        self.last_key = None
        self.active = None
        self.ready = None
        self.error = None
        self.thread = None
        self.closed = False
        self.busy = False
        self.force_detection = True
        self.frames_since_submit = 0
        self.identities = TrackIdentities()
        self.lifecycle = TrackLifecycle(max_missed_detections)
        self.last_metrics = {}
        self.last_recognition = None
        self.last_replay = {}
        self.last_handoff = {}
        self.last_tracking = {}

    def has_result(self):
        with self.lock:
            return (self.ready is not None or self.recognized is not None
                    or self.error is not None or (self.force_detection and not self.busy))

    def scheduler_metrics(self):
        with self.lock:
            return dict(self.last_replay, backend='nvdcf', busy=self.busy,
                        initializing=self.active is None, handoff=dict(self.last_handoff),
                        recognition_error=self.recognition_error,
                        **self.last_tracking)

    def _job(self, version, serial, pixels, lease, detect, recognize, seed_key):
        candidate = None
        started = time.monotonic()
        published = False
        deferred = self.defer_recognition and recognize is not None
        try:
            try:
                proposals, metrics = detect(lease if lease is not None else pixels)
            finally:
                if lease is not None:
                    lease.release()
            recognition = None
            seed_proposals = proposals
            if deferred:
                seed_proposals = [dict(p, bbox=list(p['bbox']),
                                       _recognition_key=(version, serial, index))
                                  for index, p in enumerate(proposals)]
                proposals = [dict(p, label='unknown', score=None,
                                  reason='recognition_pending', recognition_pending=True)
                             for p in seed_proposals]
            elif recognize is not None:
                proposals, recognition = recognize(pixels, proposals, seed_key)
            with self.lock:
                if self.closed or self.version != version:
                    return
            create_started = time.monotonic()
            candidate = self.factory()
            created_at = time.monotonic()
            candidate.detected_at = started
            candidate.update(pixels, serial, proposals)
            seeded_at = time.monotonic()
            with self.lock:
                if self.closed or self.version != version:
                    return
                replay_serial, replay_image = self.latest
            if replay_serial != serial:
                candidate.update(replay_image, replay_serial)
            with self.lock:
                if self.closed or self.version != version:
                    return
                self.ready = (candidate, metrics, recognition)
                candidate = None
                published = True
                self.last_replay = dict(seed_frame_key=seed_key, ready_frame_id=replay_serial,
                                        catchup_skipped_frames=max(0, replay_serial-serial-1),
                                        replay_frames=int(replay_serial != serial),
                                        candidate_create_ms=(created_at-create_started)*1000,
                                        candidate_seed_ms=(seeded_at-created_at)*1000,
                                        candidate_catchup_ms=(time.monotonic()-seeded_at)*1000,
                                        geometry_ready_ms=(time.monotonic()-started)*1000,
                                        job_ms=(time.monotonic()-started)*1000)
            # Transfer geometry ownership before DINO. No native tracker is
            # touched by the recognition thread after this point.
            self.geometry_ready.set()
            self.wake()
            if deferred:
                classified, recognition = recognize(pixels, seed_proposals, seed_key)
                classes = {
                    p['_recognition_key']: {k: p[k] for k in _CLASSIFICATION_FIELDS if k in p}
                    for p in classified
                }
                with self.lock:
                    if not self.closed and self.version == version:
                        self.recognized = (classes, recognition)
                        self.last_replay["job_ms"] = (time.monotonic()-started)*1000

        except Exception as error:
            with self.lock:
                if not self.closed and self.version == version:
                    if deferred and published:
                        self.recognition_error = str(error)
                    else:
                        self.error = error
        finally:
            if candidate is not None:
                self._reaper.retire(candidate)
            with self.lock:
                self.busy = False
                if self.version != version and not self.closed:
                    self.force_detection = True
            self.geometry_ready.set()
            self.wake()

    def _apply_recognition(self, classes):
        """Update surviving seed owners by immutable key, never list position.

        Apply only classification fields: delayed seed boxes cannot rewind live
        geometry, revive removed objects, or label a newer detector generation.
        """
        if self.active is None:
            return False
        applied = False
        for tracker in self.active.contexts:
            for item in [*tracker.metadata.values(), *tracker.proposals]:
                classification = classes.get(item.get("_recognition_key"))
                if classification is not None:
                    for key in _CLASSIFICATION_FIELDS:
                        item.pop(key, None)
                    item.update(classification, recognition_pending=False)
                    applied = True
        # Public outputs can be separate dictionaries from native metadata.
        for item in self.active.proposals:
            classification = classes.get(item.get("_recognition_key"))
            if classification is not None:
                for key in _CLASSIFICATION_FIELDS:
                    item.pop(key, None)
                item.update(classification, recognition_pending=False)
        return applied

    def _filter(self, detected_ids=None, detected_at=None):
        hidden = self.active.hidden
        # Occluded owners use the frame grace, not the SAM3 miss limit. Preserve
        # their lifecycle counters without counting stale boxes as detections.
        hidden_states = {p['track_id']: self.lifecycle.states[p['track_id']]
                         for p in hidden if p['track_id'] in self.lifecycle.states}
        retired = contained_track_ids(self.active.proposals, self.containment_threshold)
        retired |= self.identities.retire_unknown(
            [p for p in self.active.proposals if not p.get("recognition_pending")])
        proposals, expired = self.lifecycle.update(
            [p for p in self.active.proposals if p['track_id'] not in retired],
            detected_ids, detected_at,
        )
        retired |= expired
        self.lifecycle.states.update(hidden_states)
        self.active.keep([*proposals, *hidden])
        self.identities.forget(retired)
        self.identities.observe(self.active.owned)
        with self.lock:
            self.last_tracking = dict(visible_tracks=len(self.active.proposals),
                                      shadow_tracks=len(self.active.hidden),
                                      native_contexts=len(self.active.contexts))

    def update(self, pixels, frame_key, context, interval, detect, *, frame=None,
               recognize=None, wait_for_initial_result=False):
        with self._update_lock:
            return self._update(pixels, frame_key, context, interval, detect, frame=frame,
                                recognize=recognize, wait_for_initial_result=wait_for_initial_result)

    def _update(self, pixels, frame_key, context, interval, detect, *, frame=None,
                recognize=None, wait_for_initial_result=False):
        validate_detection_interval(interval)
        with self.lock:
            if self.closed:
                return None
            identity = (context, interval)
            if identity != self.context:
                self.context = identity
                self.version += 1
                if self.active is not None:
                    self.active.close()
                if self.ready is not None:
                    self._reaper.retire(self.ready[0])
                self.active = self.ready = None
                self.error = None
                self.latest = None
                self.last_key = None
                self.frames_since_submit = 0
                self.force_detection = True
                self.last_recognition = None
                self.recognized = None
                self.recognition_error = None
                self.last_metrics = {}
                self.last_replay = {}
                self.last_handoff = {}
                self.last_tracking = {}
                self.identities.clear()
                self.lifecycle.clear()
                self._sync_live = []
            if self.last_key is not None and frame_key < self.last_key:
                return None
            if self.error is not None:
                error, self.error = self.error, None
                self.force_detection = True
                raise error
            changed = frame_key != self.last_key
            if changed:
                self.serial += 1
                self.last_key = frame_key
                self.latest = (self.serial, pixels.copy())
                if self.active is not None:
                    self.frames_since_submit += 1
            serial, version = self.serial, self.version
            ready, self.ready = self.ready, None
            recognized, self.recognized = self.recognized, None

        if interval == 1:
            proposals, self.last_metrics = detect(frame if frame is not None else pixels)
            if recognize is not None:
                proposals, self.last_recognition = recognize(pixels, proposals, frame_key)
            proposals, _ = self.identities.assign(proposals, getattr(self, '_sync_live', []))
            retired = contained_track_ids(proposals, self.containment_threshold)
            retired |= self.identities.retire_unknown(proposals)
            self._sync_live = [dict(p, bbox_source='sam3') for p in proposals if p['track_id'] not in retired]
            return self._sync_live, 'sam3'

        result = None
        detected_ids = None
        detected_at = None
        lost = False
        if self.active is not None:
            self.active.update(pixels, serial)
            self.identities.forget(self.active.expired)
            for identity in self.active.expired:
                self.lifecycle.states.pop(identity, None)
            result = self.active.proposals, 'nvdcf'
            lost = bool(self.active.newly_lost)

        if ready is not None:
            candidate, metrics, recognition = ready
            try:
                ready_serial = candidate.frame_key
                candidate.update(pixels, serial)
                live = self.active.owned if self.active is not None else []
                candidate.proposals, matches = self.identities.assign(candidate.proposals, live)
                if self.defer_recognition:
                    for index, old in matches.items():
                        item = candidate.proposals[index]
                        if item.get('recognition_pending'):
                            previous = live[old]
                            item.update({k: previous[k] for k in _CLASSIFICATION_FIELDS if k in previous})
                detected_ids = {p['track_id'] for p in candidate.proposals}
                detected_at = candidate.detected_at
                for item in candidate.proposals:
                    candidate.metadata[item['nvdcf_id']] = dict(item)
                # Move unmatched owners intact; native contexts cannot transfer
                # individual tracker handles between generations.
                replaced = set(matches.values())
                retained = [p for index, p in enumerate(live) if index not in replaced]
                contexts = []
                last_seen = {}
                if self.active is not None:
                    self.active.keep(retained)
                    last_seen = self.active.last_seen
                    contexts, self.active.contexts = self.active.contexts, []
                self.active = _NvDCFTracks(
                    [candidate, *contexts], max_missed_frames=self.max_missed_frames,
                    serial=serial, last_seen=last_seen, retire=self._reaper.retire)
                self.last_metrics = metrics
                if recognition is not None:
                    self.last_recognition = recognition
                self.last_handoff = dict(ready_frame_id=ready_serial, frame_id=serial,
                                         skipped_frames=max(0, serial-ready_serial-1))
                result = self.active.proposals, 'sam3'
            except BaseException:
                self._reaper.retire(candidate)
                raise

        if recognized is not None:
            classes, recognition = recognized
            if self._apply_recognition(classes):
                self.last_recognition = recognition

        if result is not None:
            # Apply containment/unknown retirement once after merging, then count
            # misses only for committed detector results, never camera frames.
            self._filter(detected_ids, detected_at)
            result = self.active.proposals, result[1]

        with self.lock:
            # An empty committed detection waits for the configured interval;
            # visual tracking loss requests an earlier refresh.
            if changed and ready is None and lost:
                self.force_detection = True
            # Native inactive tracks may require a new detection to reactivate.
            # Retry within the grace instead of waiting the full normal interval.
            refresh_interval = interval
            if self.active is not None and self.active.hidden:
                refresh_interval = min(interval, max(1, self.max_missed_frames // 3))
            due = self.force_detection or self.frames_since_submit >= refresh_interval
            if due and not self.busy:
                self.busy = True
                self.recognition_error = None
                self.geometry_ready.clear()
                self.force_detection = False
                self.frames_since_submit = 0
                lease = frame.retain() if frame is not None else None
                self.thread_version = version
                self.thread = threading.Thread(
                    target=self._job,
                    args=(version, serial, self.latest[1], lease, detect, recognize, frame_key),
                    name='sam3-nvdcf-replay', daemon=True)
                self.thread.start()
            initial = self.thread if wait_for_initial_result and self.active is None and self.busy else None
        if initial is not None:
            if self.defer_recognition and self.thread_version == version:
                self.geometry_ready.wait()
            else:
                initial.join()
            return self._update(pixels, frame_key, context, interval, detect,
                                frame=frame, recognize=recognize, wait_for_initial_result=True)
        return result

    def close(self):
        with self._update_lock:
            with self.lock:
                self.closed = True
                self.version += 1
                thread = self.thread
            if thread is not None:
                thread.join()
            if self.ready is not None:
                self._reaper.retire(self.ready[0])
                self.ready = None
            if self.active is not None:
                self.active.close()
                self.active = None
            self._reaper.close()
