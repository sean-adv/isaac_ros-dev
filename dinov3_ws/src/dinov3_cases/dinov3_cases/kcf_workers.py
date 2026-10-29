"""Bounded actor workers. Public update results expose handles, not trackers."""
from concurrent.futures import Future, wait
from dataclasses import dataclass
import itertools
import queue
import threading


class StaleTracker(RuntimeError):
    """A command refers to a retired generation, phase, or older frame."""


@dataclass(frozen=True)
class TrackerHandle:
    token: int
    generation: int
    worker: int
    track_id: object = None


@dataclass
class _State:
    native: object
    generation: int
    frame_id: int
    phase: str
    root_id: int


class PersistentKCFWorkers:
    """One bounded mailbox and private tracker dictionary per long-lived thread.

    Mailbox items are batches, not frames per object. Synchronous producers allow
    at most one outstanding batch each (foreground and replay). Full mailboxes
    apply backpressure; lifecycle commands are never dropped. Replay's latest
    image slot, outside this class, explicitly coalesces incoming camera frames.
    """
    def __init__(self, workers, queue_capacity=2):
        if type(workers) is not int or workers < 1:
            raise ValueError('workers must be a positive integer')
        if type(queue_capacity) is not int or queue_capacity < 1:
            raise ValueError('queue_capacity must be a positive integer')
        self.queues = [queue.Queue(maxsize=queue_capacity) for _ in range(workers)]
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._closing = False
        self._roots = set()
        self._provisional = 0
        self._stats = [dict(commands=0, stale=0, max_queue=0, owned=0) for _ in self.queues]
        self.threads = [threading.Thread(target=self._run, args=(i,), name=f'kcf-owner-{i}')
                        for i in range(workers)]
        for thread in self.threads:
            thread.start()

    @staticmethod
    def _root(native):
        while hasattr(native, 'tracker'):
            native = native.tracker
        return id(native)

    def _claim(self, root):
        with self._lock:
            if root in self._roots:
                raise ValueError('KCF objects must have distinct tracker instances')
            self._roots.add(root)

    def _run(self, index):
        states = {}  # Native methods execute only here; export/adopt transfers ownership.
        while True:
            item = self.queues[index].get()
            if item is None:
                for state in states.values():
                    with self._lock:
                        self._roots.discard(state.root_id)
                states.clear()
                self._stats[index]["owned"] = 0
                return
            commands, future = item
            results = []
            error = None
            for action, handle, frame, payload in commands:
                try:
                    state = states.get(handle.token)
                    if action == 'create':
                        create, initialize, phase = payload
                        native = create()
                        root = self._root(native)
                        self._claim(root)
                        try:
                            ok = initialize(native)
                        except BaseException:
                            with self._lock:
                                self._roots.discard(root)
                            raise
                        states[handle.token] = _State(native, handle.generation, frame, phase, root)
                        result = ok
                    elif action == 'adopt':
                        # Source removed its entry before acknowledging export.
                        state = payload
                        state.generation, state.phase = handle.generation, 'active'
                        states[handle.token] = state
                        result = None
                    else:
                        if state is None or state.generation != handle.generation:
                            raise StaleTracker('retired tracker generation')
                        if action == 'update':
                            if state.phase == 'ready' or frame <= state.frame_id:
                                raise StaleTracker('frozen tracker or non-increasing frame_id')
                            value = payload(state.native)
                            state.frame_id = frame
                            if value is not None:
                                native, roi, bbox, refresh = value
                                root = self._root(native)
                                if root != state.root_id:
                                    self._claim(root)
                                    with self._lock:
                                        self._roots.discard(state.root_id)
                                    state.root_id = root
                                state.native = native
                                metrics = (dict(kcf_lab_similarity=native.lab_similarity,
                                                kcf_lab_changed_frames=native.lab_changed_frames)
                                           if hasattr(native, 'lab_similarity') else {})
                                value = (handle, roi, bbox, refresh, metrics)
                            result = (handle.generation, frame, value)
                        elif action == 'phase':
                            state.phase = payload
                            result = None
                        elif action == 'export':
                            if state.phase != 'ready' or state.frame_id != frame:
                                raise StaleTracker('handoff requires READY at the common frame')
                            result = states.pop(handle.token)
                        elif action == 'release':
                            states.pop(handle.token)
                            with self._lock:
                                self._roots.discard(state.root_id)
                            result = None
                        elif action == 'inspect':
                            result = payload(state.native)
                        else:
                            raise ValueError(action)
                    results.append(result)
                except BaseException as exc:
                    if isinstance(exc, StaleTracker):
                        self._stats[index]['stale'] += 1
                    if error is None:
                        error = exc
                    results.append(None)
                self._stats[index]['commands'] += 1
            self._stats[index]['owned'] = len(states)
            # Execute/drain the whole batch even if one object raised.
            if error is None:
                future.set_result(results)
            else:
                future.set_exception(error)

    def _dispatch(self, commands):
        grouped = {}
        for position, command in enumerate(commands):
            grouped.setdefault(command[1].worker, []).append((position, command))
        futures = []
        try:
            for index, group in grouped.items():
                future = Future()
                # Lock only the enqueue operation, never native work or waiting.
                with self._lock:
                    if self._closing:
                        raise RuntimeError('KCF workers are closed')
                self.queues[index].put(([command for _, command in group], future))
                self._stats[index]['max_queue'] = max(self._stats[index]['max_queue'],
                                                       self.queues[index].qsize())
                futures.append((group, future))
        finally:
            if futures:
                wait([future for _, future in futures])
        output = [None]*len(commands)
        for group, future in futures:
            for (position, _), result in zip(group, future.result()):
                output[position] = result
        return output

    def create(self, track_id, frame_id, create, initialize, phase):
        with self._lock:
            token = next(self._ids)
            if track_id is None:
                index = self._provisional % len(self.queues)
                self._provisional += 1
            else:
                index = track_id % len(self.queues)
        handle = TrackerHandle(token, token, index, track_id)
        ok = self._dispatch([('create', handle, frame_id, (create, initialize, phase))])[0]
        return handle, ok

    def update(self, handles, frame_id, operation):
        replies = self._dispatch([('update', h, frame_id, operation(i)) for i, h in enumerate(handles)])
        output = []
        for handle, (generation, frame, result) in zip(handles, replies):
            if (generation, frame) != (handle.generation, frame_id):
                raise StaleTracker('outdated worker result')
            output.append(result)
        return output

    def phase(self, handles, phase):
        self._dispatch([('phase', h, None, phase) for h in handles])

    def release(self, handles):
        if handles:
            self._dispatch([('release', h, None, None) for h in handles])

    def handoff(self, handle, track_id, frame_id):
        state = self._dispatch([('export', handle, frame_id, None)])[0]
        with self._lock:
            generation = next(self._ids)
        target = TrackerHandle(generation, generation, track_id % len(self.queues), track_id)
        self._dispatch([('adopt', target, frame_id, state)])
        return target

    def inspect(self, handle, read):
        """Test/diagnostic read executed by the owner; return copied data only."""
        return self._dispatch([('inspect', handle, None, read)])[0]

    def metrics(self):
        return [dict(stats, worker=i, thread_id=self.threads[i].native_id,
                     queue_depth=self.queues[i].qsize(), queue_capacity=self.queues[i].maxsize)
                for i, stats in enumerate(self._stats)]

    def shutdown(self):
        # Caller has stopped/joined producers before closing mailboxes.
        with self._lock:
            if self._closing:
                return
            self._closing = True
        for mailbox in self.queues:
            mailbox.put(None)
        for thread in self.threads:
            thread.join()
