"""Opt-in cooperative, cross-process admission. No CUDA imports in the broker.

One connection is one lease. An active disconnect poisons the broker: a dead
socket is NOT evidence that CUDA kernels completed. Restart only after stopping
all participating GPU clients. Wait timeouts fail closed, never bypass admission.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import select
import socket
import socketserver
import threading
import time

_local = threading.local()


class Admission:
    def __init__(self, max_normal_wait=0.5):
        self.condition = threading.Condition()
        self.queue = []
        self.active = None
        self.poisoned = False
        self.max_normal_wait = max_normal_wait

    def next(self):
        # FIFO within each class. Once aged, a normal waiter goes before highs.
        now = time.monotonic()
        aged = [q for q in self.queue if q[1] == b'N' and now - q[2] >= self.max_normal_wait]
        return (aged or [q for q in self.queue if q[1] == b'H'] or self.queue)[0]


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        state = self.server.admission
        request = self.request
        granted = released = False
        item = None
        try:
            request.settimeout(5)
            priority = request.recv(1)
            if priority not in (b'H', b'N'):
                return
            item = (object(), priority, time.monotonic())
            with state.condition:
                state.queue.append(item)
                while True:
                    if state.poisoned:
                        request.sendall(b'X')
                        return
                    if state.active is None and state.next() is item:
                        state.queue.remove(item)
                        state.active = item
                        granted = True
                        request.sendall(b'G')
                        break
                    if select.select([request], [], [], 0)[0]:
                        return  # queued client cancelled/disconnected; never granted
                    state.condition.wait(0.02)
            # No lease expiry: expiry cannot cancel kernels already on the GPU.
            request.settimeout(None)
            if request.recv(1) == b'R':
                released = True
                request.sendall(b'R')
        except OSError:
            pass
        finally:
            with state.condition:
                if item in state.queue:
                    state.queue.remove(item)
                if granted:
                    if not released:
                        state.poisoned = True
                    state.active = None
                state.condition.notify_all()


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path, max_normal_wait=0.5):
        self.admission = Admission(max_normal_wait)
        super().__init__(str(path), Handler)


class Lease:
    def __init__(self, model):
        self.model = model
        self.sock = None
        self.wait_ms = 0.0

    def acquire(self):
        path = os.environ.get('DINOV3_GPU_SCHEDULER', '')
        if not path:
            return
        timeout = float(os.environ.get('DINOV3_GPU_TIMEOUT', '30'))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('DINOV3_GPU_TIMEOUT must be finite and positive')
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        start = time.perf_counter()
        try:
            self.sock.connect(path)
            self.sock.sendall(b'H' if self.model == 'sam3' else b'N')
            if self.sock.recv(1) != b'G':
                raise RuntimeError('GPU scheduler unavailable/poisoned; stop clients before restarting broker')
        except BaseException:
            self.sock.close()
            self.sock = None
            raise
        self.wait_ms = (time.perf_counter() - start) * 1000

    def close(self, completed):
        if self.sock is None:
            return
        try:
            if completed:
                self.sock.sendall(b'R')
                if self.sock.recv(1) != b'R':
                    raise RuntimeError('GPU scheduler release was not acknowledged')
        finally:
            self.sock.close()


class Measurement:
    def __init__(self):
        self.values = {}
        self.events = {}

    @contextmanager
    def span(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] = (time.perf_counter() - start) * 1000

    @contextmanager
    def cuda_span(self, name):
        import torch
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events[name] = (start, end)


def instrumented():
    return bool(os.environ.get('DINOV3_GPU_METRICS'))


@contextmanager
def gpu_region(model, device='cuda:0', measurement=None):
    """Keep lease through ALL process/device streams, including exception paths.

    Reentrant per thread, so encoder -> TRTBackbone does not deadlock. The entire
    participating device workload must use this API; CUDA work outside it remains
    uncontrolled. Events measure current-stream elapsed time, not kernel busy time.
    """
    if str(device) == 'cpu':
        yield measurement
        return
    if getattr(_local, 'depth', 0):
        _local.depth += 1
        try:
            yield measurement
        finally:
            _local.depth -= 1
        return
    enabled = bool(os.environ.get('DINOV3_GPU_SCHEDULER'))
    measured = instrumented()
    if not enabled and not measured:
        yield measurement
        return
    import torch
    lease = Lease(model)
    lease.acquire()
    _local.depth = 1
    completed = False
    start = end = None
    try:
        with torch.cuda.device(device):
            if measurement is not None:
                measurement.values['scheduler_wait_ms'] = lease.wait_ms
                measurement.values['gpu_start_ns'] = time.monotonic_ns()
            if measured:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            try:
                yield measurement
            finally:
                sync_start = time.perf_counter()
                if end is not None:
                    end.record()
                # Not just Python forward return or a single unverified stream.
                torch.cuda.synchronize(device)
                completed = True
                if measurement is not None:
                    measurement.values['completion_wait_ms'] = (time.perf_counter() - sync_start) * 1000
                    measurement.values['gpu_completed_ns'] = time.monotonic_ns()
                    if start is not None:
                        measurement.values['gpu_execution_ms'] = start.elapsed_time(end)
                        for name, (first, last) in measurement.events.items():
                            measurement.values[name] = first.elapsed_time(last)
    finally:
        _local.depth = 0
        lease.close(completed)


def write_metrics(model, values):
    directory = os.environ.get('DINOV3_GPU_METRICS')
    if not directory:
        return
    # Per-process files; one append write prevents interleaving between threads.
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    row = dict(model=model, pid=os.getpid(), tid=threading.get_ident(), **values)
    raw = (json.dumps(row, allow_nan=False) + '\n').encode()
    fd = os.open(path / f'{os.getpid()}.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', required=True, type=Path)
    parser.add_argument('--max-normal-wait', type=float, default=0.5,
                        help='aging threshold, not a deadline or kernel preemption')
    args = parser.parse_args()
    if not math.isfinite(args.max_normal_wait) or args.max_normal_wait <= 0:
        parser.error('max-normal-wait must be finite and positive')
    # Never unlink an existing broker socket (could create two admission domains).
    # Caller must use a private 0700 directory; socket is restricted to its owner.
    with Server(args.socket, args.max_normal_wait) as server:
        os.chmod(args.socket, 0o600)
        print(f'GPU scheduler ready: {args.socket}', flush=True)
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            pass
        finally:
            args.socket.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
