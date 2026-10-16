"""Bounded RGB frame storage shared by inference and asynchronous rendering.

Only selected inference frames are copied; camera callbacks keep the latest ROS
message. A slot remains immutable until every lease has been released.
"""
from multiprocessing import shared_memory
import threading
import time

import numpy as np

MAX_IMAGE_BYTES = 128 << 20


class SharedFrame:
    def __init__(self, pool, slot, shape, frame_id, timestamp, copy_ms):
        self.pool, self.slot, self.shape = pool, slot, tuple(shape)
        self.frame_id, self.timestamp, self.copy_ms = frame_id, timestamp, copy_ms
        self.released = False

    @property
    def shm(self):
        return self.pool.slots[self.slot][0]

    @property
    def pixels(self):
        if self.released:
            raise RuntimeError("Frame lease has been released")
        view = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
        view.flags.writeable = False
        return view

    def retain(self):
        with self.pool.lock:
            if self.released:
                raise RuntimeError("Frame lease has been released")
            self.pool.slots[self.slot][1] += 1
            return SharedFrame(self.pool, self.slot, self.shape, self.frame_id,
                               self.timestamp, self.copy_ms)

    def release(self):
        with self.pool.lock:
            if not self.released:
                self.released = True
                self.pool.slots[self.slot][1] -= 1
                self.pool._dispose_if_closed(self.slot)


class SharedFramePool:
    def __init__(self, slots=3):
        if type(slots) is not int or not 1 <= slots <= 3:
            raise ValueError("Frame pool requires one to three slots")
        self.lock = threading.Lock()
        self.slots = [[None, 0] for _ in range(slots)]
        self.closed = False

    def acquire(self, pixels, *, frame_id=0, timestamp=None):
        pixels = np.asarray(pixels)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape) <= 0:
            raise ValueError("Expected a nonempty uint8 HWC RGB frame")
        if pixels.nbytes > MAX_IMAGE_BYTES:
            raise ValueError("RGB frame exceeds 128 MiB shared-memory limit")
        with self.lock:
            if self.closed:
                raise RuntimeError("Frame pool is closed")
            for index, slot in enumerate(self.slots):
                if slot[1]:
                    continue
                if slot[0] is None or slot[0].size < pixels.nbytes:
                    replacement = shared_memory.SharedMemory(create=True, size=pixels.nbytes)
                    if slot[0] is not None:
                        slot[0].close()
                        slot[0].unlink()
                    slot[0] = replacement
                start = time.monotonic_ns()
                np.ndarray(pixels.shape, dtype=np.uint8, buffer=slot[0].buf)[:] = pixels
                elapsed = (time.monotonic_ns() - start) / 1e6
                slot[1] = 1
                return SharedFrame(self, index, pixels.shape, frame_id, timestamp, elapsed)
        return None  # Never wait for old output or overwrite a live frame.

    def _dispose_if_closed(self, index):
        slot = self.slots[index]
        if self.closed and slot[1] == 0 and slot[0] is not None:
            slot[0].close()
            slot[0].unlink()
            slot[0] = None

    def close(self):
        with self.lock:
            self.closed = True
            for index in range(len(self.slots)):
                self._dispose_if_closed(index)
