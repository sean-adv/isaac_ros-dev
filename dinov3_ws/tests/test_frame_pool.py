import numpy as np
import pytest
from multiprocessing import shared_memory
from dinov3_cases.frame_pool import SharedFramePool


def test_leases_prevent_overwrite_and_pool_is_bounded():
    pool = SharedFramePool()
    frames = []
    try:
        for i in range(3):
            frames.append(pool.acquire(np.full((4, 6, 3), i, np.uint8), frame_id=i))
        assert pool.acquire(np.zeros((4, 6, 3), np.uint8)) is None
        lease = frames[0].retain()
        frames[0].release()
        assert pool.acquire(np.zeros((4, 6, 3), np.uint8)) is None
        np.testing.assert_array_equal(lease.pixels, 0)
        with pytest.raises(ValueError):
            lease.pixels[0, 0] = 99
        lease.release()
        replacement = pool.acquire(np.full((4, 6, 3), 99, np.uint8))
        assert replacement.slot == 0
        np.testing.assert_array_equal(frames[1].pixels, 1)
        replacement.release()
    finally:
        for frame in frames:
            frame.release()
        pool.close()


def test_shutdown_defers_cleanup_until_last_reader_and_resize_handles_stride():
    pool = SharedFramePool()
    source = np.arange(12 * 8 * 3, dtype=np.uint8).reshape(12, 8, 3)[::2, ::2]
    frame = pool.acquire(source)
    name = frame.shm.name
    pool.close()
    np.testing.assert_array_equal(frame.pixels, source)
    frame.release()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)
