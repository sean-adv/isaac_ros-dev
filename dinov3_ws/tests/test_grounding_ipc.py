"""Exercise real subprocess IPC, SHM ownership, sequence checks and cancellation."""
from multiprocessing import shared_memory
from pathlib import Path
import os
import sys
import threading
import time

from PIL import Image
import pytest
from dinov3_cases.grounding_ipc import SharedMemoryGroundingDetector


@pytest.fixture
def proxy(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    opened = []
    def create(**options):
        item = SharedMemoryGroundingDetector('fixture', 'cpu', None, 'trt', 'unused',
                    python=sys.executable, worker_module='ipc_fake_worker', **options)
        opened.append(item)
        return item
    yield create
    for item in opened:
        item.close()


def detect(proxy, color='red', size=(40, 20), prompt='box.'):
    return proxy.detect(Image.new('RGB', size, color), prompt, .3, .25, .5, 20)


def test_rgb_frame_identity_resize_and_shm_cleanup(proxy):
    p = proxy()
    assert p.worker_info['pid'] != os.getpid()
    assert detect(p)[0]['color'] == [255, 0, 0]
    first_id, first_name = p.last_metrics['frame_id'], p.shm.name
    assert detect(p, 'blue')[0]['color'] == [0, 0, 255]
    assert p.shm.name == first_name
    assert p.last_metrics['frame_id'] > first_id
    assert detect(p, 'green', (200, 100))[0]['bbox'] == [0, 0, 200, 100]
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=first_name)
    last_name = p.shm.name
    p.close()
    assert p.process.poll() is not None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=last_name)


def test_prompt_error_keeps_worker_available(proxy):
    p = proxy()
    with pytest.raises(ValueError, match='invalid test prompt'):
        p.validate_prompt('invalid')
    assert detect(p)


def test_worker_crash_fails_and_releases_shm(proxy):
    p = proxy()
    detect(p)
    name = p.shm.name
    p.process.kill(); p.process.wait()
    with pytest.raises(RuntimeError, match='IPC failed'):
        detect(p)
    assert p.closed
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def test_timeout_stops_worker_without_stale_result(proxy):
    p = proxy(timeout=.1)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='IPC failed'):
        detect(p, prompt='hang')
    assert time.monotonic() - start < 3
    assert p.closed and p.process.poll() is not None


def test_wrong_frame_id_is_rejected(proxy, monkeypatch):
    monkeypatch.setenv('IPC_FAKE_MISMATCH', '1')
    p = proxy()
    with pytest.raises(RuntimeError, match='frame ID mismatch'):
        detect(p)
    assert p.closed


def test_close_interrupts_pending_request(proxy):
    p = proxy(timeout=60)
    errors = []
    def pending():
        try:
            detect(p, prompt='hang')
        except RuntimeError as error:
            errors.append(error)
    thread = threading.Thread(target=pending)
    thread.start()
    deadline = time.monotonic() + 3
    while p.sequence == 0 and time.monotonic() < deadline:
        time.sleep(.01)
    assert p.sequence > 0
    name = p.shm.name
    p.close()
    thread.join(3)
    assert not thread.is_alive() and errors
    assert p.process.poll() is not None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)
