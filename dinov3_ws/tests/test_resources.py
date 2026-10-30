"""Resource accounting contracts using a synthetic Linux proc filesystem."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from dinov3_cases.resources import ResourceMonitor, ResourceSampler


def process(proc, pid, parent, command, ticks=10, start=1, pages=2):
    folder = proc / str(pid)
    folder.mkdir(exist_ok=True)
    fields = ['0'] * 22
    fields[0], fields[1] = 'S', str(parent)
    fields[11], fields[12], fields[19], fields[21] = str(ticks), '0', str(start), str(pages)
    (folder / 'stat').write_text(f'{pid} (name with ) spaces) ' + ' '.join(fields))
    (folder / 'cmdline').write_bytes(command.encode().replace(b' ', b'\0'))


def cpu_stat(proc, aggregate, count):
    (proc / 'stat').write_text(aggregate + '\n' + ''.join(
        f'cpu{index} 0 0 0 0 0 0 0 0\n' for index in range(count)))


@pytest.fixture
def proc(tmp_path):
    cpu_stat(tmp_path, 'cpu 100 0 100 800 0 0 0 0 0 0', 14)
    (tmp_path / 'meminfo').write_text('MemTotal: 10000 kB\nMemAvailable: 4000 kB\n')
    return tmp_path


def test_accounting_and_pid_reuse(proc, monkeypatch):
    sampler = ResourceSampler(proc, root_pid=10)
    monkeypatch.setattr(sampler, '_gpu', lambda: None)
    clock = iter([1., 2., 3., 4.])
    monkeypatch.setattr('dinov3_cases.resources.time.monotonic', lambda: next(clock))
    process(proc, 10, 1, 'main')
    process(proc, 20, 10, 'python -m dinov3_cases.grounding_ipc')
    process(proc, 30, 10, '/tmp/nvdcf_bridge')
    process(proc, 31, 10, '/usr/local/bin/dinov3-nvdcf-bridge')
    process(proc, 90, 1, '/tmp/nvdcf_bridge')  # unrelated pipeline
    first = sampler.sample()
    assert first['processes']['nvdcf']['pids'] == [30, 31]
    assert first['processes']['nvdcf']['rss_bytes'] == 4 * sampler.page
    assert first['processes']['sam3']['cpu_percent'] is None
    assert first['system']['ram_used_bytes'] == 6000 * 1024
    for pid, parent, command in [(10, 1, 'main'), (20, 10, 'python -m dinov3_cases.grounding_ipc'),
                                 (30, 10, 'nvdcf_bridge'), (31, 10, 'nvdcf_bridge')]:
        process(proc, pid, parent, command, ticks=10 + sampler.ticks)
    cpu_stat(proc, 'cpu 200 0 100 900 0 0 0 0 0 0', 14)
    second = sampler.sample()
    assert second['system']['cpu_count'] == 14
    assert second['processes']['nvdcf']['cpu_percent'] == pytest.approx(200 / 14)
    assert second['processes']['nvdcf']['cpu_core_percent'] == pytest.approx(200)
    assert second['system']['cpu_percent'] == pytest.approx(50)
    process(proc, 30, 10, 'nvdcf_bridge', ticks=5, start=2)
    third = sampler.sample()
    assert third['processes']['nvdcf']['cpu_percent'] is None
    assert not third['processes']['nvdcf']['cpu_complete']
    for path in (proc / '31').iterdir():
        path.unlink()
    (proc / '31').rmdir()
    assert sampler.sample()['processes']['nvdcf']['pids'] == [30]


def test_monitor_csv_publish_and_shutdown(tmp_path, monkeypatch):
    monkeypatch.setattr('dinov3_cases.resources.shutil.which', lambda _: None)
    published = []
    monitor = ResourceMonitor(.1, str(tmp_path), publish=published.append)
    monitor.start()
    try:
        deadline = time.monotonic() + 3
        while len(published) < 2 and time.monotonic() < deadline:
            time.sleep(.02)
        assert len(published) >= 2
        payload = json.loads(published[-1])
        assert os.getpid() in payload['processes']['dinov3_main']['pids']
        assert payload['system']['ram_total_bytes'] > 0
        snapshot = monitor.snapshot()
        snapshot['sample']['processes'].clear()
        assert monitor.snapshot()['sample']['processes']
    finally:
        monitor.close()
    assert not monitor.thread.is_alive()
    rows = list(csv.DictReader(Path(monitor.csv_path).open()))
    assert {row['component'] for row in rows} == {'system', 'dinov3_main', 'sam3', 'nvdcf'}
    assert float(rows[-4]['timestamp']) == json.loads(published[-1])['timestamp']
    payload = json.loads(published[-1])
    count = payload['system']['cpu_count']
    assert count >= 1
    assert all(int(row['cpu_count']) == count for row in rows[-4:])
    main = next(row for row in rows[-4:] if row['component'] == 'dinov3_main')
    raw = payload['processes']['dinov3_main']['cpu_core_percent']
    assert float(main['cpu_core_percent']) == pytest.approx(raw)
    assert float(main['cpu_percent']) == pytest.approx(raw / count)


def test_csv_failure_does_not_disable_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr('dinov3_cases.resources.shutil.which', lambda _: None)
    blocker = tmp_path / 'file'
    blocker.write_text('not a directory')
    monitor = ResourceMonitor(.1, str(blocker / 'logs'))
    try:
        assert monitor.snapshot()['csv_error']
        monitor.start()
        deadline = time.monotonic() + 3
        while monitor.snapshot()['sample'] is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert monitor.snapshot()['sample'] is not None
    finally:
        monitor.close()


def test_tegrastats_data_expires_and_process_is_closed(monkeypatch):
    monkeypatch.setattr('dinov3_cases.resources.shutil.which', lambda _: None)
    monitor = ResourceMonitor(.1)
    monitor.tegrastats = subprocess.Popen(
        [sys.executable, '-u', '-c', 'import time; print("RAM 1/2MB GR3D_FREQ 42%@[100,100]"); time.sleep(30)'],
        stdout=subprocess.PIPE,
    )
    os.set_blocking(monitor.tegrastats.stdout.fileno(), False)
    try:
        sample = {'system': {'gpu_percent': None, 'gpu_source': None}}
        deadline = time.monotonic() + 3
        while monitor.gpu_sample is None and time.monotonic() < deadline:
            monitor._gpu(sample)
            time.sleep(.02)
        assert sample['system']['gpu_percent'] == 42
        assert sample['system']['gpu_source'] == 'tegrastats'
        monitor.gpu_sample = (time.monotonic() - 10, 42)
        sample = {'system': {'gpu_percent': None, 'gpu_source': None}}
        monitor._gpu(sample)
        assert sample['system']['gpu_percent'] is None
        assert sample['system']['gpu_unavailable_reason']
    finally:
        monitor.close()
    assert monitor.tegrastats.poll() is not None


@pytest.mark.parametrize('interval', [0, -.1, .01, float('nan'), float('inf'), True])
def test_invalid_interval(interval):
    with pytest.raises(ValueError):
        ResourceMonitor(interval)


def test_online_core_count_changes(proc, monkeypatch):
    sampler = ResourceSampler(proc, root_pid=10)
    monkeypatch.setattr(sampler, '_gpu', lambda: None)
    clock = iter([1., 2., 3.])
    monkeypatch.setattr('dinov3_cases.resources.time.monotonic', lambda: next(clock))
    process(proc, 10, 1, 'main', ticks=0)
    assert sampler.sample()['system']['cpu_count'] == 14
    process(proc, 10, 1, 'main', ticks=sampler.ticks)
    cpu_stat(proc, 'cpu 200 0 100 900 0 0 0 0', 7)
    second = sampler.sample()
    assert second['system']['cpu_count'] == 7
    assert second['processes']['dinov3_main']['cpu_percent'] == pytest.approx(100 / 7)
    process(proc, 10, 1, 'main', ticks=2 * sampler.ticks)
    cpu_stat(proc, 'cpu 300 0 100 1000 0 0 0 0', 1)
    third = sampler.sample()
    assert third['system']['cpu_count'] == 1
    assert third['processes']['dinov3_main']['cpu_percent'] == pytest.approx(100)
