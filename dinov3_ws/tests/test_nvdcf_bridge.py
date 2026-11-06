"""Bridge failures must expose native diagnostics and release IPC resources."""
import sys

import numpy as np
import pytest

from dinov3_cases import nvdcf


def fake_bridge(tmp_path, monkeypatch, body):
    executable = tmp_path / 'bridge'
    executable.write_text(f'#!{sys.executable}\nimport os, struct, sys\n{body}\n')
    executable.chmod(0o755)
    monkeypatch.setattr(nvdcf, 'BRIDGE', executable)
    monkeypatch.setattr(nvdcf, 'check_nvdcf', lambda config: None)


def test_startup_failure_includes_native_stderr_and_releases_resources(tmp_path, monkeypatch):
    fake_bridge(tmp_path, monkeypatch,
                "sys.stderr.write('NvDCF bridge: usage: nvdcf_bridge LIB CONFIG WIDTH HEIGHT\\n')\n"
                "sys.exit(1)")
    tracker = nvdcf.NvDCF.__new__(nvdcf.NvDCF)
    with pytest.raises(RuntimeError, match=r'exited \(1\).*usage: nvdcf_bridge'):
        tracker.__init__(timeout=2)
    assert tracker.process is tracker.input_shm is tracker.stderr_log is None
    tracker.close()


def test_frame_failure_drains_large_stderr_without_blocking(tmp_path, monkeypatch):
    fake_bridge(tmp_path, monkeypatch,
                "sys.stdout.buffer.write(struct.pack('<I', 0x4e564443))\n"
                "sys.stdout.buffer.flush()\n"
                "sys.stdin.buffer.read(12)\n"
                "sys.stderr.write('x' * 200000 + '\\nNvDCF bridge: RGB upload failed\\n')\n"
                "sys.exit(1)")
    tracker = nvdcf.NvDCF(timeout=2, transport='pipe', width=32, height=32)
    with pytest.raises(RuntimeError, match=r'(?s)exited \(1\).*RGB upload failed') as error:
        tracker.update(np.zeros((32, 32, 3), dtype=np.uint8), 0, [])
    assert len(str(error.value)) < 8300
    assert tracker.process is tracker.stderr_log is None
    tracker.close()
