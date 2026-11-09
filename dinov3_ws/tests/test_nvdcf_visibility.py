"""Native confidence gates visibility without waiting for detector results."""
import sys

import numpy as np
import pytest

from dinov3_cases.tracking import nvdcf
from dinov3_cases.tracking.async_nvdcf import _NvDCFTracks


def test_low_confidence_and_missing_output_hide_then_recover_same_id(tmp_path, monkeypatch):
    bridge = tmp_path / 'bridge'
    bridge.write_text(f'#!{sys.executable}\n' + '''
import struct
import sys

def read(size):
    data = bytearray()
    while len(data) < size:
        chunk = sys.stdin.buffer.read(size - len(data))
        if not chunk:
            raise SystemExit
        data.extend(chunk)
    return data

sys.stdout.buffer.write(struct.pack('<I', 0x4e564443))
sys.stdout.buffer.flush()
for confidence in (0.9, 0.1, float('nan'), None, 0.8, 0.2):
    serial, detected, count = struct.unpack('<III', read(12))
    read(32 * 32 * 4 + count * 20)
    sys.stdout.buffer.write(struct.pack('<I', int(confidence is not None)))
    if confidence is not None:
        sys.stdout.buffer.write(struct.pack('<Q5fi', 7, 2, 3, 10, 12,
                                             confidence, 0 if count else -1))
    sys.stdout.buffer.flush()
''')
    bridge.chmod(0o755)
    config = tmp_path / 'nvdcf.yaml'
    config.write_text('%YAML:1.0\nTargetManagement:\n  minTrackerConfidence: 0.2\n')
    monkeypatch.setattr(nvdcf, 'BRIDGE', bridge)
    monkeypatch.setattr(nvdcf, 'check_nvdcf', lambda config: None)
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    tracker = nvdcf.NvDCF(config=config, width=32, height=32, transport='pipe', timeout=2)
    try:
        seed = dict(bbox=[2, 3, 12, 15], label='medicine', track_id=42)
        assert tracker.update(pixels, 1, [seed])[0]['track_id'] == 42
        tracks = _NvDCFTracks([tracker], serial=1, max_missed_frames=15)
        # No further SAM3 results: even a native record must disappear at low
        # confidence. Non-finite confidence and absent output also hide it.
        for frame in (2, 3, 4):
            tracks.update(pixels, frame)
            assert tracks.proposals == []
            assert tracks.hidden[0]['track_id'] == 42
            assert tracks.last_seen[42] == 1
        for frame in (5, 6):
            tracks.update(pixels, frame)
            assert tracks.proposals[0]['track_id'] == 42
            assert tracks.proposals[0]['label'] == 'medicine'
            assert tracks.hidden == []
    finally:
        tracker.close()


@pytest.mark.parametrize('threshold', [-0.1, 1.1, '.Nan'])
def test_invalid_visibility_threshold_is_rejected(tmp_path, threshold):
    config = tmp_path / 'nvdcf.yaml'
    config.write_text(f'%YAML:1.0\nTargetManagement:\n  minTrackerConfidence: {threshold}\n')
    with pytest.raises(ValueError, match='minTrackerConfidence'):
        nvdcf._min_tracker_confidence(config)
