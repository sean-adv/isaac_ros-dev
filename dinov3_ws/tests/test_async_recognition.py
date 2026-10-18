"""3D geometry continues while classification is blocked; stale labels never leak."""
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from dinov3_cases import async_recognition as module
from dinov3_cases.depth_bbox import attach_bboxes3d, bboxes3d_message
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header


@pytest.fixture
def worker(monkeypatch, tmp_path):
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    gallery = Mock(labels=['medicine'])
    gallery.refresh.return_value = False
    recognizer = Mock(encoded=True, diagnostic_detections=[])
    recognizer.classifications = [dict(label='medicine', score=.9)]

    def recognize(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return [], [], True

    recognizer.update.side_effect = recognize
    monkeypatch.setattr(module, 'ObjectGallery', lambda: gallery)
    monkeypatch.setattr(module, 'TrackedObjectRecognizer', lambda: recognizer)
    monkeypatch.setattr(module, 'ObjectVisualizationSnapshot', lambda *args: 'snapshot')
    errors = []
    pipeline = module.AsyncRecognition(Mock(), tmp_path, 32, 16,
                                       wake=completed.set, on_error=errors.append)
    yield pipeline, entered, release, completed, errors, recognizer
    release.set()
    pipeline.close()


def update(worker, token='track', x=25, frame=1, count=1):
    return worker.update(np.zeros((120, 160, 3), dtype=np.uint8),
                         [dict(bbox=[x, 30, x+110, 90], detection_label='box')]*count,
                         token=token, threshold=.75, margin=.05, frame_id=frame)


def test_slow_dino_does_not_block_current_3d_geometry(worker):
    pipeline, entered, release, completed, errors, recognizer = worker
    result = update(pipeline)
    assert entered.wait(1)
    assert result[3] == 'pending'
    for frame in (2, 3, 4):
        detections, _, _, source, recognition_frame = update(pipeline, frame=frame)
        assert source == 'pending' and recognition_frame is None
        depth = np.full((120, 160), 2.)
        depth[40:80, 35:125] = 1. + frame * .1
        info = CameraInfo(width=160, height=120,
                          k=[200., 0., 80., 0., 200., 60., 0., 0., 1.],
                          distortion_model='plumb_bob', d=[0.]*5)
        attach_bboxes3d(detections, depth, info)
        message = bboxes3d_message(detections, Header(frame_id=str(frame)))
        assert len(message.detections) == 1
        assert message.detections[0].bbox.center.position.z == pytest.approx(1. + frame*.1)
        assert message.header.frame_id == str(frame)
    release.set()
    assert completed.wait(1)
    detections, snapshot, changed, source, recognition_frame = update(pipeline, x=27, frame=5)
    assert detections[0]['label'] == 'medicine'
    assert detections[0]['bbox'][0] == 27
    assert recognition_frame == 1 and snapshot == 'snapshot'
    assert changed and source == 'dino'
    assert recognizer.update.call_count == 1 and not errors


def test_old_set_results_are_discarded_and_queue_keeps_only_latest(worker):
    pipeline, entered, release, completed, errors, _ = worker
    update(pipeline, token='old')
    assert entered.wait(1)
    update(pipeline, token='middle')
    detections, snapshot, changed, source, _ = update(pipeline, token='new', count=0)
    assert detections == [] and snapshot is None and changed
    assert pipeline.pending[3] == 'new'
    # Prevent a mocked recognizer returning a wrong number of labels for the empty job.
    with pipeline.condition:
        pipeline.pending = None
    release.set()
    assert completed.wait(1)
    assert pipeline.result is None
    assert not errors


def test_gallery_failure_preserves_geometry_and_bounds_retry(worker):
    pipeline, entered, release, completed, errors, recognizer = worker
    recognizer.update.side_effect = RuntimeError('encoder failed')
    update(pipeline)
    assert completed.wait(1)
    assert len(errors) == 1
    for n in range(5):
        detections, _, _, source, _ = update(pipeline, frame=n)
        assert detections[0]['bbox'] == [25, 30, 135, 90]
        assert source == 'pending'
    assert recognizer.update.call_count == 1
