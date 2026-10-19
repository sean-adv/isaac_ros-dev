"""Independent frame cadence, bounded jobs and coherent cached 3D measurements."""
import threading

import pytest
from builtin_interfaces.msg import Time
from std_msgs.msg import Header

from dinov3_cases.async_bbox3d import AsyncBBox3D, validate_bbox3d_schedule
from dinov3_cases.depth_bbox import bboxes3d_message


def detection(x=0, label="medicine"):
    return dict(bbox=[x, 0, x+20, 10], label=label, score=.9)


def submit(worker, frame=1, items=None, context="camera", inputs=(1, "calibration"), **options):
    return worker.update([detection()] if items is None else items, inputs,
                         header=Header(frame_id="camera", stamp=Time(sec=frame)),
                         frame_key=(0, frame), context=context, **options)


@pytest.fixture
def worker():
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    calls, errors = [], []

    def measure(items, inputs):
        calls.append(([item['bbox'][:] for item in items], inputs))
        entered.set()
        assert release.wait(3)
        for item in items:
            item['bbox3d_status'] = 'ok' if inputs else 'missing_depth'
            if inputs:
                item['bbox3d'] = dict(position=[float(item['bbox'][0]), 0., float(inputs[0])],
                                      size=[.2, .1, .01],
                                      rotation=[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])

    pipeline = AsyncBBox3D(measure, wake=completed.set, on_error=errors.append)
    yield pipeline, entered, release, completed, calls, errors
    release.set()
    pipeline.close()


def test_five_unique_frame_cadence_and_nonblocking_full_measurement(worker):
    pipeline, entered, release, completed, calls, errors = worker
    first, _, _ = submit(pipeline)
    assert first[0]['bbox3d_status'] == 'pending_background_measurement'
    assert entered.wait(1)
    for frame in (1, 2, 2, 3, 4, 5):
        output, _, metrics = submit(pipeline, frame)
        assert 'bbox3d' not in output[0]
        assert metrics['interval'] == 5
    assert len(calls) == 1 and pipeline.pending is None
    release.set()
    assert completed.wait(1)
    assert pipeline.has_result()
    # Rendering refreshes the already queued frame without scheduling another job.
    rendered, info = pipeline.latest(first)
    assert info == 'calibration'
    assert rendered[0]['bbox3d']['position'][2] == 1.
    assert rendered[0]['bbox3d_source']['stamp']['sec'] == 1
    assert 'bbox3d' not in first[0]  # Never mutate a queued output snapshot.
    completed.clear()
    submit(pipeline, 6, inputs=(2, 'calibration'))
    assert completed.wait(1)
    latest, _, _ = submit(pipeline, 6)
    assert latest[0]['bbox3d']['position'][2] == 2.
    assert len(calls) == 2
    assert not errors


def test_busy_worker_replaces_pending_with_latest_due_frame(worker):
    pipeline, entered, release, completed, calls, _ = worker
    submit(pipeline, 1)
    assert entered.wait(1)
    for frame in range(2, 12):
        submit(pipeline, frame, inputs=(frame, 'calibration'))
    assert len(calls) == 1
    assert pipeline.pending[4]['stamp']['sec'] == 11
    latest_finished = threading.Event()
    pipeline.wake = lambda: latest_finished.set() if len(calls) == 2 else None
    release.set()
    assert latest_finished.wait(1)
    assert [call[1][0] for call in calls] == [1, 11]
    assert pipeline.pending is None


def test_reordered_objects_keep_their_measurements_and_lost_ids_are_dropped(worker):
    pipeline, entered, release, completed, _, _ = worker
    submit(pipeline, items=[detection(0), detection(80)])
    assert entered.wait(1)
    release.set()
    assert completed.wait(1)
    output, _, _ = submit(pipeline, 2, items=[detection(81), detection(1)])
    assert [item['bbox3d']['position'][0] for item in output] == [80., 0.]
    output, _, _ = submit(pipeline, 3, items=[detection(2)])
    assert output[0]['bbox3d']['position'][0] == 0.
    assert len(pipeline.cache) == 1
    submit(pipeline, 4, items=[])
    output, _, _ = submit(pipeline, 5)
    assert 'bbox3d' not in output[0]


def test_context_switch_discards_running_and_pending_results(worker):
    pipeline, entered, release, completed, _, _ = worker
    old, _, _ = submit(pipeline)
    assert entered.wait(1)
    submit(pipeline, 2, context='new', items=[])
    release.set()
    assert completed.wait(1)
    assert not pipeline.cache
    refreshed, _ = pipeline.latest(old)
    assert 'bbox3d' not in refreshed[0]


def test_failed_measurement_replaces_previous_success():
    completed = threading.Event()
    errors = []

    def fail(items, inputs):
        raise RuntimeError('depth conversion failed')

    pipeline = AsyncBBox3D(fail, wake=completed.set, on_error=errors.append)
    try:
        submit(pipeline)
        assert completed.wait(1)
        output, _, _ = submit(pipeline)
        assert output[0]['bbox3d_status'] == 'background_measurement_failed'
        assert 'bbox3d' not in output[0]
        assert len(errors) == 1
    finally:
        pipeline.close()


def test_age_expiry_uses_source_capture_time_and_ros_preserves_measurement_stamp(worker, monkeypatch):
    pipeline, entered, release, completed, _, _ = worker
    monkeypatch.setattr('dinov3_cases.async_bbox3d.time.monotonic', lambda: 10.)
    queued, _, _ = submit(pipeline, captured_at=9.)
    assert entered.wait(1)
    release.set()
    assert completed.wait(1)
    latest, _ = pipeline.latest(queued)
    assert latest[0]['bbox3d_age_seconds'] == 1.
    message = bboxes3d_message(latest, Header(frame_id='camera', stamp=Time(sec=99)))
    assert message.header.stamp.sec == 1
    assert message.detections[0].header.stamp.sec == 1
    assert message.detections[0].results[0].pose.pose == message.detections[0].bbox.center
    monkeypatch.setattr('dinov3_cases.async_bbox3d.time.monotonic', lambda: 11.)
    stale, _ = pipeline.latest(queued)
    assert stale[0]['bbox3d_status'] == 'stale_background_measurement'
    assert 'bbox3d' not in stale[0]


@pytest.mark.parametrize('interval', [True, 0, -1, 1.5])
def test_invalid_interval(interval):
    with pytest.raises(ValueError, match='bbox3d_interval'):
        validate_bbox3d_schedule(interval, 1.)


@pytest.mark.parametrize('max_age', [True, 0., -1., float('nan'), float('inf')])
def test_invalid_max_age(max_age):
    with pytest.raises(ValueError, match='bbox3d_max_age'):
        validate_bbox3d_schedule(5, max_age)


def test_cloud_histories_follow_object_identity_and_reset_on_context_change():
    completed = threading.Event()
    calls = []

    def measure(items, inputs):
        for item in items:
            state = item['_cloud_filter_state']
            calls.append((item['bbox'][0], state.get('accepted')))
            state['accepted'] = item['bbox'][0]
            item['bbox3d_status'] = 'ok'

    pipeline = AsyncBBox3D(measure, interval=1, wake=completed.set)
    try:
        submit(pipeline, 1, items=[detection(0), detection(80)])
        assert completed.wait(1)
        completed.clear()
        submit(pipeline, 2, items=[detection(81), detection(1)])
        assert completed.wait(1)
        assert calls[-2:] == [(81, 80), (1, 0)]
        completed.clear()
        submit(pipeline, 3, items=[detection(2)], context='new')
        assert completed.wait(1)
        assert calls[-1] == (2, None)
        submit(pipeline, 3, items=[], context='new')
        assert pipeline.cloud_states == {}
    finally:
        pipeline.close()


def test_running_old_context_cannot_commit_cloud_history():
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()

    def measure(items, inputs):
        entered.set()
        assert release.wait(2)
        items[0]['_cloud_filter_state']['accepted'] = 'obsolete'

    pipeline = AsyncBBox3D(measure, wake=completed.set)
    try:
        submit(pipeline)
        assert entered.wait(1)
        submit(pipeline, 2, items=[], context='new')
        release.set()
        assert completed.wait(1)
        assert pipeline.cloud_states == {}
    finally:
        release.set()
        pipeline.close()
