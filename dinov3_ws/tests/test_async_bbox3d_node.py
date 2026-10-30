"""The node's 3D cadence stays independent of SAM3 and never waits for depth work."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from dinov3_cases.async_bbox3d import AsyncBBox3D
from dinov3_cases.frame_pool import SharedFramePool
from dinov3_cases.node import CaseNode
from dinov3_cases.tracking import BBoxSmoother


@pytest.mark.parametrize('grounding_interval', [1, 10])
def test_node_keeps_five_frame_measurements_independent_of_grounding(tmp_path, grounding_interval):
    entered, release = threading.Event(), threading.Event()
    pixels = np.zeros((80, 100, 3), dtype=np.uint8)
    measured = []

    def measure(items, inputs):
        measured.append(inputs)
        entered.set()
        assert release.wait(3)
        for item in items:
            item['bbox3d_status'] = 'test_measurement'

    boxes = AsyncBBox3D(measure)
    recognition = dict(frame_id=1, snapshot=None, class_count=1, reference_count=1, recognize_ms=0.)
    scheduler = Mock(last_recognition=recognition, last_metrics={}, lock=threading.Lock())
    scheduler.has_result.return_value = True
    scheduler.scheduler_metrics.return_value = {}
    scheduler.update.return_value = ([dict(bbox=[20, 20, 60, 40], label='medicine', score=.9)], 'nvdcf')
    params = dict(engine_path='unused', grounding_engine_path='unused',
                  grounding_worker_python='unused', grounding_ipc_timeout=1.,
                  nvdcf_config='unused', nvdcf_width=640, nvdcf_height=384,
                  nvdcf_max_missed_detections=3,
                  nvdcf_max_missed_frames=30,
                      grounding_max_detections=20,
                      grounding_containment_threshold=.95,
                  grounding_min_bbox_width=0, grounding_min_bbox_height=0,
                  grounding_min_bbox_area=0, pose_enabled=True)
    from dinov3_cases.timing import ModelTimings
    node = SimpleNamespace(model_timings=ModelTimings(),
        lock=threading.Lock(), generation=0, source='camera', root=tmp_path,
        frame=Image(width=100, height=80, header=Header(frame_id='camera')), frame_seq=1,
        camera_received_at=time.monotonic(), stop=threading.Event(), image_changed=threading.Event(),
        frame_pool=SharedFramePool(), bbox3d_worker=boxes, bbox_smoother=BBoxSmoother(1),
        bridge=Mock(), performance={}, pending_render=None, pending_publication=None,
        object_debug_revision=0, object_debug_snapshot=None,
        render_changed=threading.Event(), publish_changed=threading.Event(),
        param=params.__getitem__, get_logger=Mock(return_value=Mock()),
        grounding_settings=dict(prompt='box.', box_threshold=.1, detection_interval=grounding_interval,
                                recognition_threshold=.75, margin=.05),
    )
    node.bridge.imgmsg_to_cv2.return_value = pixels
    node.depth_inputs_for = lambda frame: (node.frame_seq, 'calibration')
    node.acquire_input_frame = lambda *a, **kw: CaseNode.acquire_input_frame(node, *a, **kw)
    observed = []

    def ticks():
        for seq in (1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11):
            node.frame_seq = seq
            yield
            if seq == 1:
                assert entered.wait(1)
            assert node.performance['bbox3d_schedule']['interval'] == 5
            assert node.performance['detection_interval'] == grounding_interval
            observed.append(seq)
        assert boxes.pending[4]['frame_key'] == [0, 11]

    node.worker_ticks = ticks
    encoder = Mock(description='test', device='cpu', model=object())
    try:
        with patch('dinov3_cases.node.Encoder', return_value=encoder), \
                patch('dinov3_cases.node.feature_stride', return_value=16), \
                patch('dinov3_cases.node.SharedMemoryGroundingDetector'), \
                patch('dinov3_cases.node.ObjectGallery'), \
                patch('dinov3_cases.node.AsyncNvDCFProposals', return_value=scheduler):
            CaseNode.run_grounded_worker(node)
        assert len(observed) == 12
        assert measured == [(1, 'calibration')]
        node.get_logger.return_value.error.assert_not_called()
    finally:
        release.set()
        boxes.close()
        if node.pending_render is not None:
            node.pending_render[0].release()
        node.frame_pool.close()


@pytest.fixture
def coalescing_node(tmp_path):
    """A real foreground loop with controllable background completion events."""
    pixels = np.zeros((80, 100, 3), dtype=np.uint8)
    flags = dict(proposals=False, depth=False)
    recognition = dict(frame_id=1, snapshot=None, class_count=1, reference_count=1, recognize_ms=0.)
    scheduler = Mock(last_recognition=recognition, last_metrics={}, lock=threading.Lock())
    scheduler.force_detection = False
    scheduler.has_result.side_effect = lambda: flags['proposals'] or scheduler.force_detection
    scheduler.scheduler_metrics.return_value = {}
    label = ['original']

    def proposals(*args, **kwargs):
        flags['proposals'] = False
        scheduler.force_detection = False
        return ([dict(bbox=[20, 20, 60, 40], track_id=1, label=label[0], score=.9,
                      detection_label='box')], 'nvdcf')

    scheduler.update.side_effect = proposals
    boxes = Mock()
    boxes.has_result.side_effect = lambda: flags['depth']

    def depth_update(detections, *args, **kwargs):
        flags['depth'] = False
        return detections, None, dict(interval=1, measurement_ms=5., pending=False)

    boxes.update.side_effect = depth_update
    params = dict(engine_path='unused', grounding_engine_path='unused', grounding_worker_python='unused',
                  grounding_ipc_timeout=1., nvdcf_config='unused', nvdcf_width=640, nvdcf_height=384,
                  nvdcf_max_missed_detections=3, nvdcf_max_missed_frames=30,
                  grounding_containment_threshold=.8, grounding_max_detections=20,
                  grounding_min_bbox_width=0, grounding_min_bbox_height=0, grounding_min_bbox_area=0,
                  pose_enabled=True)
    from dinov3_cases.timing import ModelTimings
    node = SimpleNamespace(model_timings=ModelTimings(),
        lock=threading.Lock(), generation=0, source='camera', root=tmp_path,
        frame=Image(width=100, height=80, header=Header(frame_id='camera')), frame_seq=1,
        camera_received_at=time.monotonic(), frame_pool=SharedFramePool(),
        bbox3d_worker=boxes, bbox_smoother=BBoxSmoother(1), bridge=Mock(),
        performance={}, pending_render=None, pending_publication=None,
        object_debug_revision=0, object_debug_snapshot=None,
        stop=threading.Event(), close_grounding_worker=Mock(),
        image_changed=threading.Event(), render_changed=threading.Event(), publish_changed=threading.Event(),
        param=params.__getitem__, get_logger=Mock(return_value=Mock()),
        grounding_settings=dict(prompt='box.', box_threshold=.5, detection_interval=3,
                                recognition_threshold=.8, margin=.05))
    node.bridge.imgmsg_to_cv2.return_value = pixels
    node.depth_inputs_for = Mock(return_value=(object(), None))
    node.acquire_input_frame = Mock(side_effect=lambda *a, **kw: CaseNode.acquire_input_frame(node, *a, **kw))
    encoder = Mock(description='test', device='cpu', model=object())

    def run(ticks):
        node.worker_ticks = ticks
        with patch('dinov3_cases.node.Encoder', return_value=encoder), \
                patch('dinov3_cases.node.feature_stride', return_value=16), \
                patch('dinov3_cases.node.SharedMemoryGroundingDetector'), \
                patch('dinov3_cases.node.ObjectGallery'), \
                patch('dinov3_cases.node.AsyncNvDCFProposals', return_value=scheduler):
            CaseNode.run_grounded_worker(node)
        node.get_logger.return_value.error.assert_not_called()

    try:
        yield node, scheduler, boxes, flags, label, run
    finally:
        if node.pending_render is not None:
            node.pending_render[0].release()
        node.frame_pool.close()


def test_live_rgb_coalesces_background_updates_without_copy_tracking_or_render(coalescing_node):
    node, scheduler, boxes, flags, label, run = coalescing_node

    def ticks():
        node.camera_received_at = time.monotonic()
        yield
        first_job = node.pending_render
        first_completed = node.performance['completed_at']
        flags.update(proposals=True, depth=True)
        label[0] = 'updated'
        yield  # Same RGB: both background completions must wait for the next RGB.
        assert node.pending_render is first_job
        assert node.performance['completed_at'] == first_completed
        assert flags == dict(proposals=True, depth=True)
        assert scheduler.update.call_count == boxes.update.call_count == 1
        assert node.acquire_input_frame.call_count == node.bridge.imgmsg_to_cv2.call_count == 1
        assert node.depth_inputs_for.call_count == 1
        node.frame_seq = 2
        node.camera_received_at = time.monotonic()
        yield
        assert scheduler.update.call_count == boxes.update.call_count == 2
        assert node.performance['frame_id'] == 2
        assert node.detections[0]['label'] == 'updated'
        assert flags == dict(proposals=False, depth=False)
        assert node.performance['new_rgb_result']
        # A generation change with the same sequence must still be processed.
        node.generation = 1
        yield
        assert scheduler.update.call_count == 3
        assert node.performance['result_fps'] is None

    run(ticks)


@pytest.mark.parametrize('camera_mode', [True, False])
def test_roi_filters_sam3_and_moving_tracks_while_outputs_keep_original_image(coalescing_node, camera_mode):
    node, scheduler, boxes, flags, label, run = coalescing_node
    from PIL import Image as PILImage
    node.sam3_roi = (40, 30, 40, 30)  # Native [20, 15, 60, 45].
    node.sam3_event_sequence, node.sam3_events = 0, []
    if not camera_mode:
        node.source = 'upload'
        node.frame = PILImage.new('RGB', (100, 80))
    inside = dict(bbox=[20, 20, 60, 40], label='medicine', score=.9, detection_label='box')
    outside = dict(bbox=[55, 20, 70, 40], label='medicine', score=.9, detection_label='box')
    raw = [inside, outside]
    seeds = []

    def update(pixels, token, context, interval, detect, **kwargs):
        assert pixels.shape == (80, 100, 3)
        proposals, _ = detect(pixels)
        seeds.append(proposals)
        return ([inside, outside] if token[1] == 1 else [outside]), 'nvdcf'

    scheduler.update.side_effect = update

    def ticks():
        node.grounding_detector.detect.return_value = raw
        node.grounding_detector.last_metrics = {'worker_detect_ms': 1.0}
        yield
        assert seeds[0] == ([inside] if camera_mode else raw)
        assert [item['bbox'] for item in node.detections] == (
            [inside['bbox']] if camera_mode else [inside['bbox'], outside['bbox']])
        assert node.pending_render[0].pixels.shape == (80, 100, 3)
        assert node.pending_publication[0] == node.detections
        if camera_mode:
            assert boxes.update.call_args.args[0][0]['bbox'] == [20, 20, 60, 40]
        node.frame_seq = 2
        yield
        assert [item['bbox'] for item in node.detections] == ([] if camera_mode else [outside['bbox']])

    run(ticks)
    assert raw == [inside, outside]  # Filtering must not rewrite SAM3 geometry.


def test_paused_camera_flushes_background_once_without_counting_an_extra_rgb(coalescing_node):
    node, scheduler, boxes, flags, label, run = coalescing_node
    from dinov3_cases.timing import ResultRate
    rate = ResultRate()

    def ticks():
        yield
        flags['depth'] = True
        node.camera_received_at = time.monotonic() - .25
        yield  # Paused input must still publish the first available 3D result.
        assert scheduler.update.call_count == 2
        assert not node.performance['new_rgb_result']
        assert node.performance['result_fps_instant'] is None
        assert node.performance['result_fps'] is None
        assert not flags['depth']
        completed = node.performance['completed_at']
        yield  # No new background result: do not refresh repeatedly while paused.
        assert scheduler.update.call_count == 2
        assert node.performance['completed_at'] == completed

    with patch('dinov3_cases.node.ResultRate', return_value=rate), \
            patch.object(rate, 'update', wraps=rate.update) as count:
        run(ticks)
    assert count.call_count == 1


def test_upload_applies_background_labels_without_waiting_for_new_rgb(coalescing_node):
    node, scheduler, boxes, flags, label, run = coalescing_node
    from PIL import Image as PILImage
    node.source = 'upload'
    node.frame = PILImage.new('RGB', (100, 80))

    def ticks():
        yield
        flags['proposals'] = True
        label[0] = 'classified'
        yield
        assert scheduler.update.call_count == 2
        assert node.detections[0]['label'] == 'classified'
        assert not node.performance['new_rgb_result']
        assert node.performance['result_fps'] is None

    run(ticks)
