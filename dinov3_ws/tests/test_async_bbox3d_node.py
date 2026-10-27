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
    scheduler.update.return_value = ([dict(bbox=[20, 20, 60, 40], label='medicine', score=.9)], 'kcf')
    params = dict(engine_path='unused', grounding_engine_path='unused',
                  grounding_worker_python='unused', grounding_ipc_timeout=1.,
                  kcf_roi_padding=.5, kcf_max_roi_side=0, kcf_replay_max_frames=8, kcf_replay_budget_ms=150.,
                  kcf_max_missed_detections=3,
                  kcf_max_unconfirmed_seconds=5.,
                  grounding_nms_iou=1., grounding_max_detections=20,
                  grounding_min_bbox_width=0, grounding_min_bbox_height=0,
                  grounding_min_bbox_area=0, pose_enabled=True)
    node = SimpleNamespace(
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
                patch('dinov3_cases.node.AsyncKCFProposals', return_value=scheduler):
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
