"""Single-shot seed recognition and NvDCF classification/diagnostic lifetime."""
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from dinov3_cases.recognition.grounded import ObjectGallery, recognize_objects
from dinov3_cases.timing import ModelTimings
from encoder_stub import ColorEncoder


@pytest.fixture
def setup(tmp_path):
    for label in ('red', 'blue'):
        (tmp_path / label).mkdir()
        Image.new('RGB', (32, 32), label).save(tmp_path / label / 'ref.png')
    encoder, gallery = ColorEncoder(size=32), ObjectGallery()
    gallery.refresh(tmp_path, encoder, 32, 16)
    encoder.calls = 0
    pixels = np.zeros((40, 80, 3), dtype=np.uint8)
    pixels[:, :40] = [255, 0, 0]
    pixels[:, 40:] = [0, 0, 255]
    return encoder, gallery, pixels


def proposal(x=0):
    return dict(bbox=[x, 0, x + 30, 30], area=900,
                detection_score=.9, detection_label='box')


def test_single_shot_recognition_uses_current_crop_order_and_empty_results(setup):
    encoder, gallery, pixels = setup
    image = Image.fromarray(pixels)
    first = recognize_objects(image, [proposal(), proposal(45)], encoder, gallery, 32, 16, .75)
    assert [item['label'] for item in first] == ['red', 'blue']
    diagnostics = []
    second = recognize_objects(image, [proposal(45)], encoder, gallery, 32, 16, .75,
                               diagnostics=diagnostics, defer_diagnostics=True)
    assert second[0]['label'] == 'blue' and encoder.calls == 3
    assert diagnostics[0]['context']['roi'].getpixel((0, 0)) == (0, 0, 255)
    assert recognize_objects(image, [], encoder, gallery, 32, 16, .75) == []
    assert encoder.calls == 3


def test_failed_recognition_can_retry_without_retaining_partial_labels(setup):
    encoder, gallery, pixels = setup
    original = encoder.encode_with_descriptor
    encoder.encode_with_descriptor = Mock(side_effect=[(torch.ones((4, 3)), torch.ones(3)), RuntimeError('GPU failure')])
    with pytest.raises(RuntimeError, match='GPU failure'):
        recognize_objects(Image.fromarray(pixels), [proposal(), proposal(45)],
                          encoder, gallery, 32, 16, .75)
    encoder.encode_with_descriptor = original
    result = recognize_objects(Image.fromarray(pixels), [proposal(45)],
                               encoder, gallery, 32, 16, .75)
    assert result[0]['label'] == 'blue'


@pytest.mark.parametrize('scenario', ['handoff', 'same_frame', 'partial_loss', 'total_loss'])
def test_node_worker_encodes_on_handoff_and_keeps_diagnostics_on_recognition_frame(setup, tmp_path, scenario):
    import threading
    from types import SimpleNamespace
    from unittest.mock import patch
    from dinov3_cases.ros.node import CaseNode
    from dinov3_cases.frame_pool import SharedFramePool
    from dinov3_cases.tracking.identities import BBoxSmoother

    encoder, gallery, pixels = setup
    encoder.device, encoder.description = 'cpu', 'counting encoder'
    encoder.model = object()
    gallery.refresh = Mock(return_value=False)
    scheduler = Mock(active=Mock(initialized=True), last_metrics={},
                     last_recognition=None, lock=threading.Lock())
    scheduler.scheduler_metrics.return_value = {}
    # seq, bbox x positions, new tracker set, expected diagnostic revision,
    # recognition source, recognition frame. Same-frame handoff used to truncate.
    rows = {
        'handoff': [(0, [0], False, 1, 'dino', 0),
                    (1, [0], False, 1, 'cached', 0),
                    (2, [45], True, 2, 'dino', 2)],
        'same_frame': [(0, [0], False, 1, 'dino', 0),
                       (0, [45, 0], True, 2, 'dino', 0),
                       (1, [45, 0], False, 2, 'cached', 0)],
        'partial_loss': [(0, [0, 45], False, 1, 'dino', 0),
                         (1, [45], False, 1, 'cached', 0),
                         (2, [45], False, 1, 'cached', 0)],
        'total_loss': [(0, [0, 45], False, 1, 'dino', 0),
                       (1, [], False, 2, 'none', None),
                       (2, [45], True, 3, 'dino', 2)],
    }[scenario]
    scheduler.has_result.return_value = True
    calls = []
    classified = {}

    def proposal_update(*args, **kwargs):
        index = len(calls)
        calls.append(index)
        _, positions, new_set, _, _, _ = rows[index]
        if new_set:
            scheduler.active = Mock(initialized=True)
        if new_set or index == 0:
            items, scheduler.last_recognition = kwargs['recognize'](
                args[0], [proposal(x) for x in positions], args[1])
            classified.clear()
            classified.update(zip(positions, items))
        return [dict(classified[x], bbox=proposal(x)['bbox']) for x in positions], (
            'sam3' if new_set or index == 0 else 'nvdcf')

    scheduler.update.side_effect = proposal_update
    parameters = dict(engine_path='unused', grounding_engine_path='unused',
                      grounding_worker_python='unused', grounding_ipc_timeout=1.,
                      nvdcf_config='unused', nvdcf_width=640, nvdcf_height=384,
                      nvdcf_max_missed_detections=3,
                      nvdcf_max_missed_frames=30,
                      grounding_max_detections=20,
                      grounding_containment_threshold=.95,
                      grounding_min_bbox_width=0, grounding_min_bbox_height=0,
                      grounding_min_bbox_area=0, pose_enabled=False)
    logger = Mock()
    node = SimpleNamespace(
        lock=threading.Lock(), generation=0, frame=Image.fromarray(pixels), frame_seq=0,
        source='upload', camera_received_at=None, root=tmp_path, stop=threading.Event(),
        image_changed=threading.Event(), frame_pool=SharedFramePool(),
        grounding_settings=dict(prompt='box.', box_threshold=.1, detection_interval=10,
                                recognition_threshold=.75, margin=.05),
        bbox_smoother=BBoxSmoother(1), depth_inputs_for=lambda _: None,
        performance={}, model_timings=ModelTimings(),
        pending_render=None, pending_publication=None,
        object_debug_revision=0, object_debug_snapshot=None,
        render_changed=threading.Event(), publish_changed=threading.Event(),
        param=parameters.__getitem__, get_logger=lambda: logger,
        video_at=123., clear_object_debug=Mock(),
    )
    observations = []

    def ticks():
        from dinov3_cases.api.visualization import render_display_overlay
        for seq, positions, _, _, _, _ in rows:
            node.frame_seq = seq
            yield
            assert node.video_at == 123.
            assert node.pending_publication[0] is node.detections
            # Every result, including no surviving boxes, still queues an image.
            frame, items, *_ = node.pending_render
            overlay = render_display_overlay(frame.pixels, items, mode='2d')
            assert overlay.shape == (540, 960, 3)
            observations.append((node.object_debug_revision, node.object_debug_snapshot,
                                 [item['label'] for item in node.detections], node.performance.copy()))

    node.acquire_input_frame = lambda *a, **kw: CaseNode.acquire_input_frame(node, *a, **kw)
    node.worker_ticks = ticks
    try:
        with patch('dinov3_cases.ros.node.Encoder', return_value=encoder), \
                patch('dinov3_cases.ros.node.feature_stride', return_value=16), \
                patch('dinov3_cases.ros.node.SharedMemoryGroundingDetector'), \
                patch('dinov3_cases.ros.node.ObjectGallery', return_value=gallery), \
                patch('dinov3_cases.ros.node.AsyncNvDCFProposals', return_value=scheduler):
            CaseNode.run_grounded_worker(node)
        assert encoder.calls == sum(len(row[1]) for row in rows if row[4] == 'dino')
        assert [row[0] for row in observations] == [row[3] for row in rows]
        for index in range(1, len(rows)):
            assert (observations[index][1] is observations[index-1][1]) == (rows[index][3] == rows[index-1][3])
        assert [row[2] for row in observations] == [
            ['blue' if x == 45 else 'red' for x in row[1]] for row in rows
        ]
        assert [row[3]['recognition_source'] for row in observations] == [row[4] for row in rows]
        assert [row[3]['recognition_frame_id'] for row in observations] == [row[5] for row in rows]
        assert [row[3]['frame_id'] for row in observations] == [row[0] for row in rows]
        assert not any('track_id' in detection for detection in node.detections)
        logger.error.assert_not_called()
        node.clear_object_debug.assert_not_called()
    finally:
        if node.pending_render is not None:
            node.pending_render[0].release()
        node.frame_pool.close()
