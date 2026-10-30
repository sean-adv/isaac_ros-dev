"""Classification lifetime follows NvDCF sets, not numeric IDs or moving boxes."""
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from dinov3_cases.grounded import ObjectGallery, TrackedObjectRecognizer
from dinov3_cases.timing import ModelTimings


class CountingEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, image):
        self.calls += 1
        vector = torch.tensor(np.asarray(image).mean(axis=(0, 1)), dtype=torch.float32)
        return vector.repeat(4, 1)


@pytest.fixture
def setup(tmp_path):
    for label in ('red', 'blue'):
        (tmp_path / label).mkdir()
        Image.new('RGB', (32, 32), label).save(tmp_path / label / 'ref.png')
    encoder, gallery = CountingEncoder(), ObjectGallery()
    gallery.refresh(tmp_path, encoder, 32, 16)
    encoder.calls = 0
    pixels = np.zeros((40, 80, 3), dtype=np.uint8)
    pixels[:, :40] = [255, 0, 0]
    pixels[:, 40:] = [0, 0, 255]
    return TrackedObjectRecognizer(), encoder, gallery, pixels


def proposal(x=0):
    return dict(bbox=[x, 0, x + 30, 30], area=900,
                detection_score=.9, detection_label='box')


def run(setup, proposals, token, revision=1, threshold=.75):
    cache, encoder, gallery, pixels = setup
    return cache.update(pixels, proposals, encoder, gallery, 32, 16,
                        threshold, .05, token=token, gallery_revision=revision)


def test_nvdcf_frames_reuse_identity_but_keep_current_bbox_and_fresh_depth_inputs(setup):
    cache, encoder, _, _ = setup
    token = object()
    first, diagnostics, changed = run(setup, [proposal()], token)
    assert changed and cache.encoded and first[0]['label'] == 'red'
    first[0].update(bbox3d={'position': [1, 2, 3]}, track_id=99, bbox_visible=False)
    # Deliberately move onto a blue region: no DINO rerun within this set.
    moved = dict(proposal(45), bbox_source='nvdcf')
    current, reused, changed = run(setup, [moved], token)
    assert encoder.calls == 1 and not changed and not cache.encoded
    assert current[0]['label'] == 'red' and current[0]['bbox'] == moved['bbox']
    assert current[0]['bbox_source'] == 'nvdcf'
    assert not {'track_id', 'bbox3d', 'bbox_visible'} & current[0].keys()
    assert reused is diagnostics
    assert cache.diagnostic_detections[0]['bbox'] == proposal()['bbox']


def test_new_set_reclassifies_even_with_same_count_and_reordered_objects(setup):
    cache, encoder, _, _ = setup
    first, _, _ = run(setup, [proposal(), proposal(45)], object())
    assert [p['label'] for p in first] == ['red', 'blue']
    second, _, changed = run(setup, [proposal(45), proposal()], object())
    assert changed and cache.encoded and encoder.calls == 4
    assert [p['label'] for p in second] == ['blue', 'red']


def test_tracking_loss_drops_labels_and_diagnostic_crops(setup):
    cache, encoder, _, _ = setup
    token = object()
    run(setup, [proposal()], token)
    assert run(setup, [], token) == ([], [], True)
    assert cache.diagnostic_detections == []
    assert encoder.calls == 1
    assert run(setup, [], token) == ([], [], False)
    fresh, _, changed = run(setup, [proposal(45)], object())
    assert changed and fresh[0]['label'] == 'blue' and encoder.calls == 2


def test_gallery_and_threshold_changes_reclassify_descriptors_without_forward(setup):
    cache, encoder, gallery, _ = setup
    token = object()
    run(setup, [proposal()], token)
    red_index = gallery.labels.index('red')
    gallery.labels[red_index] = 'renamed_red'
    updated, _, changed = run(setup, [proposal(1)], token, revision=2)
    assert changed and not cache.encoded and encoder.calls == 1
    assert updated[0]['label'] == 'renamed_red'
    gallery.features[red_index] *= .5
    updated, _, changed = run(setup, [proposal(2)], token, revision=2, threshold=.9)
    assert changed and updated[0]['label'] == 'unknown' and encoder.calls == 1


def test_same_tracker_in_synchronous_mode_uses_frame_token(setup):
    active = object()
    run(setup, [proposal()], (active, 1))
    second, _, changed = run(setup, [proposal(45)], (active, 2))
    assert changed and second[0]['label'] == 'blue' and setup[1].calls == 2


def test_partial_failed_new_set_never_commits_or_reuses_old_classification(setup):
    cache, encoder, _, _ = setup
    run(setup, [proposal()], object())
    token = object()
    original = encoder.encode
    encoder.encode = Mock(side_effect=[torch.ones((4, 3)), RuntimeError('GPU failure')])
    with pytest.raises(RuntimeError, match='GPU failure'):
        run(setup, [proposal(), proposal(45)], token)
    encoder.encode = original
    result, _, changed = run(setup, [proposal(45)], token)
    assert changed and cache.encoded and result[0]['label'] == 'blue'


def test_count_changes_reencode_current_crops_without_reusing_old_labels(setup):
    cache, encoder, _, _ = setup
    token = object()
    run(setup, [proposal(), proposal(45)], token)
    # First object disappears. The surviving blue object must not inherit red.
    result, diagnostics, changed = run(setup, [proposal(45)], token)
    assert changed and cache.encoded and encoder.calls == 3
    assert result[0]['label'] == 'blue'
    assert result[0]['dino_score'] == pytest.approx(1)
    assert result[0]['lab_score'] == pytest.approx(1)
    assert diagnostics[0]['context']['roi'].getpixel((0, 0)) == (0, 0, 255)
    result, _, changed = run(setup, [proposal(45), proposal()], token)
    assert changed and cache.encoded and encoder.calls == 5
    assert [item['label'] for item in result] == ['blue', 'red']
    assert not run(setup, [proposal(45), proposal()], token)[2]
    assert encoder.calls == 5


def test_empty_set_can_recover_with_the_same_tracker_token(setup):
    token = object()
    run(setup, [proposal()], token)
    run(setup, [], token)
    result, _, changed = run(setup, [proposal(45)], token)
    assert changed and setup[0].encoded and result[0]['label'] == 'blue'


def test_failed_count_change_discards_invalid_cache_before_retry(setup):
    cache, encoder, _, _ = setup
    token = object()
    run(setup, [proposal(), proposal(45)], token)
    original = encoder.encode
    encoder.encode = Mock(side_effect=RuntimeError('GPU failure'))
    with pytest.raises(RuntimeError, match='GPU failure'):
        run(setup, [proposal(45)], token)
    assert cache.classifications is None and cache.diagnostic_detections == []
    encoder.encode = original
    result, _, changed = run(setup, [proposal(45)], token)
    assert changed and result[0]['label'] == 'blue'


@pytest.mark.parametrize('scenario', ['handoff', 'same_frame', 'partial_loss', 'total_loss'])
def test_node_worker_encodes_on_handoff_and_keeps_diagnostics_on_recognition_frame(setup, tmp_path, scenario):
    import threading
    from types import SimpleNamespace
    from unittest.mock import patch
    from dinov3_cases.node import CaseNode
    from dinov3_cases.frame_pool import SharedFramePool
    from dinov3_cases.tracking import BBoxSmoother

    _, encoder, gallery, pixels = setup
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
        from dinov3_cases.visualization import render_display_overlay
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
        with patch('dinov3_cases.node.Encoder', return_value=encoder), \
                patch('dinov3_cases.node.feature_stride', return_value=16), \
                patch('dinov3_cases.node.SharedMemoryGroundingDetector'), \
                patch('dinov3_cases.node.ObjectGallery', return_value=gallery), \
                patch('dinov3_cases.node.AsyncNvDCFProposals', return_value=scheduler):
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
