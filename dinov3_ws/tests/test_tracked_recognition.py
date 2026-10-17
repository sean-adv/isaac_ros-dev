"""Classification lifetime follows KCF sets, not numeric IDs or moving boxes."""
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from dinov3_cases.grounded import ObjectGallery, TrackedObjectRecognizer


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


def test_kcf_frames_reuse_identity_but_keep_current_bbox_and_fresh_depth_inputs(setup):
    cache, encoder, _, _ = setup
    token = object()
    first, diagnostics, changed = run(setup, [proposal()], token)
    assert changed and cache.encoded and first[0]['label'] == 'red'
    first[0].update(bbox3d={'position': [1, 2, 3]}, track_id=99, bbox_visible=False)
    # Deliberately move onto a blue region: no DINO rerun within this set.
    moved = dict(proposal(45), bbox_source='kcf')
    current, reused, changed = run(setup, [moved], token)
    assert encoder.calls == 1 and not changed and not cache.encoded
    assert current[0]['label'] == 'red' and current[0]['bbox'] == moved['bbox']
    assert current[0]['bbox_source'] == 'kcf'
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


def test_count_change_cannot_silently_assign_another_objects_label(setup):
    token = object()
    run(setup, [proposal()], token)
    with pytest.raises(ValueError, match='count changed'):
        run(setup, [proposal(), proposal(45)], token)


def test_async_handoff_triggers_dino_but_background_submission_does_not(setup):
    from dinov3_cases.async_kcf import AsyncKCFProposals

    class Tracker:
        def init(self, image, box):
            self.box = box

        def update(self, image):
            return True, self.box

    pipeline = AsyncKCFProposals(Tracker)
    detected = []

    def detect(pixels):
        detected.append(True)
        return [proposal(0 if len(detected) == 1 else 45)], {}

    def step(seq):
        result = pipeline.update(setup[3], seq, 'camera', 3, detect)
        if result is None:
            return None
        proposals, source = result
        # The production worker binds the cache to this active set object.
        detections, _, _ = run(setup, proposals, (pipeline.active, 'camera', None))
        return detections, source

    try:
        assert step(0) is None
        pipeline.thread.join(3)
        assert not pipeline.thread.is_alive()
        assert step(0)[0][0]['label'] == 'red'
        assert setup[1].calls == 1
        assert step(1)[1] == step(2)[1] == 'kcf'
        assert step(3)[1] == 'kcf'  # Submitting SAM3 must not rerun DINO.
        assert setup[1].calls == 1
        pipeline.thread.join(3)
        assert not pipeline.thread.is_alive()
        detections, source = step(3)
        assert source == 'sam3' and detections[0]['label'] == 'blue'
        assert setup[1].calls == 2
        step(4)
        assert setup[1].calls == 2
    finally:
        pipeline.close()


def test_node_worker_encodes_on_handoff_and_keeps_diagnostics_on_recognition_frame(setup, tmp_path):
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
    scheduler = Mock(active=object(), last_metrics={})
    scheduler.waiting_for_initial_detection.return_value = False
    scheduler.has_result.return_value = False
    scheduler.scheduler_metrics.return_value = {}
    calls = []

    def proposal_update(*args, **kwargs):
        index = len(calls)
        calls.append(index)
        if index == 2:
            scheduler.active = object()
        return [proposal(45 if index == 2 else 0)], ('kcf' if index == 1 else 'sam3')

    scheduler.update.side_effect = proposal_update
    parameters = dict(engine_path='unused', grounding_engine_path='unused',
                      grounding_worker_python='unused', grounding_ipc_timeout=1.,
                      kcf_roi_padding=.5, kcf_replay_max_frames=8, kcf_replay_budget_ms=150.,
                      grounding_nms_iou=1., grounding_max_detections=20,
                      grounding_min_bbox_width=0, grounding_min_bbox_height=0,
                      grounding_min_bbox_area=0, tracking_enabled=False, pose_enabled=False)
    node = SimpleNamespace(
        lock=threading.Lock(), generation=0, frame=Image.fromarray(pixels), frame_seq=0,
        source='upload', camera_received_at=None, root=tmp_path, stop=threading.Event(),
        image_changed=threading.Event(), frame_pool=SharedFramePool(),
        grounding_settings=dict(prompt='box.', box_threshold=.1, detection_interval=10,
                                recognition_threshold=.75, margin=.05),
        bbox_smoother=BBoxSmoother(1), depth_inputs_for=lambda _: None,
        performance={}, pending_render=None, pending_publication=None,
        object_debug_revision=0, object_debug_snapshot=None,
        render_changed=threading.Event(), publish_changed=threading.Event(),
        param=parameters.__getitem__, get_logger=lambda: Mock(),
    )
    observations = []

    def ticks():
        for seq in range(3):
            node.frame_seq = seq
            yield
            observations.append((node.object_debug_revision, node.object_debug_snapshot,
                                 node.detections[0]['label'], node.performance.copy()))

    node.worker_ticks = ticks
    try:
        with patch('dinov3_cases.node.Encoder', return_value=encoder), \
                patch('dinov3_cases.node.feature_stride', return_value=16), \
                patch('dinov3_cases.node.SharedMemoryGroundingDetector'), \
                patch('dinov3_cases.node.ObjectGallery', return_value=gallery), \
                patch('dinov3_cases.node.AsyncKCFProposals', return_value=scheduler):
            CaseNode.run_grounded_worker(node)
        assert encoder.calls == 2
        assert [row[0] for row in observations] == [1, 1, 2]
        assert observations[0][1] is observations[1][1]
        assert observations[2][1] is not observations[1][1]
        assert [row[2] for row in observations] == ['red', 'red', 'blue']
        assert [row[3]['recognition_source'] for row in observations] == ['dino', 'cached', 'dino']
        assert [row[3]['recognition_frame_id'] for row in observations] == [0, 0, 2]
        assert [row[3]['frame_id'] for row in observations] == [0, 1, 2]
        assert not any('track_id' in detection for detection in node.detections)
    finally:
        if node.pending_render is not None:
            node.pending_render[0].release()
        node.frame_pool.close()
