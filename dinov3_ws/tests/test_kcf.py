"""Periodic detection scheduling and real KCF motion without GPU models."""

from types import SimpleNamespace
import threading
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from dinov3_cases.kcf import KCFProposals, kcf_factory, validate_detection_interval


def proposal():
    return dict(bbox=[10, 12, 30, 32], area=400,
                detection_score=0.9, detection_label="box")


@pytest.fixture
def pipeline():
    trackers = []

    def factory():
        tracker = SimpleNamespace(init=Mock(return_value=None),
                                  update=Mock(return_value=(True, (14, 15, 20, 20))))
        trackers.append(tracker)
        return tracker

    return KCFProposals(factory, roi_padding=None), Mock(return_value=[proposal()]), trackers


def step(pipeline, seq, interval=3, context="camera", pixels=None):
    tracker, detect, _ = pipeline
    if pixels is None:
        pixels = np.zeros((80, 100, 3), dtype=np.uint8)
    return tracker.update(pixels, seq, context, interval, detect)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "5", None])
def test_invalid_interval(value):
    with pytest.raises(ValueError, match="integer"):
        validate_detection_interval(value)


def test_sam3_then_n_minus_one_kcf_frames(pipeline):
    # Callback sequence gaps are intentionally not counted.
    sources = [step(pipeline, seq)[1] for seq in (1, 5, 9, 12, 15, 16, 20)]
    assert sources == ["sam3", "kcf", "kcf", "sam3", "kcf", "kcf", "sam3"]
    assert pipeline[1].call_count == 3


def test_interval_one_never_creates_kcf(pipeline):
    assert [step(pipeline, n, interval=1)[1] for n in range(4)] == ["sam3"] * 4
    assert pipeline[2] == []


def test_tracking_updates_bbox_area_and_preserves_sam_score(pipeline):
    pixels = np.full((80, 100, 3), (10, 20, 30), dtype=np.uint8)
    step(pipeline, 1, pixels=pixels)
    tracker = pipeline[2][0]
    np.testing.assert_array_equal(tracker.init.call_args.args[0][0, 0], [30, 20, 10])
    assert tracker.init.call_args.args[1] == (10, 12, 20, 20)
    tracker.update.return_value = (True, (-4.5, 15.2, 30, 90))
    items, source = step(pipeline, 2)
    assert source == "kcf"
    assert items[0] == dict(proposal(), bbox=[0, 15, 26, 80], area=1690, bbox_source="kcf")


@pytest.mark.parametrize("result", [(False, (0, 0, 0, 0)),
                                    (True, (float("nan"), 0, 10, 10)),
                                    (True, (101, 0, 10, 10)),
                                    (True, (10, 10, -1, 5))])
def test_failed_or_invalid_track_redetects_current_frame(pipeline, result):
    step(pipeline, 1)
    pipeline[2][0].update.return_value = result
    items, source = step(pipeline, 2)
    assert source == "sam3"
    assert items[0]["bbox"] == proposal()["bbox"]
    assert pipeline[1].call_count == 2
    assert step(pipeline, 3)[1] == "kcf"


def test_any_failed_object_refreshes_all_trackers(pipeline):
    pipeline[1].return_value = [proposal(), dict(proposal(), bbox=[40, 40, 60, 60])]
    step(pipeline, 1)
    pipeline[2][1].update.side_effect = cv2.error("lost")
    assert step(pipeline, 2)[1] == "sam3"
    assert len(pipeline[0].trackers) == 2
    assert pipeline[1].call_count == 2


def test_empty_detection_clears_old_boxes_and_waits_for_interval(pipeline):
    step(pipeline, 1)
    pipeline[1].return_value = []
    assert step(pipeline, 2)[1] == "kcf"
    assert step(pipeline, 3)[1] == "kcf"
    assert step(pipeline, 4) == ([], "sam3")
    assert pipeline[0].trackers == []
    assert step(pipeline, 5) == ([], "kcf")
    assert pipeline[1].call_count == 2


def test_same_frame_gallery_refresh_does_not_update_or_mutate_cache(pipeline):
    items, _ = step(pipeline, 1)
    items[0]["bbox"][0] = 99
    assert step(pipeline, 1)[0][0]["bbox"][0] == 10
    step(pipeline, 2)
    step(pipeline, 2)
    assert pipeline[2][0].update.call_count == 1
    assert step(pipeline, 3)[1] == "kcf"
    assert pipeline[1].call_count == 1


def test_context_and_interval_changes_force_detection_even_same_frame(pipeline):
    step(pipeline, 1)
    assert step(pipeline, 1, context="upload")[1] == "sam3"
    assert step(pipeline, 1, context="upload", interval=5)[1] == "sam3"
    assert pipeline[1].call_count == 3


def test_failed_initialization_redetects_next_frame(pipeline):
    step(pipeline, 1)
    pipeline[2][0].update.return_value = (False, (0, 0, 0, 0))
    original_factory = pipeline[0].factory

    def bad_init():
        tracker = original_factory()
        tracker.init.return_value = False
        return tracker

    pipeline[0].factory = bad_init
    step(pipeline, 2)
    assert step(pipeline, 3)[1] == "sam3"


def test_detector_error_retries_without_reusing_partially_updated_tracks(pipeline):
    step(pipeline, 1)
    pipeline[2][0].update.return_value = (False, (0, 0, 0, 0))
    pipeline[1].side_effect = RuntimeError("worker failed")
    with pytest.raises(RuntimeError):
        step(pipeline, 2)
    pipeline[1].side_effect = None
    assert step(pipeline, 2)[1] == "sam3"


def test_factory_legacy_and_missing_support(monkeypatch):
    monkeypatch.delattr(cv2, "TrackerKCF_create", raising=False)
    legacy = SimpleNamespace(TrackerKCF_create=lambda: "legacy")
    monkeypatch.setattr(cv2, "legacy", legacy, raising=False)
    assert kcf_factory()() == "legacy"
    monkeypatch.delattr(legacy, "TrackerKCF_create")
    with pytest.raises(ValueError, match="opencv-contrib-python"):
        kcf_factory()


def test_real_opencv_kcf_tracks_beyond_initial_box_padding():
    factory = kcf_factory()
    rng = np.random.default_rng(123)
    texture = rng.integers(0, 256, (36, 36, 3), dtype=np.uint8)
    pixels = np.zeros((140, 180, 3), dtype=np.uint8)
    pixels[40:76, 50:86] = texture
    tracker = KCFProposals(factory, roi_padding=None)
    detect = Mock(return_value=[dict(proposal(), bbox=[50, 40, 86, 76])])
    tracker.update(pixels, 0, "camera", 100, detect)
    # Move beyond the former fixed ROI (50% padding) without redetection.
    for n in range(1, 13):
        moved = np.zeros_like(pixels)
        moved[40+n:76+n, 50+2*n:86+2*n] = texture
        items, source = tracker.update(moved, n, "camera", 100, detect)
        assert source == "kcf"
        x0, y0, x1, y1 = items[0]["bbox"]
        assert abs((x0 + x1) / 2 - (68 + 2*n)) < 5
        assert abs((y0 + y1) / 2 - (58 + n)) < 5
    detect.assert_called_once()


def test_web_settings_accept_interval_and_reject_invalid_values(tmp_path):
    from dinov3_cases.web import create_app

    node = SimpleNamespace(
        root=tmp_path, lock=threading.Lock(), image_changed=threading.Event(),
        stream_changed=threading.Condition(), generation=0,
        grounding_settings=dict(prompt="box.", box_threshold=0.3,
                                recognition_threshold=0.75, margin=0.05,
                                detection_interval=1),
        grounding_detector=SimpleNamespace(validate_prompt=Mock()),
        clear_object_debug=Mock(),
    )
    client = create_app(node).test_client()
    response = client.post("/api/grounding", json={"detection_interval": 5})
    assert response.status_code == 200
    assert response.json["grounding"]["detection_interval"] == 5
    assert node.generation == 1
    for value in (0, -1, True, 1.5, "5"):
        assert client.post("/api/grounding", json={"detection_interval": value}).status_code == 400
    assert node.generation == 1


@pytest.mark.parametrize('padding', [-.1, float('nan'), float('inf'), True, '0.5'])
def test_invalid_roi_padding(padding):
    with pytest.raises(ValueError, match='roi_padding'):
        KCFProposals(roi_padding=padding)


def test_fixed_roi_local_coordinates_and_global_output():
    image = np.full((160, 200, 3), (10, 20, 30), dtype=np.uint8)
    fake = SimpleNamespace(init=Mock(return_value=None),
                           update=Mock(return_value=(True, (12, 11, 20, 20))))
    tracker = KCFProposals(lambda: fake)
    detector = Mock(return_value=[dict(proposal(), bbox=[80, 60, 100, 80])])
    tracker.update(image, 0, 'camera', 10, detector)
    assert tracker.rois == [(70, 50, 110, 90)]
    assert fake.init.call_args.args[0].shape == (40, 40, 3)
    assert fake.init.call_args.args[1] == (10, 10, 20, 20)
    np.testing.assert_array_equal(fake.init.call_args.args[0][0, 0], [30, 20, 10])
    items, source = tracker.update(image, 1, 'camera', 10, detector)
    assert source == 'kcf' and items[0]['bbox'] == [82, 61, 102, 81]
    assert fake.update.call_args.args[0].shape == (40, 40, 3)
    assert tracker.rois == [(70, 50, 110, 90)]
    assert not tracker.refresh_requested
    detector.assert_called_once()


def test_fixed_roi_near_edge_keeps_box_and_requests_refresh():
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    fake = SimpleNamespace(init=Mock(return_value=None),
                           update=Mock(return_value=(True, (19, 10, 20, 20))))
    tracker = KCFProposals(lambda: fake)
    detector = Mock(return_value=[dict(proposal(), bbox=[80, 60, 100, 80])])
    tracker.update(image, 0, 'camera', 10, detector)
    items, source = tracker.update(image, 1, 'camera', 10, detector)
    assert source == 'kcf' and items[0]['bbox'] == [89, 60, 109, 80]
    assert tracker.refresh_requested
    detector.assert_called_once()


def test_fixed_roi_escape_redetects_instead_of_returning_truncated_box():
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    fake = SimpleNamespace(init=Mock(return_value=None),
                           update=Mock(return_value=(True, (22, 10, 20, 20))))
    tracker = KCFProposals(lambda: fake)
    detector = Mock(return_value=[dict(proposal(), bbox=[80, 60, 100, 80])])
    tracker.update(image, 0, 'camera', 10, detector)
    assert tracker.update(image, 1, 'camera', 10, detector)[1] == 'sam3'
    assert detector.call_count == 2


def test_fixed_roi_physical_image_edge_clips_without_spurious_refresh():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    fake = SimpleNamespace(init=Mock(return_value=None),
                           update=Mock(return_value=(True, (-2, 10, 20, 20))))
    tracker = KCFProposals(lambda: fake)
    detector = Mock(return_value=[dict(proposal(), bbox=[0, 20, 20, 40])])
    tracker.update(image, 0, 'camera', 10, detector)
    assert tracker.rois == [(0, 10, 30, 50)]
    items, source = tracker.update(image, 1, 'camera', 10, detector)
    assert source == 'kcf' and items[0]['bbox'] == [0, 20, 18, 40]
    assert not tracker.refresh_requested


def test_real_opencv_kcf_tracks_motion_in_fixed_roi():
    texture = np.random.default_rng(123).integers(0, 256, (36, 36, 3), dtype=np.uint8)
    image = np.zeros((140, 180, 3), dtype=np.uint8)
    image[40:76, 50:86] = texture
    tracker = KCFProposals()
    detector = Mock(return_value=[dict(proposal(), bbox=[50, 40, 86, 76])])
    tracker.update(image, 0, 'camera', 100, detector)
    for n in range(1, 4):
        moved = np.zeros_like(image)
        moved[40+n:76+n, 50+2*n:86+2*n] = texture
        items, source = tracker.update(moved, n, 'camera', 100, detector)
        assert source == 'kcf'
        x0, y0, x1, y1 = items[0]['bbox']
        assert abs((x0+x1)/2-(68+2*n)) < 5
        assert abs((y0+y1)/2-(58+n)) < 5
    detector.assert_called_once()
