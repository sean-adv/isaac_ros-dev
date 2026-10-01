from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from rgbd_object_proposals.geometry import (
    Config, depth_to_pointcloud, foreground_mask, depth_connected_components,
    compute_object_obb, filter_candidate, project_points_to_image, generate_proposals,
)
from dinov3_cases.proposals import crop_bounds
from synthetic_scene import make_scene


def proposals(depth, encoding, intrinsics, cfg):
    return generate_proposals(depth, encoding, intrinsics, cfg)


def test_depth_units_invalid_and_intrinsics():
    cfg = Config(pixel_stride=1)
    mm = np.array([[0, 1000, 500]], dtype=np.uint16)
    actual = depth_to_pointcloud(mm, '16UC1', (100, 100, 1, 0), cfg)
    np.testing.assert_allclose(actual, [[0, 0, 1], [.005, 0, .5]])
    meters = np.array([[np.nan, np.inf, 0, -1, 3, 1]], dtype=np.float32)
    assert len(depth_to_pointcloud(meters, '32FC1', (100, 100, 1, 0), cfg)) == 1
    with pytest.raises(ValueError):
        depth_to_pointcloud(mm, 'mono16', (100, 100, 1, 0), cfg)
    with pytest.raises(ValueError):
        depth_to_pointcloud(mm, '16UC1', (0, 100, 1, 0), cfg)


def test_distorted_projection_roundtrip():
    depth = np.zeros((100, 120), dtype=np.float32)
    depth[20, 30] = 1
    intrinsics = (150, 150, 60, 50)
    distortion = [.1, -.02, .001, .002, 0]
    points = depth_to_pointcloud(depth, '32FC1', intrinsics, Config(pixel_stride=1), distortion)
    box = project_points_to_image(points, intrinsics, depth.shape, distortion=distortion)
    assert box[0] <= 30 < box[2] and box[1] <= 20 < box[3]
    assert box[2] - box[0] <= 3


def test_end_to_end_separated_boxes_and_units():
    _, depth, intrinsics, masks = make_scene()
    candidates, points, mask = proposals(depth, '32FC1', intrinsics, Config())
    assert mask.any() and len(points) > 0
    assert [c.candidate_id for c in candidates] == ['candidate_0', 'candidate_1']
    for c, mask in zip(candidates, masks):
        y, x = np.where(mask)
        assert c.bbox[0] <= x.min() and c.bbox[2] > x.max()
        assert c.bbox[1] <= y.min() and c.bbox[3] > y.max()
        assert np.all(c.dimensions[:-1] >= c.dimensions[1:])
    mm = (depth * 1000).round().astype(np.uint16)
    other, _, _ = proposals(mm, '16UC1', intrinsics, Config())
    assert len(other) == len(candidates)
    np.testing.assert_allclose([c.bbox for c in other], [c.bbox for c in candidates], atol=2)


def test_empty_and_table_only():
    for depth in (np.zeros((60, 80), np.float32), np.full((60, 80), 1.1, np.float32)):
        candidates, _, _ = proposals(depth, '32FC1', (100, 100, 40, 30), Config())
        assert candidates == []


def test_planar_object_bbox_exact_and_stride_independent():
    cfg = Config(background_depth=1.0, min_component_area=4, bbox_padding_ratio=0)
    depth = np.ones((20, 30), dtype=np.float32)
    depth[3:9, 7:16] = .8
    for stride in (1, 2, 7):
        candidates, _, mask = generate_proposals(depth, '32FC1', (100, 100, 15, 10),
                                                 replace(cfg, pixel_stride=stride))
        assert len(candidates) == 1
        assert candidates[0].bbox == (7, 3, 16, 9)
        assert candidates[0].point_count == 54 and mask.sum() == 54


def test_connected_components_area_connectivity_and_touching_objects():
    depth = np.ones((20, 30), dtype=np.float32)
    depth[1:3, 1:3] = .8
    depth[3:5, 3:5] = .7
    depth[10, 10] = .8  # tiny noise
    depth[10:15, 20:25] = .8  # too large
    cfg = Config(background_depth=1.0, min_component_area=4, max_component_area=8,
                 bbox_padding_ratio=0)
    candidates, _, _ = generate_proposals(depth, '32FC1', (100, 100, 15, 10), cfg)
    assert [c.bbox for c in candidates] == [(1, 1, 5, 5)]
    candidates, _, _ = generate_proposals(depth, '32FC1', (100, 100, 15, 10),
                                          replace(cfg, connectivity=4))
    assert [c.bbox for c in candidates] == [(1, 1, 3, 3), (3, 3, 5, 5)]


def test_foreground_invalid_noise_and_farther_depth():
    cfg = Config(background_depth=1.0, foreground_threshold=.02)
    depth = np.array([[np.nan, np.inf, 0, -1, 3, 1.1, .99, .9, .1]])
    np.testing.assert_array_equal(foreground_mask(depth, cfg),
                                  [[False, False, False, False, False, False, False, True, False]])


def test_planar_cloud_and_physical_filter():
    x, y = np.mgrid[:10, :10]
    candidate = compute_object_obb(np.column_stack((x.ravel() * .005, y.ravel() * .004, np.ones(100))))
    assert candidate.dimensions[-1] == 0
    assert not filter_candidate(candidate, Config())
    assert filter_candidate(candidate, Config(min_depth_size=0.0))
    assert np.isfinite(candidate.corners).all()


def test_projection_padding_boundary_and_behind_camera():
    points = np.array([[-1, -1, 1], [1, 1, 1]])
    assert project_points_to_image(points, (100, 100, 50, 40), (80, 100), .1) == (0, 0, 100, 80)
    assert project_points_to_image(np.array([[0, 0, -1]]), (100, 100, 50, 40), (80, 100)) is None
    assert project_points_to_image(np.array([[10, 0, 1]]), (100, 100, 50, 40), (80, 100)) is None


@pytest.mark.parametrize('overrides', [dict(foreground_threshold=0), dict(connectivity=6), dict(background_depth=0),
    dict(min_component_area=10, max_component_area=5),
    dict(roi=[10, 0, 5, 10]), dict(pixel_stride=0), dict(max_depth=.1), dict(foreground_threshold=float('nan'))])
def test_config_validation(overrides):
    with pytest.raises(ValueError):
        replace(Config(), **overrides).validate()


def test_roi_restricts_candidates():
    _, depth, intrinsics, _ = make_scene()
    candidates, _, _ = proposals(depth, '32FC1', intrinsics, Config(roi=[0, 0, 320, 360]))
    assert len(candidates) == 1 and candidates[0].bbox[2] < 320


def test_consumer_crop_bounds():
    box = SimpleNamespace(center=SimpleNamespace(position=SimpleNamespace(x=5., y=5.), theta=0.), size_x=20., size_y=20.)
    assert crop_bounds(box, 10, 10) == (0, 0, 10, 10)
    box.center.theta = .1
    assert crop_bounds(box, 10, 10) is None
    box.center.theta = 0
    box.size_x = float('nan')
    assert crop_bounds(box, 10, 10) is None


def test_bbox_padding_clips_image_and_roi_clips_foreground():
    cfg = Config(background_depth=1.0, min_component_area=1, bbox_padding_ratio=.5,
                 roi=[0, 0, 5, 6])
    depth = np.ones((10, 10), dtype=np.float32)
    depth[:8, :8] = .8
    candidates, _, mask = generate_proposals(depth, '32FC1', (100, 100, 5, 5), cfg)
    assert [c.bbox for c in candidates] == [(0, 0, 8, 9)]
    assert mask.sum() == 30
    assert not mask[6:].any() and not mask[:, 5:].any()


def test_depth_units_and_bad_input():
    cfg = Config(background_depth=1.0, min_component_area=1, bbox_padding_ratio=0)
    mm = np.full((10, 10), 1000, dtype=np.uint16)
    with pytest.raises(ValueError):
        generate_proposals(mm, '16UC1', (0, 100, 5, 5), cfg)
    with pytest.raises(ValueError):
        generate_proposals(mm, '16UC1', (100, 100, 5, 5),
                           replace(cfg, roi=[0, 0, 20, 20]))
    depth = mm.astype(np.float32) * .001
    depth[2:5, 3:6] = .8
    candidates, _, _ = generate_proposals(depth, '32FC1', (100, 100, 5, 5), cfg)
    assert [c.bbox for c in candidates] == [(3, 2, 6, 5)]


@pytest.mark.parametrize('encoding', ['16UC1', '32FC1'])
@pytest.mark.parametrize('stride', [1, 7])
def test_depth_jump_splits_touching_objects_before_area_filter(encoding, stride):
    depth = np.ones((20, 30), dtype=np.float32)
    depth[3:9, 5:10] = .70
    depth[3:9, 10:15] = .80
    cfg = Config(background_depth=1., depth_jump_threshold=.02,
                 min_component_area=20, max_component_area=40,
                 bbox_padding_ratio=0, pixel_stride=stride)
    data = (depth * 1000).round().astype(np.uint16) if encoding == '16UC1' else depth
    candidates, _, mask = generate_proposals(data, encoding, (100, 100, 15, 10), cfg)
    assert [c.bbox for c in candidates] == [(5, 3, 10, 9), (10, 3, 15, 9)]
    assert [c.point_count for c in candidates] == [30, 30]
    assert mask.sum() == 60
    # Binary connectivity merges them, exceeding the component area limit.
    old, _, _ = generate_proposals(data, encoding, (100, 100, 15, 10),
                                   replace(cfg, depth_jump_threshold=0.))
    assert old == []


def test_gradual_slope_remains_connected_and_input_unchanged():
    depth = np.tile(np.linspace(.5, .8, 31, dtype=np.float32), (4, 1))
    original = depth.copy()
    mask = np.ones_like(depth, dtype=bool)
    labels, stats = depth_connected_components(depth, mask, Config(depth_jump_threshold=.02))
    assert len(stats) == 2
    assert np.all(labels == 1)
    assert stats[1].tolist() == [0, 0, 31, 4, 124]
    np.testing.assert_array_equal(depth, original)
    assert mask.all()


@pytest.mark.parametrize('connectivity', [4, 8])
def test_diagonal_depth_continuity(connectivity):
    depth = np.array([[.5, 1.], [1., .51]], dtype=np.float32)
    cfg = Config(background_depth=1., depth_jump_threshold=.02, connectivity=connectivity)
    labels, stats = depth_connected_components(depth, foreground_mask(depth, cfg), cfg)
    assert len(stats) - 1 == (1 if connectivity == 8 else 2)
    assert labels[0, 1] == labels[1, 0] == 0
    depth[1, 1] = .6
    _, stats = depth_connected_components(depth, foreground_mask(depth, cfg), cfg)
    assert len(stats) - 1 == 2


def test_depth_continuity_invalid_depth_and_roi():
    depth = np.full((8, 12), .7, dtype=np.float32)
    depth[:, 5] = [0, np.nan, np.inf, -1, 3, 0, np.nan, np.inf]
    cfg = Config(depth_jump_threshold=.02, min_component_area=1,
                 bbox_padding_ratio=0, roi=[2, 1, 10, 7])
    candidates, _, mask = generate_proposals(depth, '32FC1', (100, 100, 6, 4), cfg)
    assert [c.bbox for c in candidates] == [(2, 1, 5, 7), (6, 1, 10, 7)]
    assert sum(c.point_count for c in candidates) == mask.sum() == 42
    for empty in (np.zeros_like(depth), np.full_like(depth, 1.1)):
        assert generate_proposals(empty, '32FC1', (100, 100, 6, 4), cfg)[0] == []


@pytest.mark.parametrize('value', [-.01, float('nan'), float('inf')])
def test_depth_jump_parameter_validation(value):
    with pytest.raises(ValueError):
        Config(depth_jump_threshold=value).validate()


@pytest.mark.parametrize('connectivity', [4, 8])
def test_depth_components_match_reference_graph(connectivity):
    # Independent graph traversal catches crossing edges and overlapping bboxes.
    rng = np.random.default_rng(12)
    for _ in range(10):
        depth = rng.integers(4, 10, size=(9, 11)).astype(np.float32) / 10
        mask = rng.random(depth.shape) > .2
        cfg = Config(connectivity=connectivity, depth_jump_threshold=.15)
        labels, stats = depth_connected_components(depth, mask, cfg)
        expected = np.zeros_like(labels)
        count = 0
        for y, x in zip(*np.nonzero(mask)):
            if expected[y, x]:
                continue
            count += 1
            expected[y, x] = count
            queue = [(y, x)]
            while queue:
                cy, cx = queue.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if (dy == dx == 0) or (connectivity == 4 and abs(dy) + abs(dx) != 1):
                            continue
                        ny, nx = cy + dy, cx + dx
                        if not (0 <= ny < depth.shape[0] and 0 <= nx < depth.shape[1]):
                            continue
                        if mask[ny, nx] and not expected[ny, nx] and abs(depth[ny, nx] - depth[cy, cx]) <= cfg.depth_jump_threshold:
                            expected[ny, nx] = count
                            queue.append((ny, nx))
        np.testing.assert_array_equal(labels, expected)
        assert len(stats) == count + 1
        for label in range(1, count + 1):
            y, x = np.nonzero(expected == label)
            assert stats[label].tolist() == [x.min(), y.min(), x.max()-x.min()+1, y.max()-y.min()+1, len(x)]
