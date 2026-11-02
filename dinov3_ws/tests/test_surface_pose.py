"""Observed faces must constrain a box jointly, without averaging across ridges."""
from itertools import permutations

import cv2
import numpy as np
import pytest

from dinov3_cases.depth_bbox import _surface_rotation
from dinov3_cases.surface_pose import (extract_planes, choose_faces, orthogonal_rotation,
                                      support_mask, _plane)


def cuboid(count=3, noise=.0003):
    half = np.array([.09, .14, .06])
    rotation, _ = cv2.Rodrigues(np.array([.15, -.45, .12]))
    points = []
    rng = np.random.default_rng(17)
    for axis in [2, 0, 1][:count]:
        others = [i for i in range(3) if i != axis]
        a, b = np.meshgrid(np.linspace(-half[others[0]], half[others[0]], 35),
                           np.linspace(-half[others[1]], half[others[1]], 35))
        face = np.empty((a.size, 3))
        face[:, axis] = -half[axis]
        face[:, others[0]], face[:, others[1]] = a.ravel(), b.ravel()
        face += rng.normal(0., noise, face.shape)
        points.append(face @ rotation.T + [0., 0., 1.])
    return np.vstack(points), rotation


def assert_axes(actual, expected, degrees=1.):
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-12)
    assert np.linalg.det(actual) == pytest.approx(1.)
    dots = abs(actual.T @ expected)
    assert len(set(dots.argmax(axis=1))) == 3
    assert np.all(dots.max(axis=1) > np.cos(np.deg2rad(degrees)))


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("quantile", [0., .02, .05])
def test_two_and_three_noisy_faces_recover_box_axes(count, quantile):
    points, expected = cuboid(count)
    result = _surface_rotation(points, quantile, 80)
    rotation, _, _, status, meta, faces = result
    assert len(faces) == count
    assert meta["face_count"] == count
    assert meta["surface_status"] == "orthogonal_faces"
    assert meta["orientation_confidence"] == ("high" if count == 3 else "medium")
    assert status == "ok"
    assert_axes(rotation, expected)
    assert support_mask(points, faces).mean() > .9


def test_single_face_is_explicit_low_confidence():
    points, expected = cuboid(1, noise=.0001)
    rotation, _, _, _, meta, faces = _surface_rotation(points, .05, 80)
    assert meta["face_count"] == len(faces) == 1
    assert meta["surface_status"] == "single_plane"
    assert meta["orientation_confidence"] == "low"
    assert_axes(rotation, expected)


def test_tail_outliers_and_depth_holes_do_not_merge_three_faces():
    points, expected = cuboid(3)
    rng = np.random.default_rng(9)
    points = points[rng.random(len(points)) > .15]
    outliers = points[:30] + rng.normal(0., .03, (30, 3))
    points = np.vstack((points, outliers))
    rotation, _, _, _, meta, faces = _surface_rotation(points, .05, 80)
    assert meta["face_count"] == 3
    assert_axes(rotation, expected, degrees=1.5)
    assert not support_mask(points, faces)[-30:].all()


def test_plane_fragment_merge_preserves_parallel_offset_surfaces():
    x, y = np.meshgrid(np.linspace(-.08, .08, 70), np.linspace(-.1, .1, 70))
    flat = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    fragmented = flat[abs(flat[:, 0]) > .003]
    planes = extract_planes(fragmented, .05)
    assert len(planes) == 1
    offset = flat.copy()
    offset[:, 2] += .05
    planes = extract_planes(np.vstack((flat, offset)), .05)
    assert len(planes) == 2
    assert choose_faces(planes) == []


def test_nonorthogonal_wedge_is_not_forced_into_a_cuboid():
    x, y = np.meshgrid(np.linspace(0., .1, 40), np.linspace(-.1, .1, 50))
    a = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    r, _ = cv2.Rodrigues(np.array([0., np.pi / 3, 0.]))
    b = (a - [0., 0., 1.]) @ r.T + [0., 0., 1.]
    points = np.vstack((a, b))
    planes = extract_planes(points, .05)
    assert len(planes) >= 2
    assert choose_faces(planes) == []
    _, _, _, _, meta, _ = _surface_rotation(points, .05, 80)
    assert meta["face_count"] == 1
    assert meta["surface_status"] == "incompatible_faces"
    assert meta["orientation_confidence"] == "low"


def test_distant_perpendicular_background_is_not_a_neighbor_face():
    x, y = np.meshgrid(np.linspace(-.1, .1, 30), np.linspace(-.1, .1, 30))
    a = _plane(np.column_stack((x.ravel(), y.ravel(), np.ones(x.size))), .05)
    b = _plane(np.column_stack((np.ones(x.size), x.ravel(), y.ravel()+1)), .05)
    for face in (a, b):
        face['weight'] = face['count']
    assert abs(a['normal'] @ b['normal']) < 1e-6
    assert choose_faces([a, b]) == []


def test_face_order_does_not_change_canonical_rotation():
    points, _ = cuboid(3)
    faces = choose_faces(extract_planes(points, .05))
    assert len(faces) == 3
    reference = orthogonal_rotation(faces, points)
    for ordered in permutations(faces):
        np.testing.assert_allclose(orthogonal_rotation(ordered, points), reference, atol=1e-12)


def test_registered_depth_pipeline_recovers_visible_cuboid_faces():
    from types import SimpleNamespace
    from dinov3_cases.depth_bbox import estimate_bbox3d

    # Render ray intersections with all six faces, retaining only the nearest
    # surface. This exercises depth segmentation, local normals and bounds.
    height, width, focal = 240, 320, 450.
    v, u = np.indices((height, width))
    rays = np.stack(((u - width / 2) / focal, (v - height / 2) / focal,
                     np.ones_like(u)), axis=-1)
    expected, _ = cv2.Rodrigues(np.array([-.4, -.6, .1]))
    center = np.array([0., 0., .8])
    half = np.array([.07, .10, .05])
    local_origin = -center @ expected
    local_rays = rays @ expected
    depth = np.full((height, width), 2.)
    for axis in range(3):
        for side in (-1, 1):
            distance = (side * half[axis] - local_origin[axis]) / local_rays[..., axis]
            local = local_origin + local_rays * distance[..., None]
            valid = np.all(abs(local) <= half + 1e-10, axis=-1) & (distance > 0)
            depth[valid] = np.minimum(depth[valid], distance[valid])
    mask = depth < 1.
    ys, xs = np.nonzero(mask)
    bbox = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    info = SimpleNamespace(width=width, height=height,
                           k=[focal, 0., width / 2, 0., focal, height / 2, 0., 0., 1.],
                           d=[0.] * 5, distortion_model="plumb_bob", binning_x=0, binning_y=0,
                           roi=SimpleNamespace(x_offset=0, y_offset=0))
    result = estimate_bbox3d(depth, info, bbox, depth_band=.3, quantile=.05)
    assert result["method"] == "central_depth_component_multiplane"
    assert result["face_count"] >= 2
    assert_axes(np.asarray(result["rotation"]), expected)
    np.testing.assert_allclose(result["position"], center, atol=.012)
    np.testing.assert_allclose(sorted(result["size"]), sorted(2 * half), atol=.02)
