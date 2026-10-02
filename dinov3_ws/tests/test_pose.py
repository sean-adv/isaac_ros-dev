from types import SimpleNamespace as NS
import cv2
import numpy as np
import pytest
from dinov3_cases.pose import closest_depth, estimate_pose, draw_axes

K = np.array([[400., 0, 100], [0, 400., 60], [0, 0, 1]])
BOX = [20, 40, 180, 80]


def surface():
    mask = np.zeros((120, 200), np.uint8)
    mask[40:80, 20:180] = 1
    return np.ones(mask.shape, np.float32), mask


def test_surface_pose_units_quaternion_and_overlay():
    depth, mask = surface()
    a = estimate_pose(depth, '32FC1', K, [], BOX, mask)
    b = estimate_pose((depth * 1000).astype(np.uint16), '16UC1', K, [], BOX, mask)
    np.testing.assert_allclose(a['position'], [-.00125, -.00125, 1], atol=1e-6)
    np.testing.assert_allclose(a['position'], b['position'])
    r = np.asarray(a['axes'])
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-8)
    assert np.linalg.det(r) == pytest.approx(1)
    assert r[2, 2] < -.99
    q = np.array(a['quaternion_xyzw'])
    assert np.linalg.norm(q) == pytest.approx(1)
    angle = 2 * np.arctan2(np.linalg.norm(q[:3]), q[3])
    np.testing.assert_allclose(cv2.Rodrigues(q[:3] / np.linalg.norm(q[:3]) * angle)[0], r, atol=1e-8)
    image = np.zeros((120, 200, 3), np.uint8)
    draw_axes(image, a, K, [], .05)
    assert image.any()


def test_tilted_plane_and_background_rejection():
    depth, mask = surface()
    v, u = np.indices(depth.shape)
    depth[:] = 1 / (1 - .15 * (u - 100) / 400)
    depth[mask == 0] = 3
    depth[45:47, 30:40] = np.nan
    depth[48:50, 30:40] = 0
    depth[51:53, 30:40] = 5
    pose = estimate_pose(depth, '32FC1', K, [], BOX, mask)
    normal = np.array([.15, 0, -1.])
    normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(np.asarray(pose['axes'])[:, 2], normal, atol=1e-5)


def test_invalid_and_ambiguous_depth():
    depth, mask = surface()
    for invalid in (0, np.nan, np.inf):
        with pytest.raises(ValueError, match='Too few'):
            estimate_pose(np.full_like(depth, invalid), '32FC1', K, [], BOX, mask)
    with pytest.raises(ValueError, match='dimensions'):
        estimate_pose(depth[:10], '32FC1', K, [], BOX, mask)
    with pytest.raises(ValueError, match='encoding'):
        estimate_pose(depth, '8UC1', K, [], BOX, mask)
    mask[:] = 0
    mask[40:80, 80:120] = 1
    with pytest.raises(ValueError, match='symmetric'):
        estimate_pose(depth, '32FC1', K, [], BOX, mask)


def header(seconds, frame='optical'):
    return NS(frame_id=frame, stamp=NS(sec=int(seconds), nanosec=round((seconds % 1) * 1e9)))


def test_depth_pairing():
    frames = [NS(header=header(t)) for t in (10., 10.03, 10.2)]
    assert closest_depth(frames, header(10.04), .05) is frames[1]
    for target in (header(11), header(10, 'unaligned'), header(0)):
        with pytest.raises(ValueError):
            closest_depth(frames, target, .05)
