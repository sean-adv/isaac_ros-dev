"""Final cuboids match detector pixels while retaining orthogonal local axes."""
from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dinov3_cases.depth_bbox import bbox_vertices, fit_bbox3d_to_2d, project_box_points


def calibration(model="plumb_bob"):
    return SimpleNamespace(width=640, height=480,
                           k=[600., 0., 320., 0., 600., 240., 0., 0., 1.],
                           d=([.03, -.01, .001, .002] if model == "equidistant"
                              else [.1, -.03, .001, .002, 0.]),
                           distortion_model=model)


def projected_bounds(data, info):
    points = project_box_points(bbox_vertices(data), info)
    return np.r_[points.min(axis=0), points.max(axis=0)]


@pytest.mark.parametrize("model", ["plumb_bob", "rational_polynomial", "equidistant"])
@pytest.mark.parametrize("vector", [[0., 0., 0.], [.2, .4, .3], [-.3, .1, -.8]])
def test_recovers_feasible_four_edges_and_preserves_rotation(model, vector):
    info = calibration(model)
    rotation, _ = cv2.Rodrigues(np.array(vector))
    desired = dict(position=[.05, -.03, 1.2], size=[.45, .25, .12],
                   rotation=rotation.tolist())
    target = projected_bounds(desired, info)
    target[2:] += 1.  # Detector boxes have exclusive maximum coordinates.
    measured = dict(position=[.01, -.01, 1.25], size=[.30, .15, .03],
                    rotation=rotation.tolist(), point_count=300)
    original = deepcopy(measured)
    fitted = fit_bbox3d_to_2d(measured, info, target)
    assert fitted["fit_status"] == "matched"
    assert fitted["fit_error_px"] <= .25
    np.testing.assert_allclose(projected_bounds(fitted, info), target - [0, 0, 1, 1], atol=.25)
    assert fitted["rotation"] == measured["rotation"]
    assert measured == original
    assert fitted["measured_position"] == measured["position"]
    assert fitted["measured_size"] == measured["size"]
    assert fitted["point_count"] == 300
    vertices = bbox_vertices(fitted)
    edges = np.array([vertices[1] - vertices[0], vertices[3] - vertices[0],
                      vertices[4] - vertices[0]])
    np.testing.assert_allclose(edges @ edges.T, np.diag(np.square(fitted["size"])), atol=1e-12)
    assert np.min(vertices[:, 2]) > 0.


def test_planar_measurement_fits_inclusive_pixel_bounds_and_clips_target():
    info = calibration()
    measured = dict(position=[0., 0., 1.], size=[.3, .2, 0.], rotation=np.eye(3).tolist())
    fitted = fit_bbox3d_to_2d(measured, info, [-10, 120, 650, 360])
    assert fitted["fit_status"] == "matched"
    np.testing.assert_allclose(projected_bounds(fitted, info), [0, 120, 639, 359], atol=.25)
    assert min(fitted["size"]) > 0.


def test_incompatible_aspect_ratio_reports_approximation_without_changing_axes():
    info = calibration()
    info.d = [0.] * 5
    rotation, _ = cv2.Rodrigues(np.array([0., 0., np.pi / 4]))
    measured = dict(position=[0., 0., 1.], size=[.2, .1, .01], rotation=rotation.tolist())
    fitted = fit_bbox3d_to_2d(measured, info, [200, 220, 441, 261])
    assert fitted["fit_status"] == "approximate"
    assert fitted["fit_error_px"] > .25
    assert fitted["rotation"] == measured["rotation"]
    assert np.min(bbox_vertices(fitted)[:, 2]) > 0.


def test_solver_failure_retains_original_measurement(monkeypatch):
    from dinov3_cases import depth_bbox
    measured = dict(position=[0., 0., 1.], size=[.3, .2, .01])
    def fail(*args, **kwargs):
        raise ValueError("solver failed")
    monkeypatch.setattr(depth_bbox, "least_squares", fail)
    fitted = fit_bbox3d_to_2d(measured, calibration(), [200, 120, 440, 360])
    assert fitted["fit_status"] == "retained_measurement"
    assert fitted["position"] == measured["position"]
    assert fitted["size"] == measured["size"]


@pytest.mark.parametrize("target", [[20, 20, 20, 30], [20, 20, 21, 30]])
def test_empty_pixel_targets_do_not_create_invalid_cuboids(target):
    measured = dict(position=[0., 0., 1.], size=[.3, .2, .01])
    fitted = fit_bbox3d_to_2d(measured, calibration(), target)
    assert fitted["fit_status"] == "invalid_target_or_depth"
    assert fitted["size"] == measured["size"]
