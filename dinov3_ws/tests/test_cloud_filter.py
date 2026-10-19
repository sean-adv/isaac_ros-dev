"""Coverage and temporal rejection must happen before any RANSAC call."""

from unittest.mock import Mock

import numpy as np
import pytest

from dinov3_cases.cloud_filter import CloudFilter
from dinov3_cases import depth_bbox
from test_depth_bbox import calibration, scene, BOX


def test_sparse_components_never_become_accepted_by_repetition():
    guard, state = CloudFilter(), {}
    guard.check(.65, .475, state, 0)
    for frame in range(1, 8):
        with pytest.raises(ValueError, match="insufficient_cloud_coverage"):
            guard.check(.04, .62, state, frame)
    assert state["accepted"] == (.65, .475)
    assert "pending" not in state


@pytest.mark.parametrize("coverage, depth", [(.20, .475), (.65, .625)])
def test_changed_cloud_requires_three_distinct_depth_frames(coverage, depth):
    guard, state = CloudFilter(), {}
    guard.check(.65, .475, state, 0)
    for frame in (1, 2):
        with pytest.raises(ValueError, match=f"cloud_change_pending:{frame}/3"):
            guard.check(coverage, depth, state, frame)
        # Reprocessing the same depth message cannot advance confirmation.
        with pytest.raises(ValueError, match=f"cloud_change_pending:{frame}/3"):
            guard.check(coverage, depth, state, frame)
        assert state["accepted"] == (.65, .475)
    guard.check(coverage, depth, state, 3)
    assert state["accepted"] == (coverage, depth)


def test_recovery_and_inconsistent_candidates_restart_confirmation():
    guard, state = CloudFilter(), {}
    guard.check(.65, .475, state, 0)
    for frame, depth in ((1, .625), (2, .80), (3, .625)):
        with pytest.raises(ValueError, match="cloud_change_pending:1/3"):
            guard.check(.65, depth, state, frame)
    guard.check(.64, .476, state, 4)
    assert "pending" not in state
    with pytest.raises(ValueError, match="cloud_change_pending:1/3"):
        guard.check(.65, .625, state, 5)


def test_quantile_trims_depth_outliers_before_ransac(monkeypatch):
    points_seen = []
    original = depth_bbox._surface_rotation

    def fit(points, *args):
        points_seen.append(points.copy())
        return original(points, *args)

    monkeypatch.setattr(depth_bbox, "_surface_rotation", fit)
    depth = scene()
    depth[60, 80] = .95
    result = depth_bbox.estimate_bbox3d(depth, calibration(), BOX, cloud_filter=CloudFilter())
    assert result["nearest_depth"] == .95  # diagnostics retain the original sample
    assert result["plane_input_count"] == 3599
    assert len(points_seen) == 1
    np.testing.assert_array_equal(points_seen[0][:, 2], 1.)


def test_rejected_region_never_reaches_ransac_or_camera_aligned_fallback(monkeypatch):
    fit = Mock(side_effect=AssertionError("rejected region reached RANSAC"))
    monkeypatch.setattr(depth_bbox, "_surface_rotation", fit)
    depth = np.zeros((120, 160))
    depth[55:65, 75:85] = .62
    items = [dict(bbox=BOX, bbox3d={"old": True})]
    depth_bbox.attach_bboxes3d(items, depth, calibration(), cloud_filter=CloudFilter())
    assert items[0]["bbox3d_status"] == "insufficient_cloud_coverage"
    assert "bbox3d" not in items[0]
    fit.assert_not_called()


def test_depth_jump_is_rejected_before_ransac(monkeypatch):
    state = {"accepted": (.55, .475)}
    fit = Mock(side_effect=AssertionError("unconfirmed depth reached RANSAC"))
    monkeypatch.setattr(depth_bbox, "_surface_rotation", fit)
    with pytest.raises(ValueError, match="cloud_change_pending:1/3"):
        depth_bbox.estimate_bbox3d(scene(), calibration(), BOX,
                                  cloud_filter=CloudFilter(), cloud_state=state, cloud_frame=1)
    fit.assert_not_called()


@pytest.mark.parametrize("options", [
    {"min_coverage": -.1}, {"min_coverage": np.nan}, {"min_coverage": 1.1},
    {"min_coverage_ratio": -.1}, {"min_coverage_ratio": np.inf},
    {"max_depth_jump": 0.}, {"max_depth_jump": np.nan},
    {"confirmations": 0}, {"confirmations": True}, {"confirmations": 2.5},
])
def test_invalid_filter_configuration(options):
    with pytest.raises(ValueError):
        CloudFilter(**options)
