"""Startup configuration validation without constructing a ROS node or workers."""

import pytest

from dinov3_cases.config import default_parameters, validate_parameters


def test_default_settings_are_valid_and_prompt_is_normalized():
    values = default_parameters()
    values['grounding_prompt'] = ' box '
    assert validate_parameters(values.__getitem__) == dict(
        prompt='box.', box_threshold=0.3, recognition_threshold=0.75,
        margin=0.05, detection_interval=0.0,
    )


@pytest.mark.parametrize('overrides,message', [
    ({'bbox_visualization_mode': 'invalid'}, 'bbox_visualization_mode'),
    ({'bbox3d_plane_backend': 'invalid'}, 'bbox3d_plane_backend'),
    ({'opencv_num_threads': -1}, 'opencv_num_threads'),
    ({'grounding_ipc_timeout': float('nan')}, 'grounding_ipc_timeout'),
    ({'pose_min_depth': 6.0}, 'pose_min_depth'),
    ({'bbox3d_interval': -1}, 'bbox3d_interval'),
    ({'video_fps': True}, 'video_fps'),
    *[({'pose_sync_tolerance': value}, 'pose_sync_tolerance')
      for value in (True, -1, float('nan'), float('inf'))],
    *[({'bbox3d_max_age': value}, 'bbox3d_max_age')
      for value in (True, 0, -1, float('nan'), float('inf'))],
])
def test_invalid_startup_settings_fail_before_workers(overrides, message):
    values = {**default_parameters(), **overrides}
    with pytest.raises(ValueError, match=message):
        validate_parameters(values.__getitem__)


def test_zero_seconds_sync_tolerance_is_valid():
    values = {**default_parameters(), 'pose_sync_tolerance': 0}
    validate_parameters(values.__getitem__)


@pytest.mark.parametrize("fps,seconds", [(30, 1.0), (15, 1.0), (30, .1)])
def test_bbox3d_seconds_lifetime_is_independent_of_fps(fps, seconds):
    from unittest.mock import patch
    values = {**default_parameters(), 'video_fps': fps, 'bbox3d_max_age': seconds}
    with patch('dinov3_cases.config.validate_bbox3d_schedule') as validate:
        validate_parameters(values.__getitem__)
    validate.assert_called_once_with(values['bbox3d_interval'], seconds)


@pytest.mark.parametrize('interval', [0.0, .1, 1.0])
def test_nvdcf_is_required_at_every_detection_interval(interval):
    from unittest.mock import patch
    values = {**default_parameters(), 'grounding_detection_interval': interval}
    with patch('dinov3_cases.config.check_nvdcf') as check:
        validate_parameters(values.__getitem__)
    check.assert_called_once_with(values['nvdcf_config'])
    assert 'nvdcf_max_missed_detections' not in values
    assert 'nvdcf_max_missed_frames' not in values
    assert 'bbox_smoothing_frames' not in values


@pytest.mark.parametrize('name', ['bbox3d_interval', 'grounding_detection_interval',
                                  'bbox_smoothing_seconds', 'nvdcf_max_missed_seconds'])
@pytest.mark.parametrize('value', [True, -1, float('nan'), float('inf')])
def test_invalid_seconds_parameters(name, value):
    values = {**default_parameters(), name: value}
    with pytest.raises(ValueError):
        validate_parameters(values.__getitem__)
