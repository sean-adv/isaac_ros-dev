"""Startup configuration validation without constructing a ROS node or workers."""

import pytest

from dinov3_cases.config import default_parameters, validate_parameters


def test_default_settings_are_valid_and_prompt_is_normalized():
    values = default_parameters()
    values['grounding_prompt'] = ' box '
    assert validate_parameters(values.__getitem__) == dict(
        prompt='box.', box_threshold=0.3, recognition_threshold=0.75,
        margin=0.05, detection_interval=1,
    )


@pytest.mark.parametrize('overrides,message', [
    ({'bbox_visualization_mode': 'invalid'}, 'bbox_visualization_mode'),
    ({'bbox3d_plane_backend': 'invalid'}, 'bbox3d_plane_backend'),
    ({'opencv_num_threads': -1}, 'opencv_num_threads'),
    ({'grounding_ipc_timeout': float('nan')}, 'grounding_ipc_timeout'),
    ({'pose_min_depth': 6.0}, 'pose_min_depth'),
    ({'bbox3d_interval': 0}, 'bbox3d_interval'),
    ({'video_fps': True}, 'video_fps'),
])
def test_invalid_startup_settings_fail_before_workers(overrides, message):
    values = {**default_parameters(), **overrides}
    with pytest.raises(ValueError, match=message):
        validate_parameters(values.__getitem__)
