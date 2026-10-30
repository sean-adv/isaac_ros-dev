"""Camera-region validation, full-box containment and live setting updates."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sensor_msgs.msg import Image

from dinov3_cases.sam3_roi import DEFAULT_ROI, filter_roi, roi_bounds, validate_roi
from dinov3_cases.web import create_app


@pytest.mark.parametrize('region,expected', [
    (DEFAULT_ROI, (0, 0, 1920, 1080)),
    ((640, 480, -1, -1), (640, 300, 1280, 780)),
    ((640, 480, 600, 400), (280, 160, 920, 640)),
    ((640, 480, 0, 0), (0, 0, 320, 240)),
    ((0, 480, 0, 400), (0, 160, 1920, 640)),
    ((640, 0, 600, 0), (280, 0, 920, 1080)),
    ((4000, 3000, -1, -1), (0, 0, 1920, 1080)),
    ((100, 100, 3000, 2000), (1920, 1080, 1920, 1080)),
])
def test_native_region_bounds(region, expected):
    assert roi_bounds(1920, 1080, validate_roi(*region)) == expected


def test_entire_box_must_be_inside_and_coordinates_are_unchanged():
    bounds = (100, 100, 300, 300)
    inside = dict(bbox=[100, 100, 300, 300], label='medicine')
    boxes = [inside] + [dict(bbox=box) for box in (
        [99, 150, 200, 250], [150, 99, 200, 250], [150, 150, 301, 250],
        [150, 150, 200, 301], [150, 150, 150, 250], [400, 400, 500, 500],
    )]
    result = filter_roi(boxes, bounds)
    assert result == [inside]
    assert result[0] is inside
    assert inside['bbox'] == [100, 100, 300, 300]
    assert filter_roi(boxes, None) is boxes
    assert filter_roi(boxes, (100, 100, 100, 100)) == []


@pytest.fixture
def api(tmp_path):
    node = SimpleNamespace(root=tmp_path, lock=threading.Lock(), generation=7,
                           sam3_roi=DEFAULT_ROI, camera_frame=Image(width=1920, height=1080),
                           image_changed=threading.Event(), render_changed=threading.Event(),
                           video_hub=Mock(), clear_object_debug=Mock())
    return create_app(node).test_client(), node


def test_live_update_is_atomic_and_clears_previous_results(api):
    client, node = api
    assert client.get('/api/sam3-roi').json['enabled'] is False
    result = client.post('/api/sam3-roi', json=dict(width=640, height=480, center_x=600, center_y=400))
    assert result.status_code == 200
    assert result.json['bounds'] == [280, 160, 920, 640]
    assert result.json['native_size'] == [1920, 1080]
    assert node.sam3_roi == (640, 480, 600, 400)
    assert node.generation == 8
    assert node.detections == []
    assert node.image_changed.is_set() and node.render_changed.is_set()
    client.post('/api/sam3-roi', json={'center_x': 600})
    assert node.generation == 8  # Reapplying identical settings keeps valid results.
    assert client.post('/api/sam3-roi', json={'width': 0, 'height': 0}).json['enabled'] is False
    assert node.sam3_roi == (0, 0, 600, 400)


@pytest.mark.parametrize('payload', [None, [], {}, {'width': -1}, {'height': 2.5},
                                    {'width': True}, {'center_x': -2}, {'center_y': False},
                                    {'center_y': '10'}, {'preview': True}])
def test_bad_values_do_not_change_live_configuration(api, payload):
    client, node = api
    assert client.post('/api/sam3-roi', json=payload).status_code == 400
    assert node.sam3_roi == DEFAULT_ROI
    assert node.generation == 7


def test_settings_available_before_camera_connects(api):
    client, node = api
    node.camera_frame = None
    result = client.post('/api/sam3-roi', json={'width': 300})
    assert result.status_code == 200
    assert result.json['native_size'] is None
    assert result.json['bounds'] is None


def test_region_disclosure_is_closed_and_preserves_live_video(api):
    client, node = api
    node.bbox_visualization_mode = '2d'
    html = client.get('/').get_data(as_text=True)
    assert '<details class="help" id="roi-panel">' in html
    assert 'id="roi-preview-enabled"' in html
    assert 'name="center_x"' in html and 'name="center_y"' in html
    assert '/static/sam3_roi.js' in html
    assert 'data-live-video' in html
