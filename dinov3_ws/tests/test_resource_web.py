"""Resource endpoint and page integration without loading inference models."""
import threading
from types import SimpleNamespace

import pytest

from dinov3_cases.resources import ResourceMonitor
from dinov3_cases.web import create_app


@pytest.mark.parametrize('enabled', [False, True])
def test_resource_endpoint_and_page(tmp_path, monkeypatch, enabled):
    monkeypatch.setattr('dinov3_cases.resources.shutil.which', lambda _: None)
    monitor = ResourceMonitor(.5) if enabled else None
    if monitor is not None:
        monitor.latest = monitor.sampler.sample()
    node = SimpleNamespace(root=tmp_path, lock=threading.Lock(), resource_monitor=monitor,
                           performance={'frame_id': 12}, bbox_visualization_mode='3d')
    try:
        client = create_app(node).test_client()
        response = client.get('/api/resources')
        assert response.status_code == 200
        assert response.headers['Cache-Control'] == 'no-store'
        data = response.get_json()
        assert data['enabled'] is enabled
        assert data['performance'] == {'frame_id': 12}
        assert (data['sample'] is not None) is enabled
        page = client.get('/')
        assert page.status_code == 200
        assert b'id="resource-rows"' in page.data
        assert client.get('/static/resources.js').status_code == 200
    finally:
        if monitor is not None:
            monitor.close()
