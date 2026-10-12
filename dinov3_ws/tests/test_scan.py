"""Frozen RGB-D sessions and fail-closed grasp result validation."""
from copy import deepcopy
from types import SimpleNamespace
import threading
import time

from flask import Flask
import pytest

from dinov3_cases.scan import new_session, planner_update, register_scan


def object_item():
    return dict(label="medicine", bbox=[0, 0, 20, 40], pose_status="ok",
                pose=dict(position=[0, 0, 1], quaternion=[0, 0, 0, 1]),
                scan=dict(points=[[0, 0, 1], [.01, 0, 1]], point_count=80,
                          stage="candidates", frame_id="camera_optical", stamp=10.))


def ready_payload(session):
    return dict(id=session["id"], frame_id="camera_optical", state="ready", confidence=.9,
                checks={key:True for key in ("antipodal", "contact", "clearance",
                                            "collision_free", "unknown_space_clear")},
                grasp_pose=dict(position=[.01, 0, 1], quaternion=[0, 0, 0, 1], width=.04))


def test_surface_pose_does_not_become_a_grasp_and_snapshot_is_independent():
    item = object_item()
    session = new_session(item, 12)
    item["scan"]["points"].clear()
    assert session["points"]
    assert session["grasp_pose"] is None
    assert session["state"] == "candidates"
    assert session["confidence"] is None


@pytest.mark.parametrize("change", [
    {"confidence":.7}, {"confidence":float("nan")}, {"confidence":True},
    {"frame_id":"robot_base"}, {"checks":{}}, {"grasp_pose":None},
    {"grasp_pose":dict(position=[0, 0, 1], quaternion=[0, 0, 0, 0], width=.04)},
    {"grasp_pose":dict(position=[0, 0, float("inf")], quaternion=[0, 0, 0, 1], width=.04)},
    {"grasp_pose":dict(position=[0, 0, 1], quaternion=[0, 0, 0, 1], width=-.04)},
])
def test_unverified_or_invalid_grasp_is_rejected_without_mutation(change):
    session = new_session(object_item(), 1)
    original = deepcopy(session)
    with pytest.raises(ValueError):
        planner_update(session, {**ready_payload(session), **change})
    assert session == original


def test_fusion_loop_and_verified_pose():
    session = new_session(object_item(), 1)
    for state in ("rescan", "moving", "fusion", "geometry", "candidates", "scoring", "collision", "confidence"):
        session = planner_update(session, dict(state=state, frame_id="camera_optical", view_count=2))
    final = planner_update(session, ready_payload(session))
    assert final["grasp_pose"]["frame_id"] == "camera_optical"
    assert final["view_count"] == 2
    assert final["completed"][-1] == "ready"


def test_http_session_revision_and_result_identity():
    app = Flask(__name__)
    node = SimpleNamespace(lock=threading.RLock(), detections=[object_item()],
                           object_debug_revision=3, overlay_at=time.monotonic())
    register_scan(app, node)
    client = app.test_client()
    assert client.get('/api/scan').json == dict(session=None)
    assert client.post('/api/scan', json=dict(index=0, revision=2)).status_code == 409
    assert client.post('/api/scan', json=dict(index=0, revision=3)).status_code == 200
    session = client.get('/api/scan').json['session']
    assert client.post('/api/scan/result', json={**ready_payload(session), 'id':'old'}).status_code == 409
    assert client.post('/api/scan/result', json=ready_payload(session)).status_code == 200
    assert client.get('/api/scan').json['session']['state'] == 'ready'
    assert client.post('/api/scan/result', json=ready_payload(session)).status_code == 409
    client.delete('/api/scan')
    assert client.get('/api/scan').json['session'] is None
    assert client.post('/api/scan/result', json=ready_payload(session)).status_code == 409
    node.overlay_at = time.monotonic() - 4
    assert client.post('/api/scan', json=dict(index=0, revision=3)).status_code == 409


def test_rescan_discards_previous_confidence_and_candidate_progress():
    session = new_session(object_item(), 1)
    session.update(confidence=.95, completed=["scoring", "collision"],
                   candidates=[dict(position=[0,0,1], quaternion=[0,0,0,1])])
    updated = planner_update(session, dict(state="rescan", frame_id="camera_optical"))
    assert updated["confidence"] is None
    assert updated["candidates"] == []
    assert "collision" not in updated["completed"]
