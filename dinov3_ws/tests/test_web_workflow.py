"""Exercise the current pages and gallery/source controls without models or a camera."""

import io
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from dinov3_cases.web import create_app


def picture(color="red"):
    stream = io.BytesIO()
    Image.new("RGBA", (20, 12), color).save(stream, format="PNG")
    stream.seek(0)
    return stream


@pytest.fixture
def service(tmp_path):
    node = SimpleNamespace(
        root=tmp_path, lock=threading.Lock(), reference_stride=16,
        bbox_visualization_mode="3d", generation=0, selected="", source="camera",
        frame=None, frame_seq=0, detections=[], video_at=None,
        image_stream=Mock(), clear_object_debug=Mock(),
        image_changed=threading.Event(), render_changed=threading.Event(),
    )
    app = create_app(node)
    app.config["TESTING"] = True
    return app.test_client(), node


def test_current_pages_upload_reference_grid_and_source_switch(service):
    client, node = service
    home = client.get("/")
    manager = client.get("/images")
    assert home.status_code == manager.status_code == 200
    assert b'href="/images"' in home.data
    assert b'href="/"' in manager.data
    assert b'/image_annotated.mjpg' in home.data
    assert b'/image_annotated.mjpg' in manager.data

    response = client.post("/api/cases", data={
        "case": "medicine", "images": (picture(), "reference.png"),
    })
    assert response.json == {"ok": True, "count": 1}
    assert node.selected == "medicine" and node.generation == 1
    node.image_stream.reset.assert_called_once()
    assert node.image_changed.is_set() and node.render_changed.is_set()
    listing = client.get("/api/images?case=medicine").json
    assert listing["cases"] == ["medicine"]
    name = listing["images"][0]["name"]
    with Image.open(node.root / "medicine" / name) as saved:
        assert saved.mode == "RGB"  # Homepage upload contract.
    assert client.get("/api/references?case=medicine").json["images"][0]["name"] == name
    assert client.get(f"/api/references/grid/medicine/{name}").mimetype == "image/png"

    assert client.post("/api/select", json={"case": "medicine"}).json == {"ok": True}
    assert node.generation == 1  # Viewing a gallery does not invalidate inference.
    assert client.post("/api/source", data={
        "source": "upload", "image": (picture("blue"), "target.png"),
    }).json == {"ok": True}
    assert node.source == "upload" and node.frame.mode == "RGB"
    assert node.frame.getpixel((0, 0)) == (0, 0, 255)
    assert (node.generation, node.frame_seq) == (2, 1)
    assert client.post("/api/source", data={"source": "camera"}).json == {"ok": True}
    assert node.source == "camera" and node.frame is None
    assert (node.generation, node.frame_seq) == (3, 2)
    assert client.get("/api/detections").json == {"detections": []}


def test_invalid_upload_batch_does_not_publish_partial_gallery(service):
    client, node = service
    response = client.post("/api/cases", data={
        "case": "medicine",
        "images": [(picture(), "valid.png"), (io.BytesIO(b"broken"), "broken.png")],
    })
    assert response.status_code == 400
    assert not (node.root / "medicine").exists()
    assert node.generation == 0 and node.selected == ""
    node.image_stream.reset.assert_not_called()
