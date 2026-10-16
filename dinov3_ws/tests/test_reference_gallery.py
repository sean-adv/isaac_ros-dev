"""Reference grids must follow the same resize and stride as inference."""

import threading
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from dinov3_cases.web import create_app
from dinov3_cases.patches import reference_grid_image


@pytest.mark.parametrize("stride", [16, 32])
def test_grid_preserves_pixels_including_incomplete_edges(stride):
    original = Image.new("RGB", (stride * 2 + 5, stride + 3), (10, 20, 30))
    before = np.array(original)
    grid = reference_grid_image(original, stride)
    assert grid.size == original.size
    assert grid.getpixel((stride, 4)) == (255, 220, 80)
    assert grid.getpixel((4, 4)) == (10, 20, 30)
    assert grid.getpixel((stride * 2 + 1, 4)) == (10, 20, 30)
    assert grid.getpixel((4, stride + 1)) == (10, 20, 30)
    np.testing.assert_array_equal(original, before)


@pytest.fixture
def gallery(tmp_path):
    folder = tmp_path / "sample"
    folder.mkdir()
    Image.new("RGB", (101, 65), "white").save(folder / "one.png")
    Image.new("RGB", (64, 32), "blue").save(folder / "two.png")
    (folder / "broken.png").write_bytes(b"broken")
    (folder / "link.png").symlink_to(folder / "one.png")
    node = SimpleNamespace(
        root=tmp_path,
        lock=threading.Lock(),
        reference_stride=16,
    )
    app = create_app(node)
    app.config["TESTING"] = True
    return app.test_client(), node


def test_lists_every_reference_with_actual_resize_even_without_target(gallery):
    client, _ = gallery
    result = client.get("/api/references?case=sample")
    assert result.status_code == 200
    data = result.json
    assert data["patch_size"] == 16
    assert [item["name"] for item in data["images"]] == ["broken.png", "one.png", "two.png"]
    assert "error" in data["images"][0]
    item = data["images"][1]
    assert item["original_size"] == [101, 65]
    assert item["size"] == [160, 160]
    assert (item["rows"], item["columns"]) == (10, 10)
    response = client.get("/api/references/grid/sample/one.png")
    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert Image.open(BytesIO(response.data)).size == (160, 160)


def test_loading_invalid_paths_and_symlinks(gallery):
    client, node = gallery
    assert client.get("/api/references?case=../sample").status_code == 400
    assert client.get("/api/references?case=missing").status_code == 400
    assert client.get("/api/references/grid/sample/link.png").status_code == 400
    assert client.get("/api/references/grid/sample/missing.png").status_code == 400
    node.reference_stride = None
    assert client.get("/api/references?case=sample").status_code == 503
    assert client.get("/api/references/grid/sample/one.png").status_code == 503
