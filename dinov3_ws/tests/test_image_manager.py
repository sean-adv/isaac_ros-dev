import io

import cv2
import numpy as np
import pytest
from PIL import Image

from dinov3_cases.image_manager import create_app, remove_background


def options(**overrides):
    return dict(x=0, y=0, hue=15, saturation=20, value=20, feather=0, connected=True, **overrides)


def test_connected_background_preserves_isolated_same_color_and_original():
    source = np.full((9, 9, 4), 255, dtype=np.uint8)
    source[2:7, 2:7, :3] = [0, 0, 255]
    source[4, 4, :3] = 255
    image = Image.fromarray(source)
    result = np.array(remove_background(image, options()))
    assert result[0, 0, 3] == 0
    assert result[4, 4, 3] == 255
    assert result[3, 3, 3] == 255
    np.testing.assert_array_equal(np.array(image), source)
    result = np.array(remove_background(image, {**options(), "connected": False}))
    assert result[4, 4, 3] == 0
    assert (result[0, 0, :3] == 0).all()


def test_hue_wraps_across_red_boundary():
    hsv = np.array([[[179, 255, 255], [1, 255, 255], [90, 255, 255]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    result = np.array(remove_background(Image.fromarray(rgb), {**options(), "hue": 6}))
    assert result[0, :, 3].tolist() == [0, 0, 255]


def test_existing_alpha_and_feather():
    rgba = np.zeros((12, 12, 4), dtype=np.uint8)
    rgba[:, :, :3] = [0, 255, 0]
    rgba[:, :, 3] = 128
    rgba[:, 6:, :3] = [255, 0, 0]
    rgba[11, 11, 3] = 0
    result = np.array(remove_background(Image.fromarray(rgba), {**options(), "feather": 1}))
    assert result[11, 11, 3] == 0
    assert result[:, :, 3].max() <= 128
    assert 0 < result[5, 5, 3] < 128


@pytest.mark.parametrize(
    "override",
    [
        {"x": -1},
        {"x": 0.5},
        {"y": 9},
        {"hue": float("nan")},
        {"value": 256},
        {"connected": "false"},
        {"feather": -1},
        {"x": True},
    ],
)
def test_invalid_options(override):
    with pytest.raises(ValueError):
        remove_background(Image.new("RGB", (4, 4)), {**options(), **override})


def test_transparent_sample_rejected():
    with pytest.raises(ValueError):
        remove_background(Image.new("RGBA", (4, 4), (1, 2, 3, 0)), options())


@pytest.fixture
def gallery(tmp_path):
    app = create_app(tmp_path / "cases")
    app.config["TESTING"] = True
    return app.test_client(), tmp_path / "cases"


def upload(client, group="medicine_01"):
    stream = io.BytesIO()
    Image.new("RGB", (10, 10), (255, 255, 255)).save(stream, format="PNG")
    stream.seek(0)
    response = client.post(
        "/api/images/upload", data={"case": group, "images": (stream, "test.png")}
    )
    assert response.status_code == 200
    return response.json["names"][0]


def test_gallery_preview_save_and_delete(gallery):
    client, root = gallery
    name = upload(client)
    original = (root / "medicine_01" / name).read_bytes()
    assert client.get("/images").status_code == 200
    listing = client.get("/api/images?case=medicine_01").json
    assert listing["cases"] == ["medicine_01"]
    assert listing["images"][0]["name"] == name
    data = dict(case="medicine_01", name=name, target_case="processed", options=options())
    response = client.post("/api/images/preview", json=data)
    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert Image.open(io.BytesIO(response.data)).getpixel((0, 0))[3] == 0
    assert len(list((root / "medicine_01").iterdir())) == 1
    saved = client.post("/api/images/cutout", json=data)
    assert saved.status_code == 200
    assert (root / "medicine_01" / name).read_bytes() == original
    new_path = root / "processed" / saved.json["name"]
    assert new_path.is_file()
    assert client.delete("/api/images/file/processed/" + saved.json["name"]).status_code == 200
    assert not new_path.exists()
    assert client.delete("/api/images/file/medicine_01/" + name).status_code == 200
    assert client.get("/api/images?case=medicine_01").json["images"] == []


def test_invalid_payload_and_symlink_cannot_escape(gallery, tmp_path):
    client, root = gallery
    name = upload(client)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    (root / "medicine_01" / "link.png").symlink_to(outside)
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    assert client.get("/api/images?case=../escape").status_code == 400
    assert client.get("/api/images?case=escape").status_code == 400
    assert client.delete("/api/images/file/medicine_01/link.png").status_code == 400
    assert outside.exists()
    assert client.get("/api/images/file/medicine_01/link.png").status_code == 400
    assert len(client.get("/api/images?case=medicine_01").json["images"]) == 1
    for data in [
        None,
        [],
        {},
        {"case": "medicine_01", "name": "../outside.png"},
        {"case": "medicine_01", "name": name, "options": {}},
    ]:
        assert client.post("/api/images/preview", json=data).status_code == 400


def test_exif_orientation_matches_click_coordinates(gallery):
    client, root = gallery
    folder = root / "rotated"
    folder.mkdir()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (12, 8), "white").save(folder / "image.jpg", exif=exif)
    response = client.get("/api/images/file/rotated/image.jpg")
    assert Image.open(io.BytesIO(response.data)).size == (8, 12)
    response = client.post(
        "/api/images/preview",
        json={"case": "rotated", "name": "image.jpg", "options": {**options(), "x": 7, "y": 11}},
    )
    assert response.status_code == 200


def test_upload_preserves_alpha_and_rejects_invalid_image(gallery):
    client, root = gallery
    stream = io.BytesIO()
    Image.new("RGBA", (3, 3), (0, 0, 255, 64)).save(stream, format="PNG")
    stream.seek(0)
    response = client.post(
        "/api/images/upload", data={"case": "alpha", "images": (stream, "alpha.png")}
    )
    path = root / "alpha" / response.json["names"][0]
    assert Image.open(path).getpixel((0, 0))[3] == 64
    response = client.post(
        "/api/images/upload",
        data={"case": "invalid", "images": (io.BytesIO(b"not an image"), "invalid.png")},
    )
    assert response.status_code == 400
    assert not (root / "invalid").exists()


def test_reference_change_callback(tmp_path):
    from flask import Flask
    from dinov3_cases.image_manager import register_image_manager

    app = Flask(__name__)
    changes = []
    register_image_manager(app, tmp_path, changes.append)
    client = app.test_client()
    name = upload(client, "active")
    assert changes == ["active"]
    response = client.post(
        "/api/images/cutout",
        json={"case": "active", "name": name, "target_case": "other", "options": options()},
    )
    assert response.status_code == 200
    assert changes == ["active", "other"]
    client.delete("/api/images/file/active/" + name)
    assert changes == ["active", "other", "active"]


def test_grayscale_background_ignores_unstable_hue():
    image = Image.fromarray(
        np.array([[[255, 255, 255], [250, 249, 250], [100, 100, 100]]], dtype=np.uint8)
    )
    result = np.array(remove_background(image, {**options(), "hue": 0}))
    assert result[0, :, 3].tolist() == [0, 0, 255]
