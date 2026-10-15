"""Object identity, rejection, duplicate proposals, and gallery lifecycle contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from dinov3_cases.grounded import (
    ObjectGallery,
    clean_proposals,
    prepare_object,
    recognize_objects,
    validate_settings,
    validate_bbox_limits,
    recognition_message,
    object_debug_views,
    ObjectVisualizationSnapshot,
)


class ColorEncoder:
    def encode(self, image):
        vector = torch.tensor(np.asarray(image).mean(axis=(0, 1)), dtype=torch.float32)
        return torch.nn.functional.normalize(vector, dim=0).repeat(
            image.width // 16 * (image.height // 16), 1
        )


def test_clip_suppress_duplicates_without_merging_adjacent_objects():
    boxes = [
        [-1.2, 2.2, 30.1, 40.8],
        [0, 2, 31, 41],
        [31, 2, 60, 41],
        [70, 70, 80, 80],
        [float("nan"), 0, 10, 10],
        [9, 9, 8, 8],
    ]
    result = clean_proposals(boxes, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], ["bottle"] * 6, (60, 50))
    assert [r["bbox"] for r in result] == [[0, 2, 31, 41], [31, 2, 60, 41]]
    assert result[0]["detection_score"] == 0.9
    assert len(clean_proposals(boxes, [0.9] * 6, ["box"] * 6, (60, 50), limit=1)) == 1
    assert clean_proposals([], [], [], (60, 50)) == []


def test_object_preprocessing_handles_small_and_thin_crops():
    for size in [(1, 1), (1, 1000), (1000, 1), (77, 151)]:
        result = prepare_object(Image.new("RGB", size, "red"), 224, 16)
        assert min(result.size) >= 32
        assert max(result.size) == 224
        assert all(n % 16 == 0 for n in result.size)
        assert result.getpixel((result.width - 1, result.height - 1)) == (255, 0, 0)


@pytest.mark.parametrize("limits", [dict(min_width=10), dict(min_height=10),
                                   dict(min_area=100)])
def test_small_boxes_filtered_after_clipping_before_detection_limit(limits):
    # The high-score box becomes 9x9 after clipping and must not consume limit=1.
    boxes = [[-20, -20, 9, 9], [20, 20, 30, 30]]
    result = clean_proposals(boxes, [.99, .8], ["box", "box"], (50, 50),
                             limit=1, **limits)
    assert [item["bbox"] for item in result] == [[20, 20, 30, 30]]
    assert result[0]["area"] == 100  # Equality with the threshold is retained.


def test_bbox_limits_disabled_and_combined():
    boxes = [[0, 0, 1, 1], [2, 2, 12, 7], [20, 20, 30, 30]]
    args = (boxes, [.99, .9, .8], ["box"] * 3, (50, 50))
    assert len(clean_proposals(*args)) == 3
    assert [item["bbox"] for item in clean_proposals(
        *args, min_width=10, min_height=5, min_area=100
    )] == [[20, 20, 30, 30]]


@pytest.mark.parametrize("value", [-1, 1.5, float("nan"), float("inf"), True, "10"])
@pytest.mark.parametrize("name", ["min_width", "min_height", "min_area"])
def test_invalid_bbox_limits_rejected(name, value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        validate_bbox_limits(**{name: value})


def test_gallery_multi_class_unknown_and_filesystem_updates(tmp_path):
    for name, color in [("red", "red"), ("blue", "blue")]:
        folder = tmp_path / name
        folder.mkdir()
        Image.new("RGB", (32, 48), color).save(folder / "one.png")
    gallery = ObjectGallery()
    assert gallery.refresh(tmp_path, ColorEncoder(), 64, 16)
    assert not gallery.refresh(tmp_path, ColorEncoder(), 64, 16)
    red = gallery.classify(torch.tensor([1.0, 0.0, 0.0]), 0.75, 0.05)
    assert red["label"] == "red" and red["margin"] == 1.0
    assert gallery.classify(torch.tensor([0.0, 1.0, 0.0]), 0.75, 0.05)["reason"] == "low_similarity"
    shared = torch.nn.functional.normalize(torch.tensor([1.0, 0.0, 1.0]), dim=0)
    assert gallery.classify(shared, 0.6, 0.05)["reason"] == "ambiguous"
    # Additional views of the same class must not occupy both top-two slots.
    Image.new("RGB", (32, 48), "red").save(tmp_path / "red/two.png")
    assert gallery.refresh(tmp_path, ColorEncoder(), 64, 16)
    assert gallery.classify(torch.tensor([1.0, 0.0, 0.0]), 0.75, 0.05)["label"] == "red"
    (tmp_path / "blue/one.png").unlink()
    assert gallery.refresh(tmp_path, ColorEncoder(), 64, 16)
    assert gallery.classify(torch.tensor([1.0, 0.0, 0.0]), 0.75, 0.05)["margin"] is None
    for path in (tmp_path / "red").iterdir():
        path.unlink()
    assert gallery.refresh(tmp_path, ColorEncoder(), 64, 16)
    assert gallery.classify(shared, 0.75, 0.05)["reason"] == "empty_gallery"


def test_roi_coordinates_identity_and_ros_header(tmp_path):
    from std_msgs.msg import Header

    encoder = ColorEncoder()
    for label, color in [("a", "red"), ("b", "blue")]:
        (tmp_path / label).mkdir()
        Image.new("RGB", (40, 40), color).save(tmp_path / label / "one.png")
    gallery = ObjectGallery()
    gallery.refresh(tmp_path, encoder, 64, 16)
    image = Image.new("RGB", (100, 50), "red")
    image.paste(Image.new("RGB", (50, 50), "blue"), (50, 0))
    proposals = clean_proposals(
        [[0, 0, 50, 50], [50, 0, 100, 50]], [0.9, 0.8], ["box", "bottle"], image.size
    )
    result = recognize_objects(image, proposals, encoder, gallery, 64, 16, 0.75, 0.05)
    assert [item["label"] for item in result] == ["a", "b"]
    assert [item["bbox"] for item in result] == [p["bbox"] for p in proposals]
    header = Header(frame_id="camera")
    header.stamp.sec = 123
    msg = recognition_message(result, header)
    assert msg.header == header
    assert [d.results[0].hypothesis.class_id for d in msg.detections] == ["a", "b"]
    # Frame-local detections have no cross-frame tracking identity.
    assert [d.id for d in msg.detections] == ["", ""]
    assert all(d.header == header and d.bbox.size_x == 50 for d in msg.detections)
    assert recognition_message([], header).detections == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("prompt", ""),
        ("prompt", "..."),
        ("prompt", None),
        ("prompt", "a" * 1001),
        ("box_threshold", True),
        ("box_threshold", float("nan")),
        ("margin", -1),
        ("recognition_threshold", float("inf")),
    ],
)
def test_invalid_runtime_settings(key, value):
    settings = dict(
        prompt="bottle. box",
        box_threshold=0.3,
        recognition_threshold=0.75,
        margin=0.05,
    )
    settings[key] = value
    with pytest.raises(ValueError):
        validate_settings(**settings)


def test_detector_uses_original_image_size_and_class_labels():
    from dinov3_cases.omdet_trt import postprocess
    boxes = torch.tensor([[[0.5, 0.5, 0.8, 0.8]]])
    logits = torch.tensor([[[torch.logit(torch.tensor(0.9))]]])
    result = postprocess(boxes, logits, ["bottle"], (100, 50), 0.3, 0.5, 20)
    np.testing.assert_allclose(result[0]["bbox"], [10, 5, 90, 45], atol=1)
    assert result[0]["detection_label"] == "bottle"


def test_object_heatmap_reuses_forward_and_best_reference(tmp_path):
    class CountingEncoder(ColorEncoder):
        calls = 0

        def encode(self, image):
            self.calls += 1
            return super().encode(image)

    encoder = CountingEncoder()
    (tmp_path / "red").mkdir()
    Image.new("RGB", (32, 48), "red").save(tmp_path / "red/one.png")
    gallery = ObjectGallery()
    gallery.refresh(tmp_path, encoder, 64, 16)
    # The pixels displayed as reference must be the same cached input used for encoding.
    (tmp_path / "red/one.png").unlink()
    encoder.calls = 0
    roi = Image.new("RGB", (43, 79), "red")
    proposals = clean_proposals([[0, 0, 43, 79]], [0.9], ["bottle"], roi.size)
    diagnostics = []
    result = recognize_objects(roi, proposals, encoder, gallery, 64, 16, 0.75, 0.05, diagnostics)
    assert encoder.calls == 1
    assert result[0]["reference"] == dict(case="red", name="one.png")
    assert "_reference_index" not in result[0]
    views, metadata = diagnostics[0]["views"], diagnostics[0]["metadata"]
    assert set(views) == {"crop", "grid", "patches", "heatmap", "similarity", "reference"}
    assert views["heatmap"].shape == (79, 43, 3)
    assert views["grid"].shape == (64, 64, 3)
    assert metadata["content_size"] == [35, 64]
    assert metadata["patch_grid"] == [4, 4]
    assert metadata["similarity_min"] == pytest.approx(1.0)


def test_heatmap_removes_right_padding_before_mapping_back_to_roi():
    roi = Image.new("RGB", (16, 32))
    prepared = prepare_object(roi, 32, 16)
    # Actual content is the left patch column. The right column is minimum-size padding.
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    views, metadata = object_debug_views(
        roi, prepared, features, torch.tensor([[1.0, 0.0]]), 32, 16
    )
    assert metadata["content_size"] == [16, 32] and metadata["input_size"] == [32, 32]
    assert views["heatmap"].shape == (32, 16, 3)
    # Last valid pixel interpolates at the edge; ROI must not include the zero-score padding column.
    assert metadata["similarity_min"] > 0.5
    assert metadata["similarity_max"] == pytest.approx(1.0)


def test_empty_gallery_and_large_crop_visualizations():
    roi = Image.new("RGB", (1500, 3000), "blue")
    diagnostics = []
    proposal = dict(bbox=[0, 0, 1500, 3000], detection_score=0.9, detection_label="box")
    result = recognize_objects(
        roi, [proposal], ColorEncoder(), ObjectGallery(), 64, 16, 0.75, 0.05, diagnostics
    )
    assert result[0]["reason"] == "empty_gallery"
    assert set(diagnostics[0]["views"]) == {"crop", "grid"}
    assert diagnostics[0]["views"]["crop"].shape == (640, 320, 3)
    assert not diagnostics[0]["metadata"]["heatmap_available"]


def test_selected_label_compares_every_object_and_all_reference_images_without_forward(tmp_path):
    class PatchEncoder:
        calls = 0

        def encode(self, image):
            self.calls += 1
            pixels = np.asarray(image)
            features = [
                pixels[y : y + 16, x : x + 16].mean(axis=(0, 1))
                for y in range(0, image.height, 16)
                for x in range(0, image.width, 16)
            ]
            return torch.nn.functional.normalize(
                torch.tensor(np.array(features), dtype=torch.float32), dim=1
            )

    encoder = PatchEncoder()
    for label, name, color in [
        ("mix", "red.png", "red"),
        ("mix", "blue.png", "blue"),
        ("green", "one.png", (0, 255, 0)),
    ]:
        (tmp_path / label).mkdir(exist_ok=True)
        Image.new("RGB", (32, 32), color).save(tmp_path / label / name)
    gallery = ObjectGallery()
    gallery.refresh(tmp_path, encoder, 32, 16)
    mixed = Image.new("RGB", (32, 32), "red")
    mixed.paste(Image.new("RGB", (16, 32), "blue"), (16, 0))
    target = Image.new("RGB", (64, 32), (0, 255, 0))
    target.paste(mixed, (0, 0))
    proposals = clean_proposals(
        [[0, 0, 32, 32], [32, 0, 64, 32]], [0.9, 0.8], ["box"] * 2, target.size
    )
    diagnostics = []
    results = recognize_objects(target, proposals, encoder, gallery, 32, 16, 0.6, 0.05, diagnostics)
    snapshot = ObjectVisualizationSnapshot(results, diagnostics, gallery, 32, 16)
    calls = encoder.calls
    automatic, _ = snapshot.render()
    selected, images = snapshot.render("mix")
    assert encoder.calls == calls
    assert len(selected) == 2
    assert (
        [item["label"] for item in selected]
        == [item["label"] for item in automatic]
        == ["mix", "green"]
    )
    assert [item["comparison_case"] for item in selected] == ["mix", "mix"]
    assert all(item["comparison_reference_count"] == 2 for item in selected)
    assert selected[0]["similarity_min"] == pytest.approx(1.0)
    assert automatic[0]["similarity_min"] < 0.5
    assert selected[1]["similarity_max"] == pytest.approx(0.0)
    assert automatic[1]["similarity_min"] == pytest.approx(1.0)
    assert selected[1]["comparison_score"] == pytest.approx(0.0)
    assert selected[1]["score"] == pytest.approx(1.0)
    assert len(images) == 12
    with pytest.raises(ValueError):
        snapshot.render("missing")
    # Gallery changes cannot replace the reference features in an existing frame snapshot.
    for path in (tmp_path / "mix").iterdir():
        path.unlink()
    gallery.refresh(tmp_path, encoder, 32, 16)
    assert snapshot.render("mix")[0][0]["comparison_reference_count"] == 2
    assert gallery.best_reference(torch.tensor([1.0, 0.0, 0.0]), "mix") == (None, None)


def test_deferred_diagnostics_do_not_render_until_requested(monkeypatch):
    import dinov3_cases.grounded as grounded

    image = Image.new("RGB", (32, 32), "red")
    proposals = clean_proposals([[0, 0, 32, 32]], [0.9], ["box"], image.size)
    diagnostics = []
    original = grounded.build_object_diagnostic
    calls = []

    def render(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(grounded, "build_object_diagnostic", render)
    gallery = ObjectGallery()
    results = recognize_objects(
        image,
        proposals,
        ColorEncoder(),
        gallery,
        32,
        16,
        0.75,
        0.05,
        diagnostics,
        defer_diagnostics=True,
    )
    snapshot = ObjectVisualizationSnapshot(results, diagnostics, gallery, 32, 16)
    assert calls == []
    items, images = snapshot.render()
    assert len(calls) == 1 and set(items[0]["views"]) == {"crop", "grid"}
    assert images[0, "crop"].startswith(b"\xff\xd8")
    assert snapshot.render() == (items, images)
    assert len(calls) == 1


@pytest.mark.parametrize("size", [(1, 1000), (1000, 1), (77, 151), (224, 224)])
def test_square_object_input_preserves_content_and_pads_to_224(size):
    image = Image.new("RGB", size, "red")
    result = prepare_object(image, 224, 16)
    assert result.size == (224, 224)
    assert result.getpixel((223, 223)) == (255, 0, 0)


def test_square_object_padding_is_removed_from_heatmap():
    roi = Image.new("RGB", (32, 16), "red")
    prepared = prepare_object(roi, 32, 16)
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    views, metadata = object_debug_views(
        roi, prepared, features, torch.tensor([[1.0, 0.0]]), 32, 16
    )
    assert metadata["input_size"] == [32, 32]
    assert metadata["content_size"] == [32, 16]
    assert metadata["patch_grid"] == [2, 2]
    assert views["heatmap"].shape[:2] == (16, 32)
    # Bilinear interpolation may blend across the boundary, but the pure
    # padding region at score zero must not appear in the cropped ROI map.
    import cv2
    import numpy as np

    padding_color = cv2.applyColorMap(np.zeros((1, 1), dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
    assert not (views["heatmap"] == padding_color).all(axis=-1).any()


def test_ros_2d_filters_unknown_without_changing_tracking_ids_or_input():
    from std_msgs.msg import Header

    items = [dict(label=label, score=.9, track_id=track_id, bbox=[0, 0, 100, 100])
             for label, track_id in [("unknown", 7), ("medicine", 42), ("unknown", 8)]]
    header = Header(frame_id="camera")
    message = recognition_message(items, header)
    assert message.header == header
    assert len(message.detections) == 1
    assert message.detections[0].id == "42"
    assert message.detections[0].results[0].hypothesis.class_id == "medicine"
    assert not recognition_message([items[0], items[2]], header).detections
    assert [item["label"] for item in items] == ["unknown", "medicine", "unknown"]
