"""ROS/HTTP two-stage integration with tiny DINOv3 and deterministic proposals."""

import io
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import rclpy
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast

from dinov3_cases.node import CaseNode
from dinov3_cases.patches import run_dino_image


class TestEncoder:
    device = "cpu"
    description = "offline ViT fixture"

    def __init__(self, engine_path):
        torch.manual_seed(9)
        self.model = DINOv3ViTModel(
            DINOv3ViTConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_register_tokens=4,
            )
        ).eval()
        self.processor = DINOv3ViTImageProcessorFast()

    def encode(self, image):
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        return run_dino_image(pixels, self.model, self.processor)[0].reshape(-1, 32)


class Detector:
    def __init__(self, *args):
        pass

    def validate_prompt(self, prompt):
        pass

    def detect(self, image, prompt, *args):
        if prompt == "nothing.":
            return []
        return [
            dict(
                bbox=[0, 0, image.width, image.height],
                detection_label="box",
                detection_score=0.9,
                area=image.width * image.height,
            )
        ]


def png(image):
    data = io.BytesIO()
    image.save(data, format="PNG")
    data.seek(0)
    return data


def wait_result(client):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if client.get("/overlay.jpg").status_code == 200:
            return client.get("/api/detections").json["detections"]
        time.sleep(0.1)
    raise AssertionError(client.get("/api/status").json)


def check_views(client, expected_views):
    manifest = client.get("/api/object_visualizations").json
    assert len(manifest["objects"]) == 1
    item = manifest["objects"][0]
    assert set(item["views"]) == expected_views
    assert item["bbox"] == [0, 0, 96, 64]
    for view in expected_views:
        response = client.get(f'/debug/objects/{manifest["revision"]}/0/{view}.jpg')
        assert response.status_code == 200
        image = Image.open(io.BytesIO(response.data))
        if view in ("crop", "heatmap", "similarity"):
            assert image.size == (96, 64)
    assert client.get(f'/debug/objects/{manifest["revision"]}/0/mask.jpg').status_code == 404
    return manifest["revision"]


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    picture = Image.fromarray(
        np.random.default_rng(9).integers(0, 255, (64, 96, 3), dtype=np.uint8)
    )
    rclpy.init(args=["--ros-args", "-p", f'cases_dir:={root / "cases"}', "-p", "port:=0"])
    with patch("dinov3_cases.node.GroundingDetector", Detector), patch(
        "dinov3_cases.node.Encoder", TestEncoder
    ):
        node = CaseNode()
        client = node.app.test_client()
        try:
            assert client.get("/api/status").json["object_max_side"] == 224
            assert (
                client.post(
                    "/api/source", data={"source": "upload", "image": (png(picture), "target.png")}
                ).status_code
                == 200
            )
            result = wait_result(client)
            assert result[0]["reason"] == "empty_gallery" and result[0]["score"] is None
            old_revision = check_views(client, {"crop", "grid"})
            assert (
                client.post(
                    "/api/cases", data={"case": "a", "images": (png(picture), "one.png")}
                ).status_code
                == 200
            )
            assert client.get(f"/debug/objects/{old_revision}/0/crop.jpg").status_code == 410
            result = wait_result(client)
            assert result[0]["label"] == "a" and result[0]["score"] > 0.99
            assert result[0]["bbox"] == [0, 0, 96, 64]
            check_views(client, {"crop", "grid", "heatmap", "similarity", "patches", "reference"})
            revision = client.get("/api/object_visualizations").json["revision"]
            assert client.post("/api/select", json={"case": "a"}).status_code == 200
            assert client.get("/api/object_visualizations").json["revision"] == revision
            assert client.get("/overlay.jpg").status_code == 200
            assert (
                client.post(
                    "/api/cases", data={"case": "b", "images": (png(picture), "one.png")}
                ).status_code
                == 200
            )
            result = wait_result(client)
            assert result[0]["label"] == "unknown" and result[0]["reason"] == "ambiguous"
            check_views(client, {"crop", "grid", "heatmap", "similarity", "patches", "reference"})
            original = client.get("/api/detections").json
            before = client.get("/api/object_visualizations").json
            selected_views = client.get("/api/object_visualizations?reference_case=b")
            assert selected_views.status_code == 200
            selected_manifest = selected_views.json
            assert selected_manifest["reference_cases"] == ["a", "b"]
            assert selected_manifest["reference_case"] == "b"
            item = selected_manifest["objects"][0]
            assert item["comparison_case"] == item["comparison_reference"]["case"] == "b"
            assert item["comparison_reference_count"] == 1
            assert selected_manifest["revision"] == before["revision"]
            assert client.get("/api/detections").json == original
            assert (
                client.get(
                    f'/debug/objects/{before["revision"]}/0/heatmap.jpg?reference_case=b'
                ).status_code
                == 200
            )
            invalid = client.get("/api/object_visualizations?reference_case=missing")
            assert invalid.status_code == 400 and invalid.json["reference_cases"] == ["a", "b"]
            assert (
                client.get(
                    f'/debug/objects/{before["revision"]}/0/heatmap.jpg?reference_case=missing'
                ).status_code
                == 400
            )
            assert client.post("/api/grounding", json={"margin": 0.0}).status_code == 200
            assert wait_result(client)[0]["label"] in ("a", "b")
            assert client.post("/api/grounding", json={"prompt": "nothing"}).status_code == 200
            assert wait_result(client) == []
            assert client.get("/api/object_visualizations").json["objects"] == []
            for payload in [
                {},
                [],
                {"extra": 2},
                {"prompt": ""},
                {"margin": True},
                {"box_threshold": 1.1},
            ]:
                assert client.post("/api/grounding", json=payload).status_code == 400
            assert (
                client.post("/api/grounding", json={"prompt": "box", "margin": 0.05}).status_code
                == 200
            )
            assert wait_result(client)[0]["reason"] == "ambiguous"
            # Delete an unselected class through image manager and refresh static input.
            selected = client.get("/api/status").json["selected"]
            other = "a" if selected == "b" else "b"
            for path in (root / "cases" / other).iterdir():
                path.unlink()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                result = client.get("/api/detections").json["detections"]
                if result and result[0]["label"] == selected:
                    break
                time.sleep(0.1)
            assert result[0]["label"] == selected
            # Camera header survives both stages.
            captured = []
            with patch.object(
                node.recognition_pub, "publish", side_effect=captured.append
            ), patch.object(node.recognition_pub, "get_subscription_count", return_value=1):
                assert client.post("/api/source", data={"source": "camera"}).status_code == 200
                frame = node.bridge.cv2_to_imgmsg(np.asarray(picture), encoding="rgb8")
                frame.header.frame_id = "test_camera"
                frame.header.stamp.sec = 345
                node.on_image(frame)
                wait_result(client)
                deadline = time.monotonic() + 5
                while not captured and time.monotonic() < deadline:
                    time.sleep(0.01)
            assert captured and captured[-1].header == frame.header
            assert captured[-1].detections[0].results[0].hypothesis.class_id == selected
            print(
                "PASS: grounded ROI inference, per-object heatmaps, reference views, stale cache rejection, unknown, static refresh, camera headers"
            )
        finally:
            node.stop.set()
            node.server.shutdown()
            node.preview_worker.join(2)
            node.worker.join(30)
            node.publisher_worker.join(2)
            node.destroy_node()
            rclpy.shutdown()
