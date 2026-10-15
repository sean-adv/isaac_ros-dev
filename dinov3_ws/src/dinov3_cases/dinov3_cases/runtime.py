"""Paths for the SAM3 TRT10 worker and DINOv3 TRT11 process."""

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
ARTIFACTS = WORKSPACE / "dinov3_ws/artifacts"
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
GROUNDING_MODEL_ID = "facebook/sam3"
INPUT_SIZE = 224
DINO_ENGINE = ARTIFACTS / "dinov3_facebook_fp16/model.engine"
SAM3_ASSETS = WORKSPACE / "DART/dartf/assets_bbox"
GROUNDING_ENGINE = SAM3_ASSETS / "vision_int8.plan"
TRT_PYTHON = WORKSPACE / "dinov3_trt11_venv/bin/python"
GROUNDING_PYTHON = WORKSPACE / "DART/.venv/bin/python"
SAM3_PLUGIN = WORKSPACE / "DART/dartf/plugins/lq_plugins.so"
MODES = {
    "split": ("subprocess", GROUNDING_ENGINE),
}
