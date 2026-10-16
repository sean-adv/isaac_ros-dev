"""Paths for the SAM3 TRT10 worker and DINOv3 TRT11 process."""

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
ARTIFACTS = WORKSPACE / "dinov3_ws/artifacts"
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
GROUNDING_MODEL_ID = "facebook/sam3"
INPUT_SIZE = 160
DINO_ENGINE = WORKSPACE / "dinov3_tensorrt/artifacts/thor-vitb-160/vitb16-fp16-trt.ts"
DINO_MODEL = WORKSPACE / "DART/models/dinov3-vitb16"
SAM3_ASSETS = WORKSPACE / "DART/dartf/assets_bbox"
GROUNDING_ENGINE = SAM3_ASSETS / "vision_int8.plan"
GROUNDING_PYTHON = WORKSPACE / "DART/.venv/bin/python"
SAM3_PLUGIN = WORKSPACE / "DART/dartf/plugins/lq_plugins.so"
