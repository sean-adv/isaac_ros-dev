"""Paths and the two supported TensorRT deployment modes."""

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
ARTIFACTS = WORKSPACE / "dinov3_ws/artifacts"
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
INPUT_SIZE = 224
DINO_ENGINE = ARTIFACTS / "tensorrt11/dinov3_vits16_fp16_224.engine"
GROUNDING_ENGINE = ARTIFACTS / "tensorrt11/grounding_dino_mixed_fp16.engine"
LEGACY_GROUNDING_ENGINE = ARTIFACTS / "tensorrt/grounding_dino_wingdzero_fp16.engine"
LEGACY_PYTHON = WORKSPACE / "dinov3_venv/bin/python"
MODES = {
    "split": ("subprocess", LEGACY_GROUNDING_ENGINE),
    "single": ("inprocess", GROUNDING_ENGINE),
}
