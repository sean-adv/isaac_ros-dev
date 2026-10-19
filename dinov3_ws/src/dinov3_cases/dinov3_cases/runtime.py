"""Self-contained model/data paths for the official Torch-TensorRT deployment."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
WORKSPACE = PROJECT.parent  # Retained for workspace-level ROS/camera diagnostics.
ARTIFACTS = PROJECT / 'artifacts'
CASES_DIR = PROJECT / 'data/gallery'
MODEL_ID = 'zhanganyi88/Cat2Real-DINOv3-384'
GROUNDING_MODEL_ID = 'facebook/sam3'
SAM3_REVISION = '3c879f39826c281e95690f02c7821c4de09afae7'
INPUT_SIZE = 384
DINO_ENGINE = ARTIFACTS / 'cat2real-dinov3-384/cat2real-fp32-trt.ts'
DINO_MODEL = ARTIFACTS / 'cat2real-dinov3-384/source'
SAM3_ASSETS = ARTIFACTS / 'sam3'
SAM3_MODEL = SAM3_ASSETS / 'source'
GROUNDING_ENGINE = SAM3_ASSETS / 'vision-fp16.ts'
GROUNDING_PYTHON = Path(sys.executable)
