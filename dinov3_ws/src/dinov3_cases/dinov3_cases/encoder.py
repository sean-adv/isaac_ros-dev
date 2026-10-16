"""DINOv3 TensorRT patch encoder for gallery and object images."""

import numpy as np
import torch
from transformers import AutoImageProcessor

from .patches import run_dino_image
from .runtime import MODEL_ID, DINO_ENGINE, DINO_MODEL
from .trt_backend import TRTBackbone


class Encoder:
    def __init__(self, engine_path=DINO_ENGINE):
        self.model = TRTBackbone(engine_path, MODEL_ID)
        self.device = str(self.model.device)
        self.description = self.model.description
        self.processor = AutoImageProcessor.from_pretrained(
            str(DINO_MODEL), local_files_only=True
        )

    @torch.inference_mode()
    def encode(self, image):
        pixels = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).to(self.device)
        features = run_dino_image(pixels, self.model, self.processor)
        return features[0].reshape(-1, features.shape[-1])
