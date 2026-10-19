"""Cat2Real pooler descriptor and spatial features from one TRT forward."""
import numpy as np
import torch
from transformers import AutoImageProcessor

from .patches import preprocess_image, patch_features
from .cat2real import prepare_cat2real_image
from .runtime import MODEL_ID, DINO_ENGINE, DINO_MODEL, INPUT_SIZE
from .trt_backend import TRTBackbone


class Encoder:
    def __init__(self, engine_path=DINO_ENGINE):
        self.model = TRTBackbone(engine_path, MODEL_ID)
        self.device = str(self.model.device)
        self.description = self.model.description
        self.processor = AutoImageProcessor.from_pretrained(str(DINO_MODEL), local_files_only=True)

    def prepare_image(self, image, max_side, stride):
        return prepare_cat2real_image(image, INPUT_SIZE)

    @torch.inference_mode()
    def encode_with_descriptor(self, image):
        pixels = torch.from_numpy(np.array(image.convert('RGB'), dtype=np.uint8)).permute(2, 0, 1).to(self.device)
        outputs = self.model(pixel_values=preprocess_image(pixels, self.processor))
        features = patch_features(outputs, self.model, image.height, image.width)
        features = features[0].reshape(-1, features.shape[-1])
        descriptor = outputs.pooler_output[0].float()
        if not torch.isfinite(features).all() or not torch.isfinite(descriptor).all():
            raise ValueError('Non-finite Cat2Real image features')
        return features, descriptor

    def encode(self, image):
        return self.encode_with_descriptor(image)[0]
