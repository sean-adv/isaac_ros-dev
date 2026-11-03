"""CPU test double for the current encoder protocol; no runtime fallback support."""
import numpy as np
import torch

from dinov3_cases.inference.cat2real import prepare_cat2real_image
from dinov3_cases.recognition.grounded import encode_object


class ColorEncoder:
    def __init__(self, size=64):
        self.size = size
        self.calls = 0

    def prepare_image(self, image, max_side, stride):
        return prepare_cat2real_image(image, max_side)

    def encode_with_descriptor(self, image):
        self.calls += 1
        x0, y0, x1, y1 = image.info['content_box']
        content = np.asarray(image)[y0:y1, x0:x1]
        descriptor = torch.tensor(content.mean(axis=(0, 1)), dtype=torch.float32)
        vector = torch.nn.functional.normalize(descriptor, dim=0)
        features = vector.repeat(image.width // 16 * (image.height // 16), 1)
        return features, descriptor

    def encode_regions(self, image, boxes):
        return [encode_object(self, image.crop(box), self.size, 16) for box in boxes]
