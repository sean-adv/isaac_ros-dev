"""Official Torch-TensorRT SAM3 vision plus Hugging Face PyTorch bbox head."""
import html
import json
from pathlib import Path
import threading

import ftfy
import numpy as np
from PIL import Image
import torch
from transformers import CLIPTokenizer, Sam3Model

from .model_modules import Sam3BboxHead


def parse_prompts(prompt):
    if not isinstance(prompt, str):
        raise ValueError('SAM3 prompt must be text')
    labels = [s.strip() for s in prompt.strip().rstrip('.').split(',') if s.strip()]
    if not 1 <= len(labels) <= 4:
        raise ValueError('SAM3 requires one to four comma-separated class names')
    if any(len(s) > 1000 for s in labels):
        raise ValueError('SAM3 class names must be at most 1000 characters')
    return labels


def prepare_sam3_pixels(image):
    resized = image.convert('RGB').resize((1008, 1008), Image.Resampling.BILINEAR)
    pixels = np.ascontiguousarray(np.asarray(resized).transpose(2, 0, 1), dtype=np.float32)
    pixels *= np.float32(1 / 127.5)
    pixels -= np.float32(1)
    return pixels[None]


@torch.inference_mode()
def prepare_sam3_tensor(image, device):
    """GPU antialiased bilinear resize, uint8 rounding, then model normalization."""
    device = torch.device(device)
    if device.type != 'cuda':
        return torch.from_numpy(prepare_sam3_pixels(image)).to(device, dtype=torch.float16)
    rgb = torch.from_numpy(np.array(image.convert('RGB'), dtype=np.uint8)).to(device)
    pixels = rgb.permute(2,0,1).unsqueeze(0).float()
    pixels = torch.nn.functional.interpolate(pixels, size=(1008,1008), mode='bilinear',
                                             align_corners=False, antialias=True)
    pixels.round_().clamp_(0,255).mul_(1/127.5).sub_(1)
    return pixels.to(torch.float16)


def clean_text(text):
    return ' '.join(html.unescape(html.unescape(ftfy.fix_text(text))).strip().split()).lower()


class Sam3Detector:
    def __init__(self, engine_path, device):
        path = Path(engine_path).expanduser().resolve()
        source = path.parent / 'source'
        if not path.is_file() or not (source / 'model.safetensors').is_file():
            raise FileNotFoundError('Missing official SAM3 model; run build_models.py --target sam3')
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata['model_id'] != 'facebook/sam3' or metadata['input_shape'] != [1, 3, 1008, 1008]:
            raise ValueError('SAM3 artifact metadata mismatch')
        if not torch.cuda.is_available():
            raise RuntimeError('SAM3 Torch-TensorRT requires CUDA')
        import torch_tensorrt  # Registers the official serialized TRT operators.

        self.device = torch.device(device)
        self.lock = threading.RLock()
        self.vision = torch.jit.load(str(path), map_location=self.device).eval()
        model = Sam3Model.from_pretrained(str(source), local_files_only=True, dtype=torch.float16).eval()
        self.head = Sam3BboxHead(model).to(self.device).eval()
        self.text_encoder = model.text_encoder.to(self.device).eval()
        self.text_projection = model.text_projection.to(self.device).eval()
        # The compiled engine owns the vision weights; masks and the eager vision are not retained.
        del model
        self.tokenizer = CLIPTokenizer.from_pretrained(str(source), local_files_only=True)
        self.cached_prompt = None
        self.precision = 'Official Torch-TensorRT FP16 vision / PyTorch FP16 bbox head'

    def validate_prompt(self, prompt):
        return parse_prompts(prompt)

    @torch.inference_mode()
    def _text_features(self, labels):
        key = tuple(labels)
        if key != self.cached_prompt:
            tokens = self.tokenizer([clean_text(s) for s in labels], padding='max_length',
                truncation=True, max_length=32, return_tensors='pt').to(self.device)
            mask = tokens['attention_mask'].bool()
            # SAM3 uses zero padding, while the CLIP tokenizer pads with its EOS token.
            ids = tokens['input_ids'].masked_fill(~mask, 0)
            result = self.text_encoder(input_ids=ids, attention_mask=mask, return_dict=True)
            self.features = self.text_projection(result.last_hidden_state), mask
            self.cached_prompt = key
        return self.features

    @torch.inference_mode()
    def predict_tensors(self, pixels, text_features, text_mask):
        values = self.vision(pixels)
        if len(values) != 9 or tuple(values[3].shape) != (1, 256, 72, 72):
            raise RuntimeError('Unexpected SAM3 vision feature interface')
        return self.head(values[3], values[7], text_features, text_mask)

    @torch.inference_mode()
    def detect(self, image, prompt, box_threshold, limit,
               *, min_width=0, min_height=0, min_area=0, containment_threshold=0.95):
        from .grounded import clean_proposals
        with self.lock:
            labels = self.validate_prompt(prompt)
            features, mask = self._text_features(labels)
            pixels = prepare_sam3_tensor(image, self.device)
            boxes, logits, presence = self.predict_tensors(pixels, features, mask)
            scores = logits.float().sigmoid() * presence.float().sigmoid()
            packed = torch.cat((boxes.float(), scores.unsqueeze(-1)), dim=-1).cpu().numpy()
            boxes, scores = packed[...,:4], packed[...,4]
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                raise RuntimeError('Non-finite SAM3 bbox outputs')
            all_boxes, confidence, names = [], [], []
            for index, label in enumerate(labels):
                for query in np.flatnonzero(scores[index] >= box_threshold):
                    all_boxes.append((np.clip(boxes[index, query], 0, 1) * [*image.size, *image.size]).tolist())
                    confidence.append(float(scores[index, query]))
                    names.append(label)
            return clean_proposals(all_boxes, confidence, names, image.size, limit,
                min_width=min_width, min_height=min_height, min_area=min_area,
                containment_threshold=containment_threshold)
