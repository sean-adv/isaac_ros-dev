"""Load the official Torch-TensorRT Cat2Real artifact."""
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import torch
from .runtime import MODEL_ID, DINO_ENGINE


def validate_shape(shape, minimum, maximum):
    if len(shape) != 4 or shape[:2] != (1, 3):
        raise ValueError(f'TRT requires batch=1 and RGB NCHW; received {shape}.')
    if any(n < low or n > high for n, low, high in zip(shape, minimum, maximum)):
        raise ValueError(f'TRT input {shape} is outside {minimum} .. {maximum}.')


class TRTBackbone:
    def __init__(self, engine_path=DINO_ENGINE, model_id=MODEL_ID):
        if model_id != MODEL_ID:
            raise ValueError(f'Unsupported TRT model_id={model_id!r}; expected {MODEL_ID}.')
        path = Path(engine_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'TRT engine not found: {path}; run build_models.py --target cat2real.')
        info = json.loads(path.with_suffix('.json').read_text())
        if info['model_id'] != model_id or info['hidden_size'] != 1024 or info['input_shape'] != [1, 3, 384, 384]:
            raise ValueError('Cat2Real Torch-TensorRT metadata mismatch.')
        if info['dtype'] != 'fp32':
            raise ValueError('Cat2Real requires FP32: FP16 produced non-finite outputs on Thor.')
        if not torch.cuda.is_available():
            raise RuntimeError('TensorRT requires a CUDA GPU; no CPU fallback is performed.')
        import torch_tensorrt  # Registers official serialized TensorRT operators.

        self.device = torch.device('cuda', torch.cuda.current_device())
        self.config = SimpleNamespace(model_type='dinov3_vit', patch_size=16,
                                      num_register_tokens=4, hidden_size=1024)
        self.lock = threading.Lock()
        self.scripted = torch.jit.load(str(path), map_location=self.device).eval()
        self.minimum = self.maximum = (1, 3, 384, 384)
        self.description = 'Official Torch-TensorRT Cat2Real ViT-L/16 FP32 (384×384)'

    @torch.inference_mode()
    def __call__(self, pixel_values):
        validate_shape(tuple(pixel_values.shape), self.minimum, self.maximum)
        if pixel_values.device != self.device:
            raise ValueError(f'TRT input must be on {self.device}, got {pixel_values.device}.')
        with self.lock, torch.cuda.device(self.device):
            y = self.scripted(pixel_values.float().contiguous())
            if tuple(y.shape) != (1, 581, 1024):
                raise RuntimeError(f'Unexpected Cat2Real output shape: {tuple(y.shape)}')
            if not torch.isfinite(y).all():
                raise RuntimeError('Non-finite Cat2Real features')
            return SimpleNamespace(last_hidden_state=y, pooler_output=y[:, 0])
