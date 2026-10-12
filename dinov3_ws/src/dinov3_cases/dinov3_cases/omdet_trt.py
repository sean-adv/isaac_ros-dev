"""OmDet detector with an offline CLIP prompt bank and fixed-shape TensorRT I/O."""
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch


def prompt_key(prompt):
    """Legacy dot-separated UI syntax is a bank selector, never online tokenization."""
    return tuple(part.strip() for part in prompt.strip().split('.') if part.strip())


class OfflinePromptBank:
    def __init__(self, path, metadata):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['prompt_sha256']:
            raise ValueError('Offline OmDet prompt bank checksum mismatch; rebuild the bank and engine together.')
        self.profiles = metadata['profiles']
        self.entries = {}
        count = metadata['class_count']
        with np.load(path, allow_pickle=False) as data:
            for i, profile in enumerate(self.profiles):
                key = prompt_key(profile['prompt'])
                if not key or key in self.entries or len(profile['labels']) != count:
                    raise ValueError('Invalid or duplicate offline OmDet profile')
                features = {name: data[f'{i}_{name}'].copy() for name in ['label_feats', 'task_feats', 'task_mask']}
                for name, shape, dtype in [('label_feats',(count,1,512),np.float32),
                                           ('task_feats',(77,1,512),np.float32), ('task_mask',(1,77),np.int32)]:
                    value = features[name]
                    if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                        raise ValueError(f'Invalid offline OmDet {name}')
                if not np.isin(features['task_mask'], [0, 1]).all() or not features['task_mask'].any():
                    raise ValueError('Invalid OmDet task mask')
                self.entries[key] = (profile, features)

    def resolve(self, prompt):
        try:
            return self.entries[prompt_key(prompt)]
        except (KeyError, AttributeError):
            available = [p['prompt'] for p in self.profiles]
            raise ValueError(f'OmDet uses offline prompts. Available: {available}. Edit config/omdet_prompts.json and run build_omdet.sh to add a prompt.') from None


def prepare_image(image, metadata):
    # Matches official BaseEngine._load_data + detector.normalizer; no letterbox.
    pixels = np.asarray(image.convert('RGB').resize((metadata['input_size'],)*2, Image.Resampling.BILINEAR), dtype=np.float32)
    pixels = (pixels - np.asarray(metadata['mean'], dtype=np.float32)) / np.asarray(metadata['std'], dtype=np.float32)
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0)


def postprocess(boxes, logits, labels, size, threshold, nms_iou, limit):
    from .grounded import clean_proposals
    from torchvision.ops import batched_nms
    if boxes.ndim == 4:
        boxes = boxes[-1]
    if logits.ndim == 4:
        logits = logits[-1]
    boxes, logits = boxes[0].float(), logits[0].float()
    if not torch.isfinite(boxes).all() or not torch.isfinite(logits).all():
        raise RuntimeError('Non-finite OmDet outputs')
    scores = logits.sigmoid().flatten()
    # Official focal-loss postprocess selects top num_queries query/class pairs.
    scores, indices = scores.topk(min(boxes.shape[0], scores.numel()))
    mask = scores > threshold
    scores, indices = scores[mask], indices[mask]
    classes = indices % len(labels)
    selected = boxes[indices // len(labels)]
    xyxy = torch.cat([selected[:, :2] - selected[:, 2:] / 2, selected[:, :2] + selected[:, 2:] / 2], dim=-1)
    xyxy *= xyxy.new_tensor([*size, *size])
    keep = batched_nms(xyxy, scores, classes, nms_iou)
    # Also collapse overlapping synonyms for the existing per-object gallery pipeline.
    return clean_proposals(xyxy[keep].cpu().tolist(), scores[keep].cpu().tolist(),
                           [labels[c] for c in classes[keep].cpu().tolist()], size, nms_iou, limit)


class OmDetDetector:
    def __init__(self, engine_path, device):
        import tensorrt as trt
        from .runtime import GROUNDING_MODEL_ID
        path = Path(engine_path).expanduser().resolve()
        self.metadata = json.loads(path.with_suffix('.json').read_text())
        self.precision = self.metadata['precision']
        if self.metadata['model_id'] != GROUNDING_MODEL_ID:
            raise ValueError('OmDet engine model_id mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata['engine_sha256']:
            raise ValueError('OmDet engine checksum mismatch')
        self.bank = OfflinePromptBank(path.parent / 'prompts.npz', self.metadata)
        self.offline_prompts = [p['prompt'] for p in self.bank.profiles]
        # Compatibility with the node and shared-memory IPC protocol.
        self.model = SimpleNamespace(config=SimpleNamespace(max_text_len=77))
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('OmDet TensorRT requires CUDA')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Could not load OmDet TensorRT engine: {path}')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Could not create OmDet TensorRT context')
        self.inputs = {'img_tensor', 'label_feats', 'task_feats', 'task_mask'}
        self.outputs = {'boxes', 'logits'}
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        if names != self.inputs | self.outputs:
            raise ValueError(f'Unexpected OmDet TensorRT tensors: {names}')
        self.dtypes = {}
        self.buffers = {}
        for name in names:
            expected_mode = trt.TensorIOMode.INPUT if name in self.inputs else trt.TensorIOMode.OUTPUT
            dtype = self.engine.get_tensor_dtype(name)
            if self.engine.get_tensor_mode(name) != expected_mode or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise ValueError(f'Unexpected OmDet interface: {name}')
            if dtype not in (trt.float32, trt.int32):
                raise ValueError(f'Unsupported OmDet tensor dtype: {name}: {dtype}')
            self.dtypes[name] = torch.int32 if dtype == trt.int32 else torch.float32
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d <= 0 for d in shape):
                raise ValueError(f'OmDet requires fixed tensor shapes: {name}: {shape}')
            self.buffers[name] = torch.empty(shape, dtype=self.dtypes[name], device=self.device)
        self.cached = {key: {name: torch.from_numpy(value).to(self.device) for name, value in features.items()}
                       for key, (_, features) in self.bank.entries.items()}
        self.lock = threading.Lock()

    def validate_prompt(self, prompt):
        self.bank.resolve(prompt)

    @torch.inference_mode()
    def infer(self, inputs):
        with self.lock, torch.cuda.device(self.device):
            for name in self.inputs:
                value = inputs[name]
                if tuple(value.shape) != tuple(self.buffers[name].shape):
                    raise ValueError(f'OmDet shape mismatch: {name}')
                self.buffers[name].copy_(value)
            for name, value in self.buffers.items():
                if not self.context.set_tensor_address(name, value.data_ptr()):
                    raise RuntimeError(f'Could not bind OmDet {name}')
            stream = torch.cuda.current_stream(self.device)
            try:
                if not self.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError('OmDet TensorRT inference failed')
            finally:
                stream.synchronize()
            return {name: self.buffers[name].clone() for name in self.outputs}

    def detect(self, image, prompt, box_threshold, text_threshold, nms_iou, limit):
        profile, _ = self.bank.resolve(prompt)
        inputs = dict(self.cached[prompt_key(prompt)], img_tensor=prepare_image(image, self.metadata))
        result = self.infer(inputs)
        # OmDet scores classes directly: text_threshold has no meaning and is retained only for API compatibility.
        return postprocess(result['boxes'], result['logits'], profile['labels'], image.size, box_threshold, nms_iou, limit)
