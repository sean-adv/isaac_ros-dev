"""TensorRT adapter preserving the native-resolution DINO model interface."""
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import torch


ARTIFACTS = Path('/workspaces/isaac_ros-dev/dinov3_ws/artifacts/tensorrt')
DEFAULT_VARIANT = 'vits16'
MODEL_VARIANTS = {
    'vits16': {
        'model_id': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'engine_path': str(ARTIFACTS / 'dinov3_vits16_dynamic_bf16.engine'),
        'patch_size': 16, 'num_register_tokens': 4, 'hidden_size': 384,
        'cache_dir': str(ARTIFACTS.parent / 'huggingface'),
    },
    'vith16plus': {
        'model_id': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
        'engine_path': str(ARTIFACTS / 'dinov3_dynamic_bf16.engine'),
        'patch_size': 16, 'num_register_tokens': 4, 'hidden_size': 1280,
    },
    'vitb16': {
        'model_id': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'engine_path': str(ARTIFACTS / 'dinov3_vitb16_dynamic_bf16.engine'),
        'patch_size': 16, 'num_register_tokens': 4, 'hidden_size': 768,
    },
}
MODEL_ID = MODEL_VARIANTS[DEFAULT_VARIANT]['model_id']
DEFAULT_ENGINE = MODEL_VARIANTS[DEFAULT_VARIANT]['engine_path']


def model_variant(variant):
    try:
        return MODEL_VARIANTS[variant]
    except KeyError as error:
        choices = ', '.join(MODEL_VARIANTS)
        raise ValueError(f'Unknown model variant {variant!r}; choose one of: {choices}.') from error


def model_spec_for_id(model_id):
    for variant, spec in MODEL_VARIANTS.items():
        if spec['model_id'] == model_id:
            return variant, spec
    supported = ', '.join(spec['model_id'] for spec in MODEL_VARIANTS.values())
    raise ValueError(f'Unsupported TRT model_id={model_id!r}; supported: {supported}.')


def validate_shape(shape, minimum, maximum):
    if len(shape) != 4 or shape[:2] != (1, 3):
        raise ValueError(f'TRT requires batch=1 and RGB NCHW; received {shape}.')
    if any(n < low or n > high for n, low, high in zip(shape, minimum, maximum)):
        raise ValueError(f'TRT input {shape} is outside {minimum} .. {maximum}; '
                         'rebuild the engine profile or select inference_backend:=torch.')


class TRTBackbone:
    def __init__(self, engine_path, model_id):
        variant, spec = model_spec_for_id(model_id)
        path = Path(engine_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'TRT engine not found: {path}; run build_dinov3_dynamic_trt.sh.')
        if not torch.cuda.is_available():
            raise RuntimeError('TensorRT requires a CUDA GPU; no CPU fallback is performed.')
        import tensorrt as trt
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.config = SimpleNamespace(
            model_type='dinov3_vit', patch_size=spec['patch_size'],
            num_register_tokens=spec['num_register_tokens'], hidden_size=spec['hidden_size'])
        self.description = f'TensorRT {variant} (FP32 I/O)'
        metadata = path.with_suffix('.json')
        if metadata.is_file():
            info = json.loads(metadata.read_text())
            if info['model_id'] != model_id:
                raise ValueError('Engine metadata model_id mismatch.')
            for key in ('patch_size', 'num_register_tokens', 'hidden_size'):
                if info.get(key) != spec[key]:
                    raise ValueError(f'Engine metadata {key} mismatch.')
            self.description = f"TensorRT {variant} {info['precision']}"
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Could not deserialize TensorRT engine: {path}')
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        if names not in ({'pixel_values', 'last_hidden_state'},
                         {'pixel_values', 'last_hidden_state', 'pooler_output'}):
            raise ValueError('Unexpected DINOv3 TRT input/output names.')
        self.has_pooler = 'pooler_output' in names
        if self.has_pooler and (
                self.engine.get_tensor_mode('pooler_output') != trt.TensorIOMode.OUTPUT or
                self.engine.get_tensor_dtype('pooler_output') != trt.float32 or
                self.engine.get_tensor_format('pooler_output') != trt.TensorFormat.LINEAR or
                tuple(self.engine.get_tensor_shape('pooler_output')) != (1, self.config.hidden_size)):
            raise ValueError('Unexpected TensorRT interface for pooler_output.')
        for name, mode in [('pixel_values', trt.TensorIOMode.INPUT),
                           ('last_hidden_state', trt.TensorIOMode.OUTPUT)]:
            if (self.engine.get_tensor_mode(name) != mode or
                    self.engine.get_tensor_dtype(name) != trt.float32 or
                    self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                raise ValueError(f'Unexpected TensorRT interface for {name}.')
        shape = tuple(self.engine.get_tensor_shape('pixel_values'))
        if -1 in shape:
            low, _, high = self.engine.get_tensor_profile_shape('pixel_values', 0)
            self.minimum, self.maximum = tuple(low), tuple(high)
        else:
            self.minimum = self.maximum = shape
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Could not create TensorRT execution context.')
        self.lock = threading.Lock()

    @torch.inference_mode()
    def __call__(self, pixel_values):
        shape = tuple(pixel_values.shape)
        validate_shape(shape, self.minimum, self.maximum)
        if pixel_values.device != self.device:
            raise ValueError(f'TRT input must be on {self.device}, got {pixel_values.device}.')
        with self.lock, torch.cuda.device(self.device):
            x = pixel_values.to(dtype=torch.float32).contiguous()
            if not self.context.set_input_shape('pixel_values', shape):
                raise RuntimeError(f'TRT rejected input shape {shape}.')
            output_shape = tuple(self.context.get_tensor_shape('last_hidden_state'))
            patch = self.config.patch_size
            prefix = 1 + self.config.num_register_tokens
            expected = (1, (shape[2] // patch) * (shape[3] // patch) + prefix,
                        self.config.hidden_size)
            if output_shape != expected:
                raise RuntimeError(f'TRT output shape {output_shape}, expected {expected}.')
            y = torch.empty(output_shape, device=self.device, dtype=torch.float32)
            bindings = [('pixel_values', x), ('last_hidden_state', y)]
            if self.has_pooler:
                pooler = torch.empty((1, self.config.hidden_size), device=self.device, dtype=torch.float32)
                bindings.append(('pooler_output', pooler))
            for name, tensor in bindings:
                if not self.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f'TRT could not bind {name}.')
            stream = torch.cuda.current_stream(self.device)
            try:
                if not self.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError('TensorRT execution failed.')
            finally:
                # Keep buffers/context alive until GPU completion; return GPU data.
                stream.synchronize()
            return SimpleNamespace(last_hidden_state=y)
