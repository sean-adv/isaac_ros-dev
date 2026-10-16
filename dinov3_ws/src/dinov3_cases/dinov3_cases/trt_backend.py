"""TensorRT adapters for DINOv3 ViT-B/16 and legacy ViT-S/16 engines."""

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import torch
from .runtime import MODEL_ID, DINO_ENGINE, INPUT_SIZE


MODEL_SPECS = {
    MODEL_ID: dict(patch_size=16, num_register_tokens=4, hidden_size=768),
    "facebook/dinov3-vits16-pretrain-lvd1689m":
        dict(patch_size=16, num_register_tokens=4, hidden_size=384),
}


def validate_shape(shape, minimum, maximum):
    if len(shape) != 4 or shape[:2] != (1, 3):
        raise ValueError(f"TRT requires batch=1 and RGB NCHW; received {shape}.")
    if any(n < low or n > high for n, low, high in zip(shape, minimum, maximum)):
        raise ValueError(
            f"TRT input {shape} is outside {minimum} .. {maximum}; "
            "use an input size supported by this engine; see BUILD_GUIDE.md."
        )


class TRTBackbone:
    def __init__(self, engine_path=DINO_ENGINE, model_id=MODEL_ID):
        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unsupported TRT model_id={model_id!r}; expected {MODEL_ID}.")
        spec = MODEL_SPECS[model_id]
        path = Path(engine_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"TRT engine not found: {path}; see BUILD_GUIDE.md.")
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT requires a CUDA GPU; no CPU fallback is performed.")
        import tensorrt as trt

        self.device = torch.device("cuda", torch.cuda.current_device())
        self.config = SimpleNamespace(
            model_type="dinov3_vit",
            patch_size=spec["patch_size"],
            num_register_tokens=spec["num_register_tokens"],
            hidden_size=spec["hidden_size"],
        )
        self.lock = threading.Lock()
        self.scripted = None
        variant = "vitb16" if spec["hidden_size"] == 768 else "vits16"
        if path.suffix == ".ts":
            if model_id != MODEL_ID:
                raise ValueError("The Torch-TensorRT artifact requires the ViT-B model_id.")
            import torch_tensorrt  # Registers serialized TensorRT TorchScript operators.

            self.scripted = torch.jit.load(str(path), map_location=self.device).eval()
            self.minimum = self.maximum = (1, 3, INPUT_SIZE, INPUT_SIZE)
            self.description = f"Torch-TensorRT vitb16 FP16 ({INPUT_SIZE}×{INPUT_SIZE})"
            return
        self.description = f"TensorRT {variant} (FP32 I/O)"
        metadata = path.with_suffix(".json")
        if metadata.is_file():
            info = json.loads(metadata.read_text())
            if info["model_id"] != model_id:
                raise ValueError("Engine metadata model_id mismatch.")
            for key in ("patch_size", "num_register_tokens", "hidden_size"):
                if info.get(key) != spec[key]:
                    raise ValueError(f"Engine metadata {key} mismatch.")
            self.description = f"TensorRT {variant} {info['precision']}"
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {path}")
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        if names not in (
            {"pixel_values", "last_hidden_state"},
            {"pixel_values", "last_hidden_state", "pooler_output"},
        ):
            raise ValueError("Unexpected DINOv3 TRT input/output names.")
        self.has_pooler = "pooler_output" in names
        if self.has_pooler and (
            self.engine.get_tensor_mode("pooler_output") != trt.TensorIOMode.OUTPUT
            or self.engine.get_tensor_dtype("pooler_output") != trt.float32
            or self.engine.get_tensor_format("pooler_output") != trt.TensorFormat.LINEAR
            or tuple(self.engine.get_tensor_shape("pooler_output")) != (1, self.config.hidden_size)
        ):
            raise ValueError("Unexpected TensorRT interface for pooler_output.")
        for name, mode in [
            ("pixel_values", trt.TensorIOMode.INPUT),
            ("last_hidden_state", trt.TensorIOMode.OUTPUT),
        ]:
            if (
                self.engine.get_tensor_mode(name) != mode
                or self.engine.get_tensor_dtype(name) != trt.float32
                or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
            ):
                raise ValueError(f"Unexpected TensorRT interface for {name}.")
        shape = tuple(self.engine.get_tensor_shape("pixel_values"))
        if -1 in shape:
            low, _, high = self.engine.get_tensor_profile_shape("pixel_values", 0)
            self.minimum, self.maximum = tuple(low), tuple(high)
        else:
            self.minimum = self.maximum = shape
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context.")

    @torch.inference_mode()
    def __call__(self, pixel_values):
        shape = tuple(pixel_values.shape)
        validate_shape(shape, self.minimum, self.maximum)
        if pixel_values.device != self.device:
            raise ValueError(f"TRT input must be on {self.device}, got {pixel_values.device}.")
        with self.lock, torch.cuda.device(self.device):
            if self.scripted is not None:
                x = pixel_values.to(dtype=torch.float16).contiguous()
                y = self.scripted(x)
                torch.cuda.current_stream(self.device).synchronize()
                expected = (
                    1,
                    (shape[2] // self.config.patch_size) * (shape[3] // self.config.patch_size)
                    + 1 + self.config.num_register_tokens,
                    self.config.hidden_size,
                )
                if tuple(y.shape) != expected:
                    raise RuntimeError(f"TRT output shape {tuple(y.shape)}, expected {expected}.")
                return SimpleNamespace(last_hidden_state=y)
            x = pixel_values.to(dtype=torch.float32).contiguous()
            if not self.context.set_input_shape("pixel_values", shape):
                raise RuntimeError(f"TRT rejected input shape {shape}.")
            output_shape = tuple(self.context.get_tensor_shape("last_hidden_state"))
            patch = self.config.patch_size
            prefix = 1 + self.config.num_register_tokens
            expected = (
                1,
                (shape[2] // patch) * (shape[3] // patch) + prefix,
                self.config.hidden_size,
            )
            if output_shape != expected:
                raise RuntimeError(f"TRT output shape {output_shape}, expected {expected}.")
            y = torch.empty(output_shape, device=self.device, dtype=torch.float32)
            bindings = [("pixel_values", x), ("last_hidden_state", y)]
            if self.has_pooler:
                pooler = torch.empty(
                    (1, self.config.hidden_size), device=self.device, dtype=torch.float32
                )
                bindings.append(("pooler_output", pooler))
            for name, tensor in bindings:
                if not self.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"TRT could not bind {name}.")
            stream = torch.cuda.current_stream(self.device)
            try:
                if not self.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT execution failed.")
            finally:
                # Keep buffers/context alive until GPU completion; return GPU data.
                stream.synchronize()
            return SimpleNamespace(last_hidden_state=y)
