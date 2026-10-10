"""Grounding DINO TensorRT adapter for the retained Wingdzero exports."""

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import torch


class GroundingTRT:
    def __init__(self, engine_path, model_id, device):
        import tensorrt as trt

        path = Path(engine_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Grounding DINO TRT engine not found: {path}; see README.md.")
        info = json.loads(path.with_suffix(".json").read_text())
        if info["model_id"] != model_id:
            raise ValueError("Grounding DINO engine metadata model_id mismatch.")
        self.height, self.width = info["height"], info["width"]
        self.config = SimpleNamespace(max_text_len=info["max_text_len"])
        self.input_format = info["input_format"]
        self.box_output = info.get("box_output", "pred_boxes")
        self.processor_path = info["processor_path"]
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Grounding DINO TensorRT requires CUDA.")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize Grounding DINO engine: {path}")
        self.inputs = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create Grounding DINO TRT context.")
        self.lock = threading.Lock()
        self.dtypes = {
            trt.float32: torch.float32,
            trt.int64: torch.int64,
            trt.int32: torch.int32,
            trt.bool: torch.bool,
        }
        self.stream = torch.cuda.Stream(device=self.device)
        for name in self.inputs + ["logits", self.box_output]:
            if (
                self.engine.get_tensor_dtype(name) not in self.dtypes
                or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
            ):
                raise ValueError(f"Unsupported Grounding DINO TRT tensor: {name}")

    def validate_prompt(self, processor, prompt):
        ids = processor.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if len(ids) > self.config.max_text_len:
            raise ValueError(
                f"Prompt needs {len(ids)} tokens, but this Grounding DINO engine "
                f"supports at most {self.config.max_text_len}; use fewer categories "
                "or rebuild the engine with a longer prompt."
            )
        if len(ids) < 3:
            raise ValueError("Prompt must include at least one category token.")
        return ids

    def prepare_inputs(self, processor, image, prompt):
        ids = self.validate_prompt(processor, prompt)
        if self.input_format == "wingdzero":
            import cv2
            import numpy as np
            from transformers import BatchEncoding

            # The upstream exporter takes external text attention/position masks.
            # Build them for the real tokens; padded tokens only attend themselves.
            count, capacity = len(ids), self.config.max_text_len
            tokenized = processor.tokenizer(
                [prompt], padding="max_length", max_length=capacity, return_tensors="pt"
            )
            masks = torch.eye(capacity, dtype=torch.bool).unsqueeze(0)
            positions = torch.zeros((1, capacity), dtype=torch.int64)
            special = set(processor.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"]))
            previous = 0
            for col, token_id in enumerate(ids):
                if token_id in special:
                    if col not in (0, count - 1):
                        masks[0, previous + 1 : col + 1, previous + 1 : col + 1] = True
                        positions[0, previous + 1 : col + 1] = torch.arange(col - previous)
                    previous = col
            pixels = cv2.resize(np.asarray(image.convert("RGB")), (self.width, self.height))
            pixels = pixels.astype(np.float32) / 255.0
            pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                [0.229, 0.224, 0.225], dtype=np.float32
            )
            return BatchEncoding(
                {
                    "img": torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])),
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"].bool(),
                    "position_ids": positions,
                    "token_type_ids": tokenized["token_type_ids"],
                    "text_token_mask": masks,
                }
            ).to(self.device)
        raise ValueError(f"Unsupported Grounding DINO engine input format: {self.input_format}")

    @torch.inference_mode()
    def __call__(self, **inputs):
        with self.lock, torch.cuda.device(self.device):
            buffers = {}
            for name in self.inputs:
                dtype = self.dtypes[self.engine.get_tensor_dtype(name)]
                tensor = inputs[name].to(device=self.device, dtype=dtype).contiguous()
                expected = tuple(self.engine.get_tensor_shape(name))
                if -1 in expected:
                    low, _, high = self.engine.get_tensor_profile_shape(name, 0)
                    if len(tensor.shape) != len(low) or any(
                        n < lo or n > hi for n, lo, hi in zip(tensor.shape, low, high)
                    ):
                        raise ValueError(
                            f"Grounding TRT {name} shape outside profile {low} .. {high}"
                        )
                    if not self.context.set_input_shape(name, tuple(tensor.shape)):
                        raise RuntimeError(f"Could not set Grounding TRT shape for {name}")
                elif tuple(tensor.shape) != expected:
                    raise ValueError(
                        f"Grounding TRT {name} shape {tuple(tensor.shape)} != {expected}"
                    )
                buffers[name] = tensor
            for name in ["logits", self.box_output]:
                buffers[name] = torch.empty(
                    tuple(self.context.get_tensor_shape(name)),
                    device=self.device,
                    dtype=self.dtypes[self.engine.get_tensor_dtype(name)],
                )
            for name, tensor in buffers.items():
                if not self.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"Could not bind Grounding DINO tensor {name}")
            stream = self.stream
            stream.wait_stream(torch.cuda.current_stream(self.device))
            try:
                if not self.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("Grounding DINO TensorRT execution failed.")
            finally:
                stream.synchronize()
            return SimpleNamespace(logits=buffers["logits"], pred_boxes=buffers[self.box_output])
