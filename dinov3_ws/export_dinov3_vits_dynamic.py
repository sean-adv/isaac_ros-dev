"""Export the cached DINOv3 ViT-S/16 backbone with dynamic spatial dimensions."""
import json
from pathlib import Path

import onnx
import torch
from transformers import AutoModel


MODEL_ID = 'facebook/dinov3-vits16-pretrain-lvd1689m'
OUTPUT = Path(__file__).resolve().parent / 'artifacts' / 'tensorrt'


class ExportModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            MODEL_ID, cache_dir=str(OUTPUT.parent / 'huggingface'), local_files_only=True, dtype=torch.float32,
            attn_implementation='eager').eval()

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values, return_dict=False)[0]


def main():
    torch.set_num_threads(8)
    model = ExportModel().eval()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'dinov3_vits16_dynamic.onnx'
    with torch.inference_mode():
        torch.onnx.export(
            model, (torch.randn(1, 3, 448, 448),), str(path),
            input_names=['pixel_values'], output_names=['last_hidden_state'],
            opset_version=18, dynamo=True, external_data=True,
            dynamic_shapes={'pixel_values': {
                2: torch.export.Dim('height', min=32, max=2560),
                3: torch.export.Dim('width', min=32, max=2560)}})
    onnx.checker.check_model(str(path))
    graph = onnx.load(str(path), load_external_data=False)
    dims = graph.graph.input[0].type.tensor_type.shape.dim
    assert dims[2].dim_param and dims[3].dim_param, 'Spatial dimensions must be dynamic'
    metadata = {
        'model_id': MODEL_ID,
        'revision': model.model.config._commit_hash,
        'patch_size': 16,
        'num_register_tokens': 4,
        'hidden_size': 384,
        'input_min': [1, 3, 32, 32],
        'input_opt': [1, 3, 448, 448],
        'input_max': [1, 3, 2560, 2560],
        'precision': 'BF16 with FP32 RoPE/fallback; FP32 I/O',
    }
    (OUTPUT / 'dinov3_vits16_dynamic.json').write_text(
        json.dumps(metadata, indent=2) + '\n')
    print(f'Exported and checked {path}', flush=True)


if __name__ == '__main__':
    main()
