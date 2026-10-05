"""Export the cached case-matching backbone at batch=1, 448x448."""
from pathlib import Path
import json

import onnx
import torch
from transformers import AutoModel


MODEL_ID = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
OUTPUT = Path(__file__).resolve().parent / "artifacts" / "tensorrt"


class ExportModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            local_files_only=True,
            dtype=torch.float32,
        ).eval()

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values, return_dict=False)[0]


def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = ExportModel().eval()
    dummy = torch.randn(1, 3, 448, 448)
    path = OUTPUT / "dinov3_convnext_large.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (dummy,),
            str(path),
            input_names=["pixel_values"],
            output_names=["last_hidden_state"],
            opset_version=18,
            dynamo=True,
            external_data=True,
        )
    onnx.checker.check_model(str(path))
    graph = onnx.load(str(path), load_external_data=False)
    info = {
        "model_id": MODEL_ID,
        "revision": model.model.config._commit_hash,
        "torch": torch.__version__,
        "input_shape": [1, 3, 448, 448],
        "output_shape": [1, 197, 1536],
        "onnx_opsets": {x.domain: x.version for x in graph.opset_import},
        "preprocessing": "External; retain the existing AutoImageProcessor normalization.",
        "output": "Full tokens, including one global token; no L2 normalization.",
    }
    (OUTPUT / "dinov3_convnext_large_export.json").write_text(
        json.dumps(info, indent=2) + "\n")
    print(f"Exported and checked: {path}", flush=True)


if __name__ == "__main__":
    main()
