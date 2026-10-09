#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$script_dir/../GroundingDINO-TensorRT-and-ONNX-Inference"
python_bin="$script_dir/../groundingdino_venv/bin/python"
artifact_dir="$script_dir/artifacts/tensorrt"
export HF_HOME="$repo_dir/weights/huggingface"
export PYTHONPATH=""
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
caption="${1:-medicine bottle. medicine box. bottle. box.}"
mkdir -p "$artifact_dir"
cd "$repo_dir"
"$python_bin" -u export_onnx.py --caption "$caption" \
    --output "$artifact_dir/grounding_dino_wingdzero.onnx" \
    > "$artifact_dir/export_grounding_dino_wingdzero.log" 2>&1
"$python_bin" -u export_engine.py --onnx "$artifact_dir/grounding_dino_wingdzero.onnx" \
    --output "$artifact_dir/grounding_dino_wingdzero_fp16.engine" \
    > "$artifact_dir/build_grounding_dino_wingdzero.log" 2>&1
"$python_bin" - "$artifact_dir" "$caption" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import onnx
output = Path(sys.argv[1])
model = onnx.load(str(output / 'grounding_dino_wingdzero.onnx'))
onnx.checker.check_model(model)
shapes = {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in model.graph.input}
with (output / 'grounding_dino_wingdzero_fp16.engine').open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
metadata = {
    'model_id': 'IDEA-Research/grounding-dino-tiny',
    'source_repository': 'https://github.com/wingdzero/GroundingDINO-TensorRT-and-ONNX-Inference',
    'source_revision': '4192fe2', 'input_format': 'wingdzero',
    'processor_path': str(output.parent / 'grounding_dino_wingdzero_processor'),
    'height': shapes['img'][2], 'width': shapes['img'][3],
    'max_text_len': shapes['input_ids'][1], 'dynamic_text': False,
    'precision': 'FP16', 'caption': sys.argv[2], 'box_output': 'boxes',
    'input_shapes': shapes, 'source_onnx': str(output / 'grounding_dino_wingdzero.onnx'),
    'engine_sha256': digest,
}
(output / 'grounding_dino_wingdzero_fp16.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata, indent=2))
PY
