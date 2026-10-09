"""Prepare pinned community ONNX for TRT; replace dynamic EyeLike equivalently."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
import onnx
from onnx import helper, numpy_helper
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'artifacts/tensorrt'
MODEL_ID = 'onnx-community/grounding-dino-tiny-ONNX'
REVISION = 'ff690b0a8050566c290287545bd059350f3e9096'
CACHE = ROOT / 'artifacts/huggingface'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    snapshot = Path(snapshot_download(
        MODEL_ID, revision=REVISION, cache_dir=str(CACHE),
        allow_patterns=['*.json', '*.txt', 'onnx/model.onnx']))
    source = snapshot / 'onnx/model.onnx'
    onnx.checker.check_model(str(source))
    graph = onnx.load(str(source), load_external_data=False)
    input_names = [value.name for value in graph.graph.input]
    assert set(input_names) == {'pixel_values', 'pixel_mask', 'input_ids', 'attention_mask', 'token_type_ids'}
    assert [value.name for value in graph.graph.output] == ['logits', 'pred_boxes']
    tensors = {value.name: value.type.tensor_type for value in graph.graph.input}
    image_shape = [dim.dim_value for dim in tensors['pixel_values'].shape.dim]
    assert image_shape == [1, 3, 800, 800]
    assert tensors['input_ids'].shape.dim[1].dim_param
    # TRT 10.16 cannot import EyeLike with dynamic dimensions. Build an
    # equivalent diagonal from row/column indices, preserving dtype and k.
    producers = {out: node for node in graph.graph.node for out in node.output}
    nodes, patched = [], []
    for node in graph.graph.node:
        if node.op_type != 'EyeLike':
            nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        producer = producers[node.input[0]]
        assert producer.op_type == 'ConstantOfShape'
        dtype = attrs.get('dtype', next(a.t.data_type for a in producer.attribute if a.name == 'value'))
        prefix = node.name + '_trt'
        def constant(name, value):
            tensor = numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=prefix + name)
            graph.graph.initializer.append(tensor)
            return tensor.name
        zero, one = constant('_zero', 0), constant('_one', 1)
        axis0, axis1 = constant('_axis0', [0]), constant('_axis1', [1])
        nodes.extend([
            helper.make_node('Shape', [node.input[0]], [prefix+'_shape'], name=prefix+'_shape'),
            helper.make_node('Gather', [prefix+'_shape', zero], [prefix+'_rows'], axis=0, name=prefix+'_rows'),
            helper.make_node('Gather', [prefix+'_shape', one], [prefix+'_cols'], axis=0, name=prefix+'_cols'),
            helper.make_node('Range', [zero, prefix+'_rows', one], [prefix+'_r'], name=prefix+'_r'),
            helper.make_node('Range', [zero, prefix+'_cols', one], [prefix+'_c'], name=prefix+'_c'),
            helper.make_node('Unsqueeze', [prefix+'_r', axis1], [prefix+'_rr'], name=prefix+'_rr'),
            helper.make_node('Unsqueeze', [prefix+'_c', axis0], [prefix+'_cc'], name=prefix+'_cc'),
        ])
        row = prefix+'_rr'
        if attrs.get('k', 0):
            offset = constant('_offset', attrs['k'])
            nodes.append(helper.make_node('Add', [row, offset], [prefix+'_shifted'], name=prefix+'_shifted'))
            row = prefix+'_shifted'
        nodes.extend([
            helper.make_node('Equal', [row, prefix+'_cc'], [prefix+'_equal'], name=prefix+'_equal'),
            helper.make_node('Cast', [prefix+'_equal'], list(node.output), to=dtype, name=prefix+'_cast'),
        ])
        patched.append(node.name)
    assert len(patched) == 1
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    target = OUTPUT / 'grounding_dino_community.onnx'
    if target.is_symlink():
        target.unlink()
    onnx.save(graph, str(target))
    onnx.checker.check_model(str(target))
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata = {
        'model_id': MODEL_ID, 'revision': REVISION,
        'source_file': 'onnx/model.onnx', 'source_onnx': str(source),
        'source_sha256': digest,
        'compatibility_patches': {'EyeLike_to_Range_Equal_Cast': patched},
        'processor_path': str(snapshot), 'height': 800, 'width': 800,
        'max_text_len': 256, 'dynamic_text': True,
        'precision': 'FP32, TF32 disabled', 'input_names': input_names,
        'text_profile': {'min': 3, 'opt': 16, 'max': 256},
    }
    (OUTPUT / 'grounding_dino_community.source.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Prepared community ONNX: {target}\nSource SHA256: {metadata["source_sha256"]}', flush=True)


if __name__ == '__main__':
    main()
