"""Download the pinned public ViT-L ONNX and build a static Thor TRT11 engine."""
import argparse
import hashlib
import json
from pathlib import Path
import time

MODEL_ID = 'onnx-community/dinov3-vitl16-pretrain-lvd1689m-ONNX'
REVISION = '9f74690bbbba9d76c4cce9062195fc7c95e23bb5'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'artifacts/dinov3-vitl16-onnx'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_onnx(source):
    """Replace the exporter's two-way sequences with equivalent tensor Split."""
    import onnx
    from onnx import helper, numpy_helper
    model = onnx.load(source, load_external_data=False)
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer
                 if t.data_location != onnx.TensorProto.EXTERNAL}
    sequences = {n.output[0]: n for n in model.graph.node if n.op_type == 'SplitToSequence'}
    consumers = {name: {} for name in sequences}
    for node in model.graph.node:
        if node.op_type == 'SequenceAt':
            if node.input[0] not in sequences:
                raise ValueError('Unexpected SequenceAt source')
            index = int(constants[node.input[1]])
            consumers[node.input[0]][index] = node.output[0]
    nodes = []
    for node in model.graph.node:
        if node.op_type == 'SplitToSequence':
            outputs = consumers[node.output[0]]
            if set(outputs) != {0, 1}:
                raise ValueError('Expected exactly two sequence outputs')
            axis = next(a.i for a in node.attribute if a.name == 'axis')
            nodes.append(helper.make_node('Split', list(node.input),
                         [outputs[0], outputs[1]], axis=axis, name=node.name + '_tensor'))
        elif node.op_type != 'SequenceAt':
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Sequence value_info is no longer valid after replacing these intermediates.
    keep = [v for v in model.graph.value_info if v.name not in sequences]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)
    for dim, size in zip(model.graph.input[0].type.tensor_type.shape.dim, [1,3,224,224]):
        dim.dim_value = size
    target = source.with_name('model-trt.onnx')
    onnx.save(model, target)
    onnx.checker.check_model(str(target))
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'READY').unlink(missing_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=out / 'source',
                      allow_patterns=['config.json', 'preprocessor_config.json',
                                      'onnx/model.onnx', 'onnx/model.onnx_data'])
    import tensorrt as trt
    import torch
    torch.cuda.init()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    source = out / 'source/onnx/model.onnx'
    prepared = prepare_onnx(source)
    if not parser.parse_from_file(str(prepared)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    network.get_input(0).shape = (1, 3, 224, 224)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    start = time.perf_counter()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError('TensorRT engine build failed')
    target = out / 'model.engine'
    temporary = out / 'model.engine.tmp'
    temporary.write_bytes(bytes(engine))
    temporary.replace(target)
    info = dict(model_id=MODEL_ID, revision=REVISION, precision='FP32',
                patch_size=16, num_register_tokens=4, hidden_size=1024,
                input_shape=[1,3,224,224], tensorrt=trt.__version__,
                gpu=torch.cuda.get_device_name(), build_seconds=time.perf_counter()-start,
                engine_sha256=sha256(target),
                source_sha256={f.name:sha256(f) for f in source.parent.glob('model.onnx*')})
    target.with_suffix('.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
