"""Rebuild migrated SAM3 DART W8A8 artifacts with TensorRT 11."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent


def main():
    import tensorrt as trt
    import onnx
    from onnxconverter_common import float16

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/sam3/bbox_w8a8')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/sam3/trt11')
    parser.add_argument('--stage', choices=['vision', 'heads', 'all'], default='all')
    args = parser.parse_args()
    if trt.__version__.split('.')[0] != '11':
        parser.error('Run with dinov3_trt11_venv (TensorRT 11).')
    args.output.mkdir(parents=True, exist_ok=True)
    plugin = ctypes.CDLL(str(args.output / 'plugins/lq_plugins.so'), mode=ctypes.RTLD_GLOBAL)
    plugin.lq_register()
    for name in ['text_c16.onnx', 'bpe_simple_vocab_16e6.txt.gz', 'img_pos_c1.npy', 'img_pos_c4.npy']:
        shutil.copy2(args.source / name, args.output / name)
    # ONNX text weights may be external.
    for path in args.source.glob('text_c16.onnx.*'):
        shutil.copy2(path, args.output / path.name)
    logger = trt.Logger(trt.Logger.WARNING)
    names = (['vision_int8'] if args.stage == 'vision' else
             ['ground_c1m_fp16', 'ground_c4m_fp16'] if args.stage == 'heads' else
             ['vision_int8', 'ground_c1m_fp16', 'ground_c4m_fp16'])
    for name in names:
        start = time.monotonic()
        source = args.source / (name.replace('_fp16', '') + '.onnx')
        if name.startswith('vision'):
            model = onnx.load(str(source))
            nodes = []
            # TRT10 inserted these plugin boundary casts implicitly. TRT11 requires them.
            for node in model.graph.node:
                if node.op_type.startswith('LQ'):
                    for i, value in enumerate(node.input):
                        cast = f'{node.name}_input_{i}_half'
                        nodes.append(onnx.helper.make_node('Cast', [value], [cast], to=onnx.TensorProto.FLOAT16))
                        node.input[i] = cast
                    nodes.append(node)
                    for i, value in enumerate(list(node.output)):
                        raw = f'{node.name}_output_{i}_half'
                        node.output[i] = raw
                        nodes.append(onnx.helper.make_node('Cast', [raw], [value], to=onnx.TensorProto.FLOAT))
                else:
                    nodes.append(node)
            del model.graph.node[:]
            model.graph.node.extend(nodes)
            source = args.output / f'{name}.onnx'
            source.with_suffix('.onnx.data').unlink(missing_ok=True)
            onnx.save_model(model, str(source), save_as_external_data=True,
                            all_tensors_to_one_file=True, location=f'{name}.onnx.data')
        if name.startswith('ground'):
            # Strong typing replaces TRT10 FP16 flags. Preserve sensitive reductions.
            model = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
            source_types = {v.name: v.type.tensor_type.elem_type
                            for v in [*model.graph.input, *model.graph.output, *model.graph.value_info]}
            # Normalize the source's HALF I/O wrappers to FLOAT first. Mixed source
            # Cast nodes otherwise survive conversion with inconsistent type metadata.
            for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
                if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
                    value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            def promote(tensor):
                if tensor.data_type == onnx.TensorProto.FLOAT16:
                    array = onnx.numpy_helper.to_array(tensor).astype('float32')
                    tensor.CopyFrom(onnx.numpy_helper.from_array(array, tensor.name))
            for tensor in model.graph.initializer:
                promote(tensor)
            for node in model.graph.node:
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.TENSOR:
                        promote(attr.t)
                if node.op_type == 'Cast' and source_types.get(node.input[0]) in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16) and any(
                        attr.name == 'to' and attr.i in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)
                        for attr in node.attribute):
                    node.op_type = 'Identity'
                    del node.attribute[:]
            float_casts = [node.name for node in model.graph.node if node.op_type == 'Cast']
            model = float16.convert_float_to_float16(
                model, keep_io_types=True, check_fp16_ready=False,
                node_block_list=float_casts,
                op_block_list=float16.DEFAULT_OP_BLOCK_LIST +
                ['LayerNormalization', 'Softmax', 'ReduceMean', 'ReduceSum', 'Pow', 'Sqrt'])
            source = args.output / f'{name}.onnx'
            onnx.save(model, str(source))
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        reader = trt.OnnxParser(network, logger)
        if not reader.parse_from_file(str(source)):
            raise RuntimeError('\n'.join(str(reader.get_error(i)) for i in range(reader.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
        print(f'Building {name}: {network.num_layers} layers', flush=True)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(f'Build failed: {name}')
        path = args.output / f'{name}.plan'
        pending = path.with_suffix('.plan.tmp')
        pending.write_bytes(bytes(plan))
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(pending.read_bytes())
        if engine is None:
            raise RuntimeError(f'Could not deserialize {pending}')
        info = dict(tensorrt=trt.__version__, source=str(source),
                    sha256=hashlib.sha256(pending.read_bytes()).hexdigest(),
                    build_seconds=time.monotonic()-start,
                    io=[dict(name=engine.get_tensor_name(i),
                             shape=list(engine.get_tensor_shape(engine.get_tensor_name(i))),
                             dtype=str(engine.get_tensor_dtype(engine.get_tensor_name(i))))
                        for i in range(engine.num_io_tensors)])
        pending.replace(path)
        path.with_suffix('.json').write_text(json.dumps(info, indent=2)+'\n')
        print(f'Built {path} in {info["build_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
