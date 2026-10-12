"""Inspect deployed engines and measure warm, batch-one inference without rebuilding."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
from PIL import Image
import tensorrt as trt
import torch

from dinov3_cases.encoder import Encoder
from dinov3_cases.grounded import prepare_object
from dinov3_cases.omdet_trt import OmDetDetector, prepare_image, postprocess, prompt_key
from dinov3_cases.patches import preprocess_image
from dinov3_cases.runtime import DINO_ENGINE, GROUNDING_ENGINE, WORKSPACE


def stats(rows):
    return dict(median_ms=float(np.median(rows)), p95_ms=float(np.percentile(rows, 95)),
                mean_ms=float(np.mean(rows)), samples=len(rows), raw_ms=rows)


def benchmark(fn, samples, warmup):
    for _ in range(warmup):
        fn()
    rows = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        rows.append((time.perf_counter() - start) * 1000)
    return stats(rows)


def engine_info(engine, context, path, output):
    inspector = engine.create_engine_inspector()
    inspector.execution_context = context
    raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    (output / (path.stem + '_layers.json')).write_text(raw + '\n')
    data = json.loads(raw)
    layers = data.get('Layers', []) if isinstance(data, dict) else data
    formats = Counter()
    types = Counter()
    for layer in layers:
        if isinstance(layer, dict):
            types[layer.get('LayerType', 'unavailable')] += 1
            for item in layer.get('Outputs', []):
                formats[item.get('Format/Datatype', 'unavailable')] += 1
    return dict(path=str(path.relative_to(WORKSPACE)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                num_layers=engine.num_layers, profiling_verbosity=str(engine.profiling_verbosity),
                layer_types=dict(types), output_formats=dict(formats),
                io=[dict(name=engine.get_tensor_name(i),
                         dtype=str(engine.get_tensor_dtype(engine.get_tensor_name(i))),
                         shape=list(engine.get_tensor_shape(engine.get_tensor_name(i))))
                    for i in range(engine.num_io_tensors)])


def onnx_info(path):
    model = onnx.load(str(path))
    types = {v.name: v.type.tensor_type.elem_type for v in
             [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    types.update({v.name: v.data_type for v in model.graph.initializer})
    weights, elements, groups = Counter(), Counter(), {}
    for value in model.graph.initializer:
        dtype = onnx.TensorProto.DataType.Name(value.data_type)
        weights[dtype] += 1
        elements[dtype] += int(np.prod(value.dims))
    for node in model.graph.node:
        group = node.name.split('/')[1] if node.name.startswith('/') else 'other'
        row = groups.setdefault(group, dict(ops=Counter(), output_types=Counter()))
        row['ops'][node.op_type] += 1
        for name in node.output:
            row['output_types'][onnx.TensorProto.DataType.Name(types.get(name, 0))] += 1
    return dict(initializer_counts=dict(weights), initializer_elements=dict(elements),
                groups=groups, op_counts=dict(Counter(n.op_type for n in model.graph.node)),
                cast_targets=dict(Counter(onnx.TensorProto.DataType.Name(a.i)
                    for n in model.graph.node if n.op_type == 'Cast'
                    for a in n.attribute if a.name == 'to')),
                accumulation_nodes=sum(n.name.endswith('/fp32_accumulation') for n in model.graph.node))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--image', type=Path, default=WORKSPACE / 'dinov3_cases_data/stomach/1790733215386907348_2.png')
    parser.add_argument('--output', type=Path, default=WORKSPACE / 'dinov3_ws/artifacts/performance/current_precision_latency')
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 1:
        parser.error('samples and warmup must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    image = Image.open(args.image).convert('RGB')
    report = dict(measured_at=datetime.now(timezone.utc).isoformat(), image=str(args.image),
                  image_size=list(image.size), warmup=args.warmup, samples=args.samples,
                  torch_threads=torch.get_num_threads(), torch=torch.__version__,
                  cuda=torch.version.cuda, tensorrt=trt.__version__,
                  gpu=torch.cuda.get_device_name(), engines={}, timings={}, onnx={})
    for precision, path in [('mixed_fp16', GROUNDING_ENGINE),
                            ('fp32', GROUNDING_ENGINE.with_name('omdet_tiny_fp32.engine'))]:
        detector = OmDetDetector(path, 'cuda')
        prompt = detector.offline_prompts[0]
        profile, _ = detector.bank.resolve(prompt)
        pixels = prepare_image(image, detector.metadata)
        inputs = dict(detector.cached[prompt_key(prompt)], img_tensor=pixels.cuda())
        timings = report['timings'][precision] = {}
        timings['prepare_cpu'] = benchmark(lambda: prepare_image(image, detector.metadata), args.samples, args.warmup)
        timings['infer_gpu_resident_wall'] = benchmark(lambda: detector.infer(inputs), args.samples, args.warmup)
        # Time only the TensorRT launch between CUDA events; I/O buffers already bound.
        detector.infer(inputs)
        stream = torch.cuda.current_stream()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gpu_rows = []
        for i in range(args.warmup + args.samples):
            start.record(stream)
            if not detector.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError('TensorRT launch failed')
            end.record(stream)
            end.synchronize()
            if i >= args.warmup:
                gpu_rows.append(start.elapsed_time(end))
        timings['engine_cuda_event'] = stats(gpu_rows)
        result = detector.infer(inputs)
        assert all(torch.isfinite(v).all() for v in result.values())
        timings['postprocess'] = benchmark(lambda: postprocess(result['boxes'], result['logits'], profile['labels'], image.size, .3, .5, 20), args.samples, args.warmup)
        timings['detect_full'] = benchmark(lambda: detector.detect(image, prompt, .3, .5, 20), args.samples, args.warmup)
        proposals = detector.detect(image, prompt, .3, .5, 20)
        report['engines'][precision] = engine_info(detector.engine, detector.context, path, args.output)
        del detector
    encoder = Encoder(DINO_ENGINE)
    roi = prepare_object(image.crop(proposals[0]['bbox']), 224, 16)
    pixels = torch.from_numpy(np.array(roi)).permute(2, 0, 1).cuda()
    batch = preprocess_image(pixels, encoder.processor)
    timings = report['timings']['dinov3'] = {}
    timings['adapter_gpu_resident_wall'] = benchmark(lambda: encoder.model(batch), args.samples, args.warmup)
    # The adapter synchronizes before return; buffers stay live through end-event.
    gpu_rows = []
    for i in range(args.warmup + args.samples):
        start.record(stream)
        result = encoder.model(batch)
        end.record(stream)
        end.synchronize()
        if i >= args.warmup:
            gpu_rows.append(start.elapsed_time(end))
    timings['adapter_cuda_event'] = stats(gpu_rows)
    timings['encode_prepared_roi'] = benchmark(lambda: encoder.encode(roi), args.samples, args.warmup)
    timings['crop_prepare_encode'] = benchmark(lambda: encoder.encode(prepare_object(image.crop(proposals[0]['bbox']), 224, 16)), args.samples, args.warmup)
    assert torch.isfinite(result.last_hidden_state).all()
    report['engines']['dinov3'] = engine_info(encoder.model.engine, encoder.model.context, DINO_ENGINE, args.output)
    report['detections'] = proposals
    for name in ['omdet_tiny.onnx', 'omdet_tiny_fp16.onnx']:
        report['onnx'][name] = onnx_info(GROUNDING_ENGINE.parent / name)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: {n: {a:b for a,b in row.items() if a != 'raw_ms'} for n,row in v.items()} for k,v in report['timings'].items()}, indent=2))
    print(json.dumps(report['engines'], indent=2))


if __name__ == '__main__':
    main()
