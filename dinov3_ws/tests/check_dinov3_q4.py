"""Probe the upstream Q4 model in TRT and measure ORT CPU quantization error.

Run with PYTHONPATH=dinov3_ws:dinov3_ws/src/dinov3_cases and the TRT11 venv.
This experiment never changes the deployed engine or writes a READY marker.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from build_dinov3 import cosine, digest, metrics, THRESHOLDS
from build_dinov3_onnx import DEFAULT_OUTPUT, MODEL_ID, REVISION, prepare_onnx
from dinov3_cases.grounded import prepare_object


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT.parent/'dinov3-vitl16-q4')
    p.add_argument('--fp32-source', type=Path, default=DEFAULT_OUTPUT/'source')
    args = p.parse_args()
    out = args.output.resolve()
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=out/'source',
                      allow_patterns=['config.json','preprocessor_config.json',
                                      'onnx/model_q4.onnx','onnx/model_q4.onnx_data'])
    source = out/'source/onnx/model_q4.onnx'
    graph = onnx.load(source, load_external_data=False)
    counts = Counter(n.op_type for n in graph.graph.node)
    blocks = sorted({a.i for n in graph.graph.node if n.op_type=='MatMulNBits'
                     for a in n.attribute if a.name=='block_size'})
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse_from_file(str(prepare_onnx(source)))
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    (out/'trt_errors.txt').write_text('\n'.join(errors)+'\n')
    print(f'TRT parsed={parsed}; MatMulNBits={counts["MatMulNBits"]}; block_size={blocks}', flush=True)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    opts.inter_op_num_threads = 1
    sessions = {
        'q4': ort.InferenceSession(str(source),opts,providers=['CPUExecutionProvider']),
        'fp32': ort.InferenceSession(str(args.fp32_source/'onnx/model.onnx'),opts,
                                    providers=['CPUExecutionProvider'])}
    root = Path(__file__).resolve().parents[2]
    paths = [root/'TensorRT-Model-Connect/families/dinov3/tests/data/test_img.jpeg',
             root/'dinov3_cases_data/stomach/1790733215386907348_2.png',
             root/'dinov3_cases_data/stomach/1790733214842141931_1.png']
    cases = []
    for path in paths:
        with Image.open(path) as image:
            cases.append((str(path), prepare_object(image.convert('RGB'), 224)))
    cases += [('black',Image.new('RGB',(224,224))),
              ('noise',Image.fromarray(np.random.default_rng(0).integers(0,256,(224,224,3),dtype=np.uint8)))]
    cfg = json.loads((out/'source/preprocessor_config.json').read_text())
    mean = np.array(cfg['image_mean'],dtype=np.float32).reshape(1,3,1,1)
    std = np.array(cfg['image_std'],dtype=np.float32).reshape(1,3,1,1)
    rows = []
    for name, image in cases:
        x = np.asarray(image,dtype=np.float32).transpose(2,0,1)[None].copy()
        x = np.ascontiguousarray((x*cfg['rescale_factor']-mean)/std)
        results = {}
        for variant,session in sessions.items():
            results[variant] = {o.name:v.astype(np.float64) for o,v in
                               zip(session.get_outputs(),session.run(None,{'pixel_values':x}))}
        values = metrics(results['q4'],results['fp32'],4)
        descriptors = {}
        for variant,result in results.items():
            patch = result['last_hidden_state'][:,5:]
            patch /= np.maximum(np.linalg.norm(patch,axis=-1,keepdims=True),1e-30)
            descriptors[variant] = patch.mean(axis=1)
        rows.append(dict(image=name,metrics=values,
                         mean_patch_descriptor_cosine=cosine(descriptors['q4'],descriptors['fp32']),
                         meets_fp32_conversion_thresholds=all(
                             values[k]<=v if k=='relative_frobenius' else values[k]>=v
                             for k,v in THRESHOLDS.items())))
        print(json.dumps(rows[-1]),flush=True)
    timings = {}
    for variant,session in sessions.items():
        for _ in range(3): session.run(None,{'pixel_values':x})
        samples=[]
        for _ in range(10):
            start=time.perf_counter()
            session.run(None,{'pixel_values':x})
            samples.append((time.perf_counter()-start)*1000)
        timings[variant]=dict(p50_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),
                             warmup=3,samples=10)
    sizes={}
    for variant,folder,filename in [('q4',source.parent,'model_q4.onnx'),
                                  ('fp32',args.fp32_source/'onnx','model.onnx')]:
        sizes[variant]=sum((folder/name).stat().st_size for name in [filename,filename+'_data'])
    report=dict(model_id=MODEL_ID,revision=REVISION,tensorrt=trt.__version__,
                trt_parse_passed=parsed,trt_error_count=len(errors),first_trt_error=errors[:1],
                matmul_nbits=counts['MatMulNBits'],block_sizes=blocks,
                sizes_bytes=sizes,compression_ratio=sizes['fp32']/sizes['q4'],
                q4_sha256={f.name:digest(f) for f in source.parent.glob('model_q4.onnx*')},
                checks=rows,ort_cpu_latency=timings,onnxruntime=ort.__version__,
                available_providers=ort.get_available_providers(),
                scope='Original Q4 vs original FP32, ORT CPU, 8 threads; no GPU latency comparison. '
                      'FP32 thresholds are reported for context, not a Q4 acceptance criterion. '
                      'Parse probe applies only sequence-to-Split compatibility rewrite. '
                      'No TRT engine built or deployment changed.')
    (out/'q4_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(sizes_bytes=sizes,compression_ratio=report['compression_ratio'],
                          ort_cpu_latency=timings)),flush=True)


if __name__ == '__main__':
    main()
