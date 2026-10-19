"""Run original ViT-L Q4 with ORT CUDA; require all MatMulNBits nodes on GPU."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
from PIL import Image

from build_dinov3 import metrics, THRESHOLDS, digest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT/'dinov3_ws/artifacts/dinov3-vitl16-q4/ort_gpu'


def stats(samples):
    return dict(p50_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),
                min_ms=float(np.min(samples)),samples=len(samples))


def profile_summary(path):
    events = json.loads(Path(path).read_text())
    kernels = [e for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider')]
    counts = Counter((e['args']['op_name'],e['args']['provider']) for e in kernels)
    return dict(profile=str(path), nodes=[dict(op=k[0],provider=k[1],events=v)
                                         for k,v in sorted(counts.items())])


def cuda_options(out, name, fixed=True):
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=8
    opts.inter_op_num_threads=1
    opts.enable_profiling=True
    opts.profile_file_prefix=str(out/name)
    if fixed:
        for symbol,size in [('s99',1),('s100',224),('s4',224)]:
            opts.add_free_dimension_override_by_name(symbol,size)
    return opts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--iterations',type=int,default=50)
    p.add_argument('--tensor-split',action='store_true',help='Replace sequences with equivalent Split nodes')
    args=p.parse_args()
    if args.iterations < 1:p.error('--iterations must be positive')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert 'CUDAExecutionProvider' in ort.get_available_providers(),ort.get_available_providers()
    providers=[('CUDAExecutionProvider',{'use_tf32':'0','cudnn_conv_algo_search':'DEFAULT'})]
    # Force a real CUDA kernel with no CPU fallback before loading the large model.
    opts=cuda_options(out,'cuda_smoke',fixed=False)
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.add_session_config_entry('session.disable_cpu_ep_fallback','1')
    graph=helper.make_graph([helper.make_node('Mul',['x','x'],['y'])],'smoke',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT,[1024])],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT,[1024])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',20)],ir_version=10)
    smoke=ort.InferenceSession(model.SerializeToString(),opts,providers=providers)
    smoke.disable_fallback()
    x=np.arange(1024,dtype=np.float32)
    np.testing.assert_array_equal(smoke.run(None,{'x':x})[0],x*x)
    smoke_profile=profile_summary(smoke.end_profiling())
    assert any(n['op']=='Mul' and n['provider']=='CUDAExecutionProvider'
               for n in smoke_profile['nodes']),smoke_profile
    print('CUDA smoke passed',flush=True)
    del smoke
    source=ROOT/'dinov3_ws/artifacts/dinov3-vitl16-q4/source'
    paths={'q4':source/'onnx/model_q4.onnx',
           'fp32':ROOT/'dinov3_ws/artifacts/dinov3-vitl16-onnx/source/onnx/model.onnx'}
    if args.tensor_split:
        from build_dinov3_onnx import prepare_onnx
        paths={variant:prepare_onnx(path) for variant,path in paths.items()}
    sessions={}
    for variant,path in paths.items():
        sessions[variant]=ort.InferenceSession(str(path),cuda_options(out,variant),providers=providers)
        sessions[variant].disable_fallback()
        assert sessions[variant].get_providers()[0]=='CUDAExecutionProvider'
        print(variant,'loaded',flush=True)
    cpu_opts=ort.SessionOptions();cpu_opts.intra_op_num_threads=8;cpu_opts.inter_op_num_threads=1
    cpu=ort.InferenceSession(str(source/'onnx/model_q4.onnx'),cpu_opts,providers=['CPUExecutionProvider'])
    cfg=json.loads((source/'preprocessor_config.json').read_text())
    mean=np.array(cfg['image_mean'],dtype=np.float32).reshape(1,3,1,1)
    std=np.array(cfg['image_std'],dtype=np.float32).reshape(1,3,1,1)
    images=[ROOT/'TensorRT-Model-Connect/families/dinov3/tests/data/test_img.jpeg',
            ROOT/'dinov3_cases_data/stomach/1790733215386907348_2.png',
            ROOT/'dinov3_cases_data/stomach/1790733214842141931_1.png']
    cases=[]
    for path in images:
        with Image.open(path) as image:cases.append((str(path),image.convert('RGB')))
    cases += [('black',Image.new('RGB',(224,224))),
              ('noise',Image.fromarray(np.random.default_rng(0).integers(0,256,(224,224,3),dtype=np.uint8)))]
    rows=[]
    for name,image in cases:
        scale=224/max(image.size)
        width,height=[max(1,round(n*scale)) for n in image.size]
        image=image.resize((width,height),Image.Resampling.LANCZOS)
        pixels=np.pad(np.asarray(image),((0,224-height),(0,224-width),(0,0)),mode='edge')
        x=np.ascontiguousarray((pixels.astype(np.float32).transpose(2,0,1)[None]*cfg['rescale_factor']-mean)/std)
        results={}
        for variant,session in [*sessions.items(),('q4_cpu',cpu)]:
            results[variant]={o.name:v.astype(np.float64) for o,v in
                              zip(session.get_outputs(),session.run(None,{'pixel_values':x}))}
        parity=metrics(results['q4'],results['q4_cpu'],4)
        passed=all(parity[k]<=v if k=='relative_frobenius' else parity[k]>=v for k,v in THRESHOLDS.items())
        row=dict(image=name,q4_gpu_vs_q4_cpu=parity,parity_passed=passed,
                 q4_vs_fp32_gpu=metrics(results['q4'],results['fp32'],4))
        rows.append(row);print(json.dumps(row),flush=True)
    profiles={variant:profile_summary(session.end_profiling()) for variant,session in sessions.items()}
    q4_nodes=profiles['q4']['nodes']
    gpu_nbits=sum(n['events'] for n in q4_nodes if n['op']=='MatMulNBits' and n['provider']=='CUDAExecutionProvider')
    cpu_nbits=sum(n['events'] for n in q4_nodes if n['op']=='MatMulNBits' and n['provider']!='CUDAExecutionProvider')
    expected_nbits=sum(n.op_type=='MatMulNBits' for n in onnx.load(paths['q4'],load_external_data=False).graph.node)
    placement_passed=(gpu_nbits==expected_nbits*len(cases) and cpu_nbits==0)
    timings={}
    for variant,session in sessions.items():
        # Bind inputs and outputs once on CUDA; synchronize every timed call.
        xgpu=ort.OrtValue.ortvalue_from_numpy(x,'cuda',0)
        binding=session.io_binding();binding.bind_ortvalue_input('pixel_values',xgpu)
        buffers=[]
        for name,shape in [('last_hidden_state',[1,201,1024]),('pooler_output',[1,1024])]:
            buf=ort.OrtValue.ortvalue_from_shape_and_type(shape,np.float32,'cuda',0)
            buffers.append(buf);binding.bind_ortvalue_output(name,buf)
        def bound():
            session.run_with_iobinding(binding);binding.synchronize_outputs()
        for _ in range(10):bound()
        # Verify bound results, rather than benchmarking potentially stale buffers.
        expected=session.run(None,{'pixel_values':x})
        actual=binding.copy_outputs_to_cpu()
        for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
        samples=[]
        for _ in range(args.iterations):
            start=time.perf_counter();bound();samples.append((time.perf_counter()-start)*1000)
        resident=stats(samples)
        samples=[]
        for _ in range(args.iterations):
            start=time.perf_counter();session.run(None,{'pixel_values':x});samples.append((time.perf_counter()-start)*1000)
        timings[variant]=dict(cuda_resident=resident,numpy_input_output=stats(samples),warmup=10)
        print(variant,timings[variant],flush=True)
    report=dict(passed=placement_passed and all(r['parity_passed'] for r in rows),
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
                onnxruntime=ort.__version__,available_providers=ort.get_available_providers(),
                provider_options={v:s.get_provider_options() for v,s in sessions.items()},
                cuda_smoke=smoke_profile,profiles=profiles,expected_matmul_nbits=expected_nbits,
                gpu_nbits_events=gpu_nbits,cpu_nbits_events=cpu_nbits,placement_passed=placement_passed,
                checks=rows,latency=timings,thresholds=THRESHOLDS,
                tensor_split=args.tensor_split,input_shape=[1,3,224,224],source_sha256={v:digest(p) for v,p in paths.items()},
                scope=('Equivalent sequence-to-Split rewrite; ' if args.tensor_split else 'Original ONNX files; ') +
                      'session specializes symbolic dimensions to 224. CUDA TF32 off. '
                      'Profiling validates placement separately from timings. CUDA-resident timing excludes '
                      'preprocessing and I/O copies; NumPy timing includes H2D/D2H. '
                      'Q4 CPU parity does not measure recognition accuracy. Deployment unchanged.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['passed'],'Q4 GPU parity or placement failed; see report.json'
    print('PASS',flush=True)


if __name__=='__main__':main()
