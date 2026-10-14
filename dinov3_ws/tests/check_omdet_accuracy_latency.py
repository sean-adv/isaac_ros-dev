"""Compare OmDet mixed FP16/FP32 to trained PyTorch and time both engine and detect paths."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from scipy.optimize import linear_sum_assignment
from dinov3_cases.omdet_trt import OmDetDetector, prepare_image, postprocess
from check_omdet_fp16 import compare_detections

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def raw_errors(actual, expected):
    a = {n:v.detach().cpu().numpy().reshape(-1,v.shape[-1]).astype(np.float64) for n,v in actual.items()}
    e = {n:v.detach().cpu().numpy().reshape(-1,v.shape[-1]).astype(np.float64) for n,v in expected.items()}
    cost = np.abs(e['boxes'][:,None] - a['boxes'][None]).max(-1)
    cost += .05 * np.abs(e['logits'][:,None] - a['logits'][None]).max(-1)
    rows,cols = linear_sum_assignment(cost)
    # Pair queries because Top-K may reorder or select different near-tied anchors.
    return {n:dict(max_abs=float(np.abs(a[n][cols]-e[n][rows]).max()),
                   mean_abs=float(np.abs(a[n][cols]-e[n][rows]).mean())) for n in a}


class EngineRunner:
    def __init__(self, detector):
        self.detector = detector
        self.stream = torch.cuda.Stream()
        for name,value in detector.buffers.items():
            if not detector.context.set_tensor_address(name,value.data_ptr()):
                raise RuntimeError(f'Failed to bind {name}')

    def load(self, inputs):
        for name,value in inputs.items():
            self.detector.buffers[name].copy_(value)
        torch.cuda.synchronize()

    def run(self):
        if not self.detector.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT inference failed')
        return {n:self.detector.buffers[n] for n in self.detector.outputs}


def benchmark(fn, warmup, iterations, stream=None):
    for _ in range(warmup):
        result=fn()
        torch.cuda.synchronize()
        del result
    wall,gpu=[],[]
    for _ in range(iterations):
        torch.cuda.synchronize()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        t=time.perf_counter()
        if stream is not None:
            start.record(stream)
        result=fn()
        if stream is not None:
            end.record(stream)
        torch.cuda.synchronize()
        wall.append((time.perf_counter()-t)*1000)
        if stream is not None:
            gpu.append(start.elapsed_time(end))
        del result
    def summary(values):
        return dict(p50_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),samples_ms=values)
    return dict(wall=summary(wall),cuda_event=summary(gpu) if gpu else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=ROOT/'artifacts/omdet')
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--iterations',type=int,default=100)
    args=parser.parse_args()
    if args.warmup < 1 or args.iterations < 10:
        parser.error('warmup >= 1 and iterations >= 10 required')
    output=args.artifacts.resolve()
    if not (output/'omdet_tiny_fp32.engine').exists():
        parser.error('Missing FP32 baseline. Run build_omdet.sh --stage engine --precision fp32 for these artifacts first.')
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    fp16=OmDetDetector(output/'omdet_tiny.engine','cuda')
    fp32=OmDetDetector(output/'omdet_tiny_fp32.engine','cuda')
    if fp16.metadata['precision'] != 'FP16' or fp32.metadata['precision'] != 'FP32':
        raise ValueError('Expected FP16 and FP32 engines')
    for key in ['model_id','source_commit','prompt_sha256','max_labels','max_prompt_tokens','profiles','mean','std','input_size']:
        if fp16.metadata[key] != fp32.metadata[key]:
            raise ValueError(f'Baseline metadata differs in {key}')
    for engine in [fp16,fp32]:
        for name in engine.inputs|engine.outputs:
            if engine.dtypes[name] != fp16.dtypes[name]:
                raise ValueError(f'Engine I/O dtype mismatch: {name}')
    repo=WORKSPACE/'OmDet'
    source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if source_commit != fp16.metadata['source_commit']:
        raise ValueError('PyTorch source differs from exported engine source')
    from detectron2.config import get_cfg
    from omdet.omdet_v2_turbo.config import add_omdet_v2_turbo_config
    from omdet.omdet_v2_turbo.infer_model import OmDetV2TurboInfer
    cfg=get_cfg()
    add_omdet_v2_turbo_config(cfg)
    cfg.merge_from_file(str(repo/'configs/OmDet-Turbo_tiny_SWIN_T.yaml'))
    cfg.MODEL.DEVICE='cuda'
    cfg.MODEL.META_ARCHITECTURE='OmDetV2TurboInfer'
    cfg.freeze()
    model=OmDetV2TurboInfer(cfg).cuda().float().eval()
    checkpoint_path=repo/'resources/OmDet-Turbo_tiny_SWIN_T.pth'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    missing,unexpected=model.load_state_dict(checkpoint.get('model',checkpoint),strict=False)
    if missing or any(not n.startswith('language_backbone.') for n in unexpected):
        raise ValueError(f'Checkpoint mismatch: {missing}, {unexpected}')
    paths=[WORKSPACE/'dinov3_cases_data/stomach'/n for n in [
        '1790733215386907348_2.png','1790733214842141931_1.png','1790733216258269633_3.png']]
    frames=[(str(p),Image.open(p).convert('RGB')) for p in paths]
    frames.append((str(paths[-1])+' rotated 90',frames[-1][1].rotate(90,expand=True)))
    checks=[]
    all_passed=True
    with torch.inference_mode():
        for profile in fp16.bank.profiles:
            prompt=profile['prompt']
            _,features=fp16.bank.resolve(prompt)
            cuda_features={n:torch.from_numpy(v).cuda() for n,v in features.items()}
            for name,image in frames:
                cpu_inputs=dict(img_tensor=prepare_image(image,fp16.metadata),**{n:torch.from_numpy(v) for n,v in features.items()})
                cuda_inputs={n:v.cuda() for n,v in cpu_inputs.items()}
                boxes,logits=model(*(cuda_inputs[n] for n in ['img_tensor','label_feats','task_feats','task_mask']))
                expected=dict(boxes=boxes,logits=logits)
                reference=postprocess(boxes,logits,profile['labels'],image.size,.3,.5,20)
                case=dict(image=name,prompt=prompt,pytorch_detections=reference,comparisons={})
                predictions={}
                for label,detector in [('tensorrt_fp16',fp16),('tensorrt_fp32',fp32)]:
                    actual=detector.infer(cuda_inputs)
                    if not all(torch.isfinite(v).all() for v in actual.values()):
                        raise RuntimeError('Non-finite engine outputs')
                    detections=postprocess(actual['boxes'],actual['logits'],profile['labels'],image.size,.3,.5,20)
                    predictions[label]=detections
                    try:
                        parity=compare_detections(detections,reference)
                        passed=True
                    except AssertionError as error:
                        parity=dict(error=str(error))
                        passed=False
                        all_passed=False
                    case['comparisons'][label]=dict(passed=passed,detection_metrics=parity,
                        query_matched_raw_errors=raw_errors(actual,expected),detections=detections)
                try:
                    case['fp16_vs_fp32']=dict(passed=True,**compare_detections(predictions['tensorrt_fp16'],predictions['tensorrt_fp32']))
                except AssertionError as error:
                    case['fp16_vs_fp32']=dict(passed=False,error=str(error));all_passed=False
                checks.append(case)
                print(f"Checked {name}: {prompt}",flush=True)
        if not any(c['pytorch_detections'] for c in checks):
            all_passed=False
        image=frames[0][1]
        profile=fp16.bank.profiles[0]
        prompt=profile['prompt']
        _,features=fp16.bank.resolve(prompt)
        cuda_inputs=dict(img_tensor=prepare_image(image,fp16.metadata).cuda(),**{n:torch.from_numpy(v).cuda() for n,v in features.items()})
        def reference_forward():
            return model(*(cuda_inputs[n] for n in ['img_tensor','label_feats','task_feats','task_mask']))
        runners={n:EngineRunner(d) for n,d in [('tensorrt_fp16',fp16),('tensorrt_fp32',fp32)]}
        timings={}
        for name,runner in runners.items():
            runner.load(cuda_inputs)
            timings[name]=benchmark(runner.run,args.warmup,args.iterations,runner.stream)
        timings['pytorch_fp32']=benchmark(reference_forward,args.warmup,args.iterations,torch.cuda.current_stream())
        detect_timings={n:benchmark(lambda d=d:d.detect(image,prompt,.3,.5,20),args.warmup,args.iterations)
                        for n,d in [('tensorrt_fp16',fp16),('tensorrt_fp32',fp32)]}
        def pytorch_detect():
            pixels=prepare_image(image,fp16.metadata).cuda()
            boxes,logits=model(pixels,cuda_inputs['label_feats'],cuda_inputs['task_feats'],cuda_inputs['task_mask'])
            return postprocess(boxes,logits,profile['labels'],image.size,.3,.5,20)
        detect_timings['pytorch_fp32']=benchmark(pytorch_detect,args.warmup,args.iterations)
    report=dict(passed=all_passed,model_id=fp16.metadata['model_id'],source_commit=source_commit,
        checkpoint_sha256=sha(checkpoint_path),engine_sha256={n:sha(output/p) for n,p in [('fp16','omdet_tiny.engine'),('fp32','omdet_tiny_fp32.engine')]},
        prompt_sha256=sha(output/'prompts.npz'),onnx_sha256=sha(output/'omdet_tiny.onnx'),
        input_shape=[1,3,640,640],max_labels=fp16.metadata['max_labels'],max_prompt_tokens=fp16.metadata['max_prompt_tokens'],
        tolerances=dict(exact_labels_and_count=True,min_box_iou=.98,max_score_error=.03),
        checks=checks,timings=timings,detect_timings=detect_timings,warmup=args.warmup,iterations=args.iterations,seed=0,
        benchmark_image=frames[0][0],benchmark_prompt=prompt,
        reference='Official OmDetV2TurboInfer, trained checkpoint, PyTorch FP32, TF32 disabled',
        engine_scope='GPU-resident preprocessed image and prompt features; preallocated TRT buffers; CUDA completion synchronized; excludes preprocessing/transfers/postprocess/load/build. PyTorch allocates outputs.',
        detect_scope='Application detect call: PIL resize/normalization, image H2D, inference, sigmoid/Top-K/NMS and CPU result conversion. Prompt features precomputed; excludes disk I/O and model load.',
        precision_detail=fp16.metadata['precision_detail'],gpu=torch.cuda.get_device_name(),
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        versions={n:__import__('importlib.metadata',fromlist=['version']).version(n) for n in ['torch','tensorrt','timm']})
    (output/'accuracy_latency.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=all_passed,timings={n:t['wall']['p50_ms'] for n,t in timings.items()},
                         detect_timings={n:t['wall']['p50_ms'] for n,t in detect_timings.items()}),indent=2))
    if not all_passed:
        raise SystemExit('Accuracy checks failed; see accuracy_latency.json (thresholds unchanged).')


if __name__=='__main__':
    main()
