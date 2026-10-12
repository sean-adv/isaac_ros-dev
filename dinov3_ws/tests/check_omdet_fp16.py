"""Validate FP16 detections on held-out images and benchmark against retained FP32."""
import json
import time

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from dinov3_cases.omdet_trt import OmDetDetector, prepare_image, postprocess
from dinov3_cases.runtime import GROUNDING_ENGINE, WORKSPACE


def compare_detections(actual, expected):
    assert len(actual) == len(expected), (actual, expected)
    if not expected:
        return dict(count=0, min_iou=None, max_score_error=0.)
    a = np.asarray([p['bbox'] for p in actual], dtype=np.float64)
    e = np.asarray([p['bbox'] for p in expected], dtype=np.float64)
    lo = np.maximum(a[:,None,:2], e[None,:,:2])
    hi = np.minimum(a[:,None,2:], e[None,:,2:])
    intersection = np.maximum(hi-lo, 0).prod(-1)
    area_a = (a[:,2:]-a[:,:2]).prod(-1)
    area_e = (e[:,2:]-e[:,:2]).prod(-1)
    iou = intersection/(area_a[:,None]+area_e[None,:]-intersection)
    label_mismatch = np.array([[p['detection_label'] != q['detection_label'] for q in expected] for p in actual])
    rows, cols = linear_sum_assignment(1-iou+label_mismatch)
    assert not label_mismatch[rows,cols].any(), (actual, expected)
    assert iou[rows,cols].min() >= .98, (iou[rows,cols], actual, expected)
    score_error = max(abs(actual[r]['detection_score']-expected[c]['detection_score']) for r,c in zip(rows,cols))
    # Bounded confidence drift is expected from FP16 quantization and decoder query selection.
    # Keep exact class/count and >= .98 IoU checks; permit up to three percentage points.
    assert score_error <= .03, (score_error, actual, expected)
    return dict(count=len(expected), min_iou=float(iou[rows,cols].min()), max_score_error=float(score_error))


def benchmark(detector, image, prompt):
    for _ in range(5):
        detector.detect(image,prompt,.3,.25,.5,20)
    samples=[]
    for _ in range(20):
        start=time.perf_counter()
        detector.detect(image,prompt,.3,.25,.5,20)
        samples.append((time.perf_counter()-start)*1000)
    return dict(median_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),samples=20)


def main():
    fp16=OmDetDetector(GROUNDING_ENGINE,'cuda')
    assert fp16.metadata['precision']=='FP16', fp16.metadata
    fp32=OmDetDetector(GROUNDING_ENGINE.with_name('omdet_tiny_fp32.engine'),'cuda')
    assert fp32.metadata['precision']=='FP32'
    for name in fp16.inputs | fp16.outputs:
        assert fp16.dtypes[name] == fp32.dtypes[name]  # FP32 I/O, INT32 task mask.
    # The two held-out stomach images and rotated image are absent from AutoCast calibration.
    paths = [WORKSPACE/'dinov3_cases_data/stomach'/name for name in [
        '1790733215386907348_2.png','1790733214842141931_1.png','1790733216258269633_3.png']]
    frames=[(str(path),Image.open(path).convert('RGB')) for path in paths]
    frames.append((str(paths[-1])+' rotated 90',frames[-1][1].rotate(90,expand=True)))
    import onnxruntime as ort
    options=ort.SessionOptions()
    options.intra_op_num_threads=8
    onnx_fp32=ort.InferenceSession(str(GROUNDING_ENGINE.parent/'omdet_tiny.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    checks=[]
    for profile in fp16.bank.profiles:
        prompt=profile['prompt']
        _,features=fp16.bank.resolve(prompt)
        for name,image in frames:
            inputs=dict(img_tensor=prepare_image(image,fp16.metadata),**{n:torch.from_numpy(v) for n,v in features.items()})
            outputs=fp16.infer(inputs)
            assert all(torch.isfinite(v).all() for v in outputs.values())
            actual=postprocess(outputs['boxes'],outputs['logits'],profile['labels'],image.size,.3,.5,20)
            expected=fp32.detect(image,prompt,.3,.25,.5,20)
            parity=compare_detections(actual,expected)
            boxes,logits=onnx_fp32.run(None,{n:v.numpy() for n,v in inputs.items()})
            onnx_expected=postprocess(torch.from_numpy(boxes),torch.from_numpy(logits),profile['labels'],image.size,.3,.5,20)
            onnx_parity=compare_detections(actual,onnx_expected)
            checks.append(dict(image=name,prompt=prompt,fp32_trt=parity,fp32_onnx=onnx_parity,detections=actual))
    assert any(item['fp32_trt']['count'] for item in checks), 'No nonempty detections were validated'
    image=frames[0][1]
    prompt=fp16.bank.profiles[0]['prompt']
    timings=dict(fp32=benchmark(fp32,image,prompt),fp16=benchmark(fp16,image,prompt))
    timings['speedup']=timings['fp32']['median_ms']/timings['fp16']['median_ms']
    report=dict(passed=True,precision=fp16.metadata['precision'],engine_sha256=fp16.metadata['engine_sha256'],
                tolerances=dict(min_box_iou=.98,max_score_error=.03),checks=checks,timings=timings)
    path=GROUNDING_ENGINE.parent/'validation_fp16.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
