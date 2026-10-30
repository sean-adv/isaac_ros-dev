"""GPU/IPC regression using only models and packages inside dinov3_ws."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/dinov3_cases'))
os.environ['PYTHONPATH']=str(ROOT/'src/dinov3_cases')+os.pathsep+os.environ.get('PYTHONPATH','')
import numpy as np
from PIL import Image
import torch

from dinov3_cases.encoder import Encoder
from dinov3_cases.grounded import ObjectGallery, recognize_objects
from dinov3_cases.grounding_ipc import SharedMemoryGroundingDetector
from dinov3_cases.sam3_trt import Sam3Detector
from dinov3_cases.runtime import PROJECT, DINO_ENGINE, DINO_MODEL, GROUNDING_ENGINE, SAM3_MODEL, CASES_DIR, INPUT_SIZE


def iou(a,b):
    a,b=np.array(a),np.array(b)
    intersect=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(intersect/max(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-intersect,1e-9))


def main():
    torch.set_num_threads(8)
    for path in (DINO_ENGINE,DINO_MODEL,GROUNDING_ENGINE,SAM3_MODEL,CASES_DIR):
        assert path.resolve().is_relative_to(PROJECT),path
    assert Path(sys.prefix).resolve().is_relative_to(PROJECT),sys.prefix
    image=Image.open(ROOT/'artifacts/validation/person.jpg').convert('RGB')
    detector=Sam3Detector(GROUNDING_ENGINE,'cuda:0')
    single=detector.detect(image,'person',.5,20)
    assert len(single)==1,single
    reference=np.array([.4564275444,.4020643234,.5659345388,.7773836851])*[*image.size,*image.size]
    overlap=iou(single[0]['bbox'],reference)
    assert overlap>.98,(overlap,single)
    labels=['person','car','bicycle','dog']
    multiple=detector.detect(image,','.join(labels),.3,20)
    multi_checks=[]
    for label in labels:
        solo=detector.detect(image,label,.3,20)
        actual=[p for p in multiple if p['detection_label']==label]
        assert len(solo)==len(actual),(label,solo,actual)
        for expected in solo:
            closest=max(actual,key=lambda p:iou(p['bbox'],expected['bbox']))
            assert iou(closest['bbox'],expected['bbox'])>.98
            assert abs(closest['detection_score']-expected['detection_score'])<.03
        multi_checks.append(dict(prompt=label,detections=len(actual)))
    encoder=Encoder()
    gallery=ObjectGallery();gallery.refresh(CASES_DIR,encoder,INPUT_SIZE,16)
    assert gallery.features.shape[-1]==1024 and torch.isfinite(gallery.features).all()
    results=recognize_objects(image,single,encoder,gallery,INPUT_SIZE,16,.8,.05)
    assert len(results)==1 and np.isfinite(results[0]['score'])
    worker=SharedMemoryGroundingDetector(GROUNDING_ENGINE,'cuda:0',python=sys.executable)
    try:
        assert worker.worker_info['tensorrt'].startswith('11.')
        assert Path(worker.worker_info['python']).parent.parent==Path(sys.prefix)
        worker.validate_prompt('person')
        remote=worker.detect(image,'person',.5,20)
        assert len(remote)==len(single)
        assert remote[0]['bbox']==single[0]['bbox']
        assert abs(remote[0]['detection_score']-single[0]['detection_score'])<1e-5
        info=worker.worker_info
    finally:worker.close()
    for _ in range(3):detector.detect(image,'person',.5,20)
    timings=[]
    for _ in range(10):
        start=time.perf_counter();detector.detect(image,'person',.5,20)
        timings.append((time.perf_counter()-start)*1000)
    forbidden=('/DART/','/dinov3_tensorrt/','/TensorRT-Model-Connect/','/detectron2/')
    external=[getattr(m,'__file__','') for m in sys.modules.values()
              if getattr(m,'__file__',None) and any(s in str(m.__file__) for s in forbidden)]
    assert not external,external
    maps=Path('/proc/self/maps').read_text()
    assert 'lq_plugins.so' not in maps
    report=dict(passed=True,python=sys.executable,prefix=sys.prefix,gpu=torch.cuda.get_device_name(),
        detection=single,fp32_bbox_iou=overlap,multiclass=multi_checks,gallery_references=len(gallery.labels),
        descriptor_dimension=1024,recognition=results,worker=info,
        detect_wall_ms=dict(mean=float(np.mean(timings)),p95=float(np.percentile(timings,95)),samples=10),
        external_project_modules=external,custom_sam3_plugin_loaded=False,
        model_paths=dict(sam3=str(GROUNDING_ENGINE),cat2real=str(DINO_ENGINE)))
    out=ROOT/'artifacts/standalone-pipeline-validation.json'
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
