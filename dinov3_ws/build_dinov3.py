"""Legacy native-engine helpers; CLI now builds official Torch-TensorRT Cat2Real."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
REPO = WORKSPACE / 'TensorRT-Model-Connect'
DEFAULT_MODEL = 'facebook/dinov3-vits16-pretrain-lvd1689m'
REVISIONS = {
    DEFAULT_MODEL: '114c1379950215c8b35dfcd4e90a5c251dde0d32',
    'timm/vit_small_patch16_dinov3_qkvb.lvd1689m': '2c7705788ac282557562465d6443606664a55f05',
}
THRESHOLDS = dict(full_cosine=.999, cls_cosine=.999, pooler_cosine=.999,
                  register_cosine=.999, mean_patch_cosine=.999,
                  p01_patch_cosine=.995, relative_frobenius=.01)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def revision(repo):
    result = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unavailable'


def extract(bundle, engine):
    with bundle.open('rb') as f:
        if f.read(8) != b'BUNDLE\x01\x00':
            raise ValueError('Unsupported bundle format')
        header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
        start = f.tell()
        section = header['sections']['engine.plan']
        f.seek(start + section['offset'])
        data = f.read(section['length'])
        if len(data) != section['length']:
            raise ValueError('Truncated bundle engine')
    engine.write_bytes(data)
    return header


class Engine:
    """Dedicated stream, device-resident input and preallocated output buffers."""
    def __init__(self, path, pixels):
        import tensorrt as trt
        import torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError('TensorRT could not deserialize the engine')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT could not create an execution context')
        self.stream = torch.cuda.Stream()
        self.buffers = {}
        self.outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            if dtype != trt.float32 or any(n <= 0 for n in shape):
                raise ValueError(f'Expected static FP32 engine I/O: {name}: {shape} {dtype}')
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name != 'pixel_values' or shape != tuple(pixels.shape):
                    raise ValueError(f'Unexpected input: {name}: {shape}')
                self.buffers[name] = pixels.clone()
            else:
                self.outputs.append(name)
                self.buffers[name] = torch.empty(shape, device='cuda', dtype=torch.float32)
            if not self.context.set_tensor_address(name, self.buffers[name].data_ptr()):
                raise RuntimeError(f'Could not bind {name}')
        if not {'last_hidden_state', 'pooler_output'} <= set(self.outputs):
            raise ValueError('Missing DINOv3 hidden state or pooler output')
        torch.cuda.synchronize()

    def run(self, synchronize=True):
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT inference failed')
        if synchronize:
            self.stream.synchronize()
        return {n: self.buffers[n] for n in self.outputs}

    def set_input(self, pixels):
        self.buffers['pixel_values'].copy_(pixels)
        import torch
        torch.cuda.synchronize()


def cosine(a, b):
    import numpy as np
    return float(np.dot(a.ravel(), b.ravel()) /
                 max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def metrics(actual, expected, registers):
    import numpy as np
    a, b = actual['last_hidden_state'], expected['last_hidden_state']
    if a.shape != b.shape or not all(np.isfinite(x).all() for x in [*actual.values(), *expected.values()]):
        raise ValueError('Non-finite outputs or mismatched reference shape')
    delta = a - b
    ap, bp = a[:, 1+registers:], b[:, 1+registers:]
    patch = np.sum(ap * bp, axis=-1) / np.maximum(np.linalg.norm(ap, axis=-1) * np.linalg.norm(bp, axis=-1), 1e-30)
    return dict(full_cosine=cosine(a, b), cls_cosine=cosine(a[:, 0], b[:, 0]),
                pooler_cosine=cosine(actual['pooler_output'], expected['pooler_output']),
                register_cosine=cosine(a[:, 1:1+registers], b[:, 1:1+registers]) if registers else 1.,
                mean_patch_cosine=float(patch.mean()), min_patch_cosine=float(patch.min()),
                p01_patch_cosine=float(np.percentile(patch, 1)),
                relative_frobenius=float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30)),
                max_abs_error=float(np.abs(delta).max()), mean_abs_error=float(np.abs(delta).mean()))


def benchmark(fn, stream, warmup, iterations):
    import numpy as np
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    wall, gpu = [], []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        start.record(stream)
        result = fn()
        end.record(stream)
        end.synchronize()
        wall.append((time.perf_counter() - t) * 1000)
        gpu.append(start.elapsed_time(end))
        del result
    def stats(values):
        return dict(p50_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                    min_ms=float(np.min(values)), samples_ms=values)
    return dict(wall=stats(wall), cuda_event=stats(gpu))


def validate(output, metadata, args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    model_dir = Path(metadata['model_path'])
    config = json.loads((model_dir / 'config.json').read_text())
    timm_model = config.get('architecture') == 'vit_small_patch16_dinov3_qkvb'
    if timm_model:
        import timm
        from transformers import DINOv3ViTImageProcessorFast
        model = timm.create_model(config['architecture'], pretrained=False,
                                  img_size=args.image_size, checkpoint_path=str(model_dir / 'model.safetensors')).cuda().float().eval()
        processor = DINOv3ViTImageProcessorFast(size={'height': args.image_size, 'width': args.image_size}, resample=2,
                    do_rescale=True, rescale_factor=1/255, do_normalize=True,
                    image_mean=[.485, .456, .406], image_std=[.229, .224, .225])
        registers = int(model.num_prefix_tokens) - 1
        def reference(pixels):
            hidden = model.forward_features(pixels)
            return dict(last_hidden_state=hidden, pooler_output=hidden[:, 0])
    else:
        model = AutoModel.from_pretrained(model_dir, local_files_only=True).cuda().float().eval()
        processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        registers = int(getattr(model.config, 'num_register_tokens', 0))
        def reference(pixels):
            result = model(pixel_values=pixels)
            return dict(last_hidden_state=result.last_hidden_state, pooler_output=result.pooler_output)
    paths = args.images or [REPO / 'families/dinov3/tests/data/test_img.jpeg',
        WORKSPACE / 'dinov3_cases_data/stomach/1790733215386907348_2.png',
        WORKSPACE / 'dinov3_cases_data/stomach/1790733214842141931_1.png']
    def prepare(path):
        with Image.open(path) as image:
            return processor(images=image.convert('RGB'), size={'height': args.image_size, 'width': args.image_size},
                             return_tensors='pt')['pixel_values'].cuda().float().contiguous()
    with torch.inference_mode():
        pixels = prepare(paths[0])
        engine = Engine(output / 'model.engine', pixels)
        comparison = None
        if args.compare_with:
            other = args.compare_with.resolve()
            other_metadata = json.loads((other / 'model.json').read_text())
            for key in ['model_id', 'revision', 'checkpoint_sha256', 'input_shape']:
                if metadata[key] != other_metadata[key]:
                    raise ValueError(f'Comparison artifact differs in {key}')
            if other_metadata['precision'] != 'FP32' or not (other / 'READY').exists():
                raise ValueError('--compare-with requires a validated FP32 artifact')
            if digest(other / 'model.engine') != other_metadata['engine_sha256']:
                raise ValueError('Comparison engine checksum mismatch')
            comparison = Engine(other / 'model.engine', pixels)
        checks = []
        precision_checks = []
        for path in paths:
            pixels = prepare(path)
            engine.set_input(pixels)
            expected = {k: v.cpu().numpy().astype(np.float64) for k,v in reference(pixels).items()}
            actual = {k: v.cpu().numpy().astype(np.float64) for k,v in engine.run().items()}
            if comparison:
                comparison.set_input(pixels)
                baseline = {k: v.cpu().numpy().astype(np.float64) for k,v in comparison.run().items()}
                precision_checks.append(dict(image=str(path), metrics=metrics(actual, baseline, registers)))
            values = metrics(actual, expected, registers)
            passed = all(values[k] <= v if k == 'relative_frobenius' else values[k] >= v for k,v in THRESHOLDS.items())
            checks.append(dict(image=str(path), image_sha256=digest(Path(path)), passed=passed, metrics=values))
        engine.set_input(pixels)
        torch_stream = torch.cuda.current_stream()
        timings = dict(tensorrt=benchmark(lambda: engine.run(synchronize=False), engine.stream, args.warmup, args.iterations),
                       pytorch_fp32=benchmark(lambda: reference(pixels), torch_stream, args.warmup, args.iterations))
        if comparison:
            comparison.set_input(pixels)
            timings['tensorrt_fp32'] = benchmark(lambda: comparison.run(synchronize=False), comparison.stream, args.warmup, args.iterations)
    report = dict(passed=all(c['passed'] for c in checks), model_id=metadata['model_id'],
        revision=metadata['revision'], engine_sha256=digest(output / 'model.engine'),
        input_shape=list(pixels.shape), reference='PyTorch FP32, TF32 disabled',
        precision_comparison=precision_checks, comparison_engine_sha256=digest(args.compare_with / 'model.engine') if args.compare_with else None,
        thresholds=THRESHOLDS, checks=checks, timings=timings, warmup=args.warmup, iterations=args.iterations,
        seed=0, scope='Direct engine execution with CUDA-resident normalized input; synchronized wall time and CUDA events. Excludes image preprocessing, H2D, model load and build. TensorRT buffers reused; PyTorch allocates outputs.',
        versions={n: __import__('importlib.metadata', fromlist=['version']).version(n) for n in ['torch','tensorrt','transformers','timm']},
        gpu=torch.cuda.get_device_name(), driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip())
    write_json(output / 'validation.json', report)
    print(json.dumps(dict(passed=report['passed'], checks=checks,
          latency_ms={n:{k:v for k,v in t['wall'].items() if k!='samples_ms'} for n,t in timings.items()}),indent=2),flush=True)
    if not report['passed']:
        raise RuntimeError('Accuracy validation failed; see validation.json. Artifact not marked ready.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=DEFAULT_MODEL, help='HF model ID or local snapshot directory (DINOv3 ViT)')
    p.add_argument('--revision', help='HF revision; known models default to pinned revisions')
    network = p.add_mutually_exclusive_group()
    network.add_argument('--download', action='store_true', help='Allow model downloads')
    network.add_argument('--offline', action='store_true', help='Use cached/local weights only (default)')
    p.add_argument('--precision', choices=['fp16','fp32','bf16'], default='fp16')
    p.add_argument('--image-size', type=int, choices=[224], default=224, help='Current DINOv3 family uses fixed 224x224 input')
    p.add_argument('--stage', choices=['all','build','validate'], default='all')
    p.add_argument('--output', type=Path, help='New output directory, or existing directory for --stage validate')
    p.add_argument('--compare-with', type=Path, help='Validated FP32 output directory for a same-session precision/latency comparison')
    p.add_argument('--images', type=Path, nargs='+', help='Validation images; default: family fixture and two real images')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iterations', type=int, default=100)
    args = p.parse_args()
    if args.image_size < 16 or args.image_size % 16 or args.warmup < 1 or args.iterations < 10:
        p.error('image-size must be a positive multiple of 16; warmup >= 1 and iterations >= 10')
    if os.geteuid() == 0:
        p.error('Run as admin to keep weights and artifacts writable; use su - admin')
    import torch
    try:
        from cuda.bindings import runtime
        torch.ones(1, device='cuda')
        torch.cuda.synchronize()
    except ImportError as e:
        raise RuntimeError('Missing CUDA bindings. Install: dinov3_trt11_venv/bin/python -m pip install -c dinov3_ws/constraints-thor.txt cuda-python==13.0.3') from e
    except RuntimeError as e:
        raise RuntimeError('CUDA unavailable. Run dinov3_trt11_venv/bin/python dinov3_ws/tests/check_gpu_environment.py; check admin device groups and container driver libraries.') from e
    if args.stage == 'validate':
        if args.output is None:
            p.error('--stage validate requires --output')
        output = args.output.resolve()
        metadata = json.loads((output / 'model.json').read_text())
        args.image_size = metadata['input_shape'][-1]
    else:
        print('[1/5] Preparing model weights', flush=True)
        model_path = Path(args.model).expanduser()
        selected_revision = args.revision or REVISIONS.get(args.model)
        if not model_path.is_dir():
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError
            try:
                model_path = Path(snapshot_download(args.model, revision=selected_revision,
                    cache_dir=ROOT / 'artifacts/huggingface', local_files_only=not args.download,
                    allow_patterns=['*.json','*.safetensors']))
            except GatedRepoError as e:
                raise RuntimeError('Model access denied. Obtain HF model access, then run dinov3_trt11_venv/bin/hf auth login as admin (never paste tokens in chat).') from e
            except LocalEntryNotFoundError as e:
                raise RuntimeError('Weights not cached. Rerun with --download or provide --model /path/to/snapshot.') from e
        model_path = model_path.resolve()
        config = json.loads((model_path / 'config.json').read_text())
        if config.get('model_type') != 'dinov3_vit' and config.get('architecture') != 'vit_small_patch16_dinov3_qkvb':
            raise ValueError('This wrapper supports HF DINOv3 ViT and the public timm ViT-S/16 mirror')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = (args.output or ROOT / 'artifacts/dinov3' / f'{Path(args.model).name}-{args.image_size}-{args.precision}' / stamp).resolve()
        output.mkdir(parents=True, exist_ok=False)
        print(f'Output: {output}',flush=True)
        print('[2/5] Building TensorRT bundle (may take several minutes)',flush=True)
        command = [sys.executable,'-m','tensorrt_model_connect','build',str(model_path),
            '--family','dinov3','--task','image_features','--precision',args.precision,
            '--max-batch-size','1','--output',str(output / 'model.bundle')]
        start = time.perf_counter()
        with (output / 'build.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('[3/5] Extracting engine and recording metadata',flush=True)
        write_json(output / 'bundle-header.json',extract(output / 'model.bundle',output / 'model.engine'))
        metadata = dict(model_id=args.model, model_path=str(model_path),
            revision=model_path.name if model_path.parent.name == 'snapshots' else selected_revision,
            source_commit=revision(REPO), precision=args.precision.upper(),
            input_shape=[1,3,args.image_size,args.image_size],
            patch_size=config.get('patch_size',16), num_register_tokens=config.get('num_register_tokens',4),
            hidden_size=config.get('hidden_size',config.get('num_features',384)),
            engine_sha256=digest(output / 'model.engine'),
            checkpoint_sha256={f.name:digest(f) for f in model_path.glob('*.safetensors')},
            build_seconds=time.perf_counter()-start)
        import tensorrt
        metadata['tensorrt'] = tensorrt.__version__
        write_json(output / 'model.json',metadata)
    if args.stage != 'build':
        if digest(output / 'model.engine') != metadata['engine_sha256']:
            raise ValueError('Engine checksum mismatch')
        print('[4/5] Checking accuracy and measuring latency',flush=True)
        (output / 'READY').unlink(missing_ok=True)
        validate(output, metadata, args)
        (output / 'READY').write_text('Accuracy validation passed; see validation.json for scope.\n')
    print(f'[5/5] Complete: {output}',flush=True)


if __name__ == '__main__':
    from build_models import main as build_official
    sys.argv[1:1] = ['--target', 'cat2real']
    build_official()
