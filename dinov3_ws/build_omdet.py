"""Official OmDetV2TurboInfer export + precomputed official CLIP prompt features."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
MODEL_ID = "omlab/OmDet-Turbo_tiny_SWIN_T"


def capacities(metadata):
    # Existing engines have class_count and a fixed 77-token task input.
    max_labels = metadata.get('max_labels', metadata.get('class_count'))
    max_prompt_tokens = metadata.get('max_prompt_tokens', 77)
    validate_capacities(max_labels, max_prompt_tokens)
    return max_labels, max_prompt_tokens


def validate_capacities(max_labels, max_prompt_tokens):
    if isinstance(max_labels, bool) or not isinstance(max_labels, int) or max_labels < 1:
        raise ValueError('max_labels must be a positive integer')
    if (isinstance(max_prompt_tokens, bool) or not isinstance(max_prompt_tokens, int)
            or not 2 <= max_prompt_tokens <= 77):
        raise ValueError('max_prompt_tokens must be between 2 and 77 (including CLIP start/end tokens)')


def load_profiles(path, max_labels):
    profiles = json.loads(path.read_text())['profiles']
    if not isinstance(profiles, list) or not profiles:
        raise ValueError('At least one offline prompt profile is required')
    keys = set()
    for profile in profiles:
        if set(profile) - {'prompt', 'labels'}:
            raise ValueError('Profiles only accept prompt and labels; remove task/output_labels')
        labels = profile['labels']
        if (not isinstance(labels, list) or not labels
                or any(not isinstance(s, str) or not s.strip() for s in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError('Labels must be nonempty and unique')
        if len(labels) > max_labels:
            raise ValueError(f"Prompt {profile['prompt']!r} has {len(labels)} labels; engine maximum is {max_labels}")
        if not isinstance(profile['prompt'], str) or not profile['prompt'].strip():
            raise ValueError('prompt must be a nonempty string')
        # The exact prompt is both the selection key and the CLIP task sentence.
        key = tuple(s.strip() for s in profile['prompt'].split('.') if s.strip())
        if not key or key in keys:
            raise ValueError('Offline prompt keys must be nonempty and unique')
        keys.add(key)
    return profiles


def encode_profiles(repo, state, profiles, max_labels, max_prompt_tokens):
    import torch
    import torch.nn.functional as F
    from omdet.modeling.language_backbone.clip.models import clip
    language, _ = clip.load(str(repo / 'resources/ViT-B-16.pt'), device='cpu', jit=False)
    language.visual = None
    language.logit_scale = None
    language_state = {k.removeprefix('language_backbone.'): v for k, v in state.items()
                      if k.startswith('language_backbone.')}
    if language_state:
        language.load_state_dict(language_state, strict=True)
    language.eval().float()
    arrays = {}
    with torch.inference_mode():
        for i, profile in enumerate(profiles):
            label_feats = language(clip.tokenize(profile['labels'], truncate=False)).unsqueeze(1).float()
            task_tokens = clip.tokenize([profile['prompt']], truncate=False)
            # CLIP's EOT has the largest vocabulary ID; count includes SOT/EOT.
            token_count = int(task_tokens.argmax(dim=-1).item()) + 1
            if token_count > max_prompt_tokens:
                raise ValueError(f"Prompt {profile['prompt']!r} uses {token_count} tokens; engine maximum is {max_prompt_tokens}")
            task_feats, task_mask = language(task_tokens, return_tokens=True)
            task_feats = F.pad(task_feats.float(), (0, 0, 0, 0, 0, max_prompt_tokens-task_feats.shape[0]))
            task_mask = F.pad(task_mask.to(torch.int32), (0, max_prompt_tokens-task_mask.shape[1]))
            # Repeating the last valid class preserves encoder max-class scores:
            # no unrelated class or zero embedding may influence query selection.
            padding = max_labels - len(profile['labels'])
            label_feats = torch.cat([label_feats, label_feats[-1:].expand(padding, -1, -1)], dim=0)
            for name, tensor in [('label_feats', label_feats), ('task_feats', task_feats), ('task_mask', task_mask)]:
                arrays[f'{i}_{name}'] = tensor.numpy()
    return arrays


def update_prompt_bank(repo, output, profiles, max_labels, max_prompt_tokens):
    """Encode config prompts within existing capacities; never rebuild engines."""
    import numpy as np
    prompt_metadata = json.loads((output / 'prompts.json').read_text())
    if prompt_metadata['model_id'] != MODEL_ID or capacities(prompt_metadata) != (max_labels, max_prompt_tokens):
        raise ValueError('Engine capacity/model mismatch: rebuild ONNX and engines with --stage all')
    engine_metadata = {}
    for engine in sorted(output.glob('omdet_tiny*.engine')):
        metadata_path = engine.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        if hashlib.sha256(engine.read_bytes()).hexdigest() != metadata['engine_sha256']:
            raise ValueError(f'Engine checksum mismatch: {engine}')
        if capacities(metadata) != (max_labels, max_prompt_tokens):
            print(f'Skipped {engine.name}: different capacities; rebuild that precision before reuse', flush=True)
            continue
        engine_metadata[metadata_path.name] = metadata
    if not engine_metadata:
        raise ValueError('Prompt-only updates require an OmDet engine with matching capacities; run --stage all first')
    source_commit = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    for metadata in [prompt_metadata, *engine_metadata.values()]:
        if metadata['model_id'] != MODEL_ID or capacities(metadata) != (max_labels, max_prompt_tokens):
            raise ValueError('Engine capacity/model mismatch: rebuild ONNX and engines with --stage all')
        if metadata['source_commit'] != source_commit:
            raise ValueError('OmDet source commit differs from engine metadata; use the original source or rebuild')
    import torch
    torch.set_num_threads(8)
    checkpoint_path = repo / 'resources/OmDet-Turbo_tiny_SWIN_T.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    arrays = encode_profiles(repo, checkpoint.get('model', checkpoint), profiles, max_labels, max_prompt_tokens)
    sys.path.insert(0, str(ROOT / 'src/dinov3_cases'))
    from dinov3_cases.omdet_trt import OfflinePromptBank
    with tempfile.TemporaryDirectory(prefix='.prompt-update-', dir=output) as directory:
        staging = Path(directory)
        np.savez(staging / 'prompts.npz', **arrays)
        updates = dict(profiles=profiles, max_labels=max_labels, max_prompt_tokens=max_prompt_tokens,
                       prompt_sha256=hashlib.sha256((staging / 'prompts.npz').read_bytes()).hexdigest(),
                       prompt_source_commit=source_commit,
                       prompt_checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
        for name, previous in {'prompts.json': prompt_metadata, **engine_metadata}.items():
            metadata = dict(previous, **updates)
            metadata.pop('class_count', None)
            OfflinePromptBank(staging / 'prompts.npz', metadata)
            (staging / name).write_text(json.dumps(metadata, indent=2) + '\n')
        names = ['prompts.npz', 'prompts.json', *engine_metadata]
        originals = {name: (output / name).read_bytes() for name in names}
        replaced = []
        try:
            for name in names:
                os.replace(staging / name, output / name)
                replaced.append(name)
        except BaseException:
            for name in reversed(replaced):
                backup = staging / name
                backup.write_bytes(originals[name])
                os.replace(backup, output / name)
            raise
    print(f'Updated prompts within max_labels={max_labels}, max_prompt_tokens={max_prompt_tokens}; ONNX/engines unchanged', flush=True)


def enforce_fp32_accumulation(model):
    """NVIDIA cast-around-GEMM pattern: FP16 operands/output, FP32 accumulation."""
    import copy
    import onnx
    from onnx import TensorProto, helper
    model = onnx.shape_inference.infer_shapes(model)
    values = {v.name: v for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    types = {name: v.type.tensor_type.elem_type for name,v in values.items()}
    types.update({v.name: v.data_type for v in model.graph.initializer})
    nodes = []
    for node in model.graph.node:
        if node.op_type not in {'MatMul','Gemm','Conv'} or types.get(node.output[0]) != TensorProto.FLOAT16:
            nodes.append(node)
            continue
        changed = False
        for i, operand in enumerate(node.input):
            if types.get(operand) == TensorProto.FLOAT16:
                promoted = f'{node.name}/accumulate_fp32_input_{i}'
                nodes.append(helper.make_node('Cast',[operand],[promoted],name=promoted,to=TensorProto.FLOAT))
                node.input[i] = promoted
                changed = True
        if changed:
            original = node.output[0]
            promoted = f'{node.name}/accumulate_fp32_output'
            node.output[0] = promoted
            node.name += '/fp32_accumulation'
            nodes.append(node)
            nodes.append(helper.make_node('Cast',[promoted],[original],name=promoted+'/cast_back',to=TensorProto.FLOAT16))
            if original in values:
                info = copy.deepcopy(values[original])
                info.name = promoted
                info.type.tensor_type.elem_type = TensorProto.FLOAT
                model.graph.value_info.append(info)
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=WORKSPACE / 'OmDet')
    parser.add_argument('--prompts', type=Path, default=ROOT / 'config/omdet_prompts.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/omdet')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--stage', choices=['prompts', 'export', 'engine', 'all'], default='all')
    parser.add_argument('--stage-prompt', dest='stage', action='store_const', const='prompts',
                        help='Only encode config prompt/labels using existing engine capacities')
    parser.add_argument('--max-labels', type=int, help='Maximum number of labels accepted by the engine')
    parser.add_argument('--max-prompt-tokens', type=int, help='Maximum task token length, including CLIP SOT/EOT (2..77)')
    parser.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16')
    parser.add_argument('--reuse-fp16-onnx', action='store_true', help='Reuse existing AutoCast ONNX when rebuilding tactics')
    parser.add_argument('--calibration-images', type=Path, nargs='+', help='Real images for FP16 AutoCast calibration')
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    args.prompts = args.prompts.resolve()
    if args.stage in ('export', 'all'):
        if args.max_labels is None or args.max_prompt_tokens is None:
            parser.error('Building requires --max-labels and --max-prompt-tokens')
        validate_capacities(args.max_labels, args.max_prompt_tokens)
    if args.calibration_images:
        args.calibration_images = [p.resolve() for p in args.calibration_images]
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prompts':
        if args.download:
            parser.error('--stage prompts is offline; download weights with --stage all first')
        metadata = json.loads((output / 'prompts.json').read_text())
        max_labels, max_prompt_tokens = capacities(metadata)
        if ((args.max_labels is not None and args.max_labels != max_labels)
                or (args.max_prompt_tokens is not None and args.max_prompt_tokens != max_prompt_tokens)):
            parser.error('Prompt conversion cannot change existing engine capacities; rebuild with --stage all')
        profiles = load_profiles(args.prompts, max_labels)
        sys.path[:0] = [str(repo), str(WORKSPACE / 'detectron2')]
        update_prompt_bank(repo, output, profiles, max_labels, max_prompt_tokens)
        return
    if args.stage in ('export', 'all'):
        if args.download:
            from huggingface_hub import hf_hub_download
            for name in ('OmDet-Turbo_tiny_SWIN_T.pth', 'ViT-B-16.pt'):
                hf_hub_download(MODEL_ID, name, local_dir=repo / 'resources')
        sys.path[:0] = [str(repo), str(WORKSPACE / 'detectron2')]
        os.chdir(repo)
        import numpy as np
        import torch
        from detectron2.config import get_cfg
        from omdet.omdet_v2_turbo.config import add_omdet_v2_turbo_config
        from omdet.omdet_v2_turbo.infer_model import OmDetV2TurboInfer
        torch.set_num_threads(8)
        torch.manual_seed(0)
        profiles = load_profiles(args.prompts, args.max_labels)
        cfg = get_cfg()
        add_omdet_v2_turbo_config(cfg)
        cfg.merge_from_file(str(repo / 'configs/OmDet-Turbo_tiny_SWIN_T.yaml'))
        cfg.MODEL.META_ARCHITECTURE = 'OmDetV2TurboInfer'
        cfg.MODEL.DEVICE = 'cpu'
        cfg.freeze()
        model = OmDetV2TurboInfer(cfg).eval()
        checkpoint = torch.load(repo / 'resources/OmDet-Turbo_tiny_SWIN_T.pth', map_location='cpu', weights_only=True)
        state = checkpoint.get('model', checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or any(not name.startswith('language_backbone.') for name in unexpected):
            raise RuntimeError(f'Detector weight mismatch: missing={missing}, unexpected={unexpected}')
        print('Ignored checkpoint keys:', unexpected, flush=True)
        # Prompt-only and full export use the same trained official CLIP encoder.
        arrays = encode_profiles(repo, state, profiles, args.max_labels, args.max_prompt_tokens)
        np.savez(output / 'prompts.npz', **arrays)
        metadata = dict(model_id=MODEL_ID, profiles=profiles, max_labels=args.max_labels,
                        max_prompt_tokens=args.max_prompt_tokens, input_size=640,
                        mean=list(cfg.MODEL.PIXEL_MEAN), std=list(cfg.MODEL.PIXEL_STD),
                        source_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                        prompt_sha256=hashlib.sha256((output / 'prompts.npz').read_bytes()).hexdigest())
        (output / 'prompts.json').write_text(json.dumps(metadata, indent=2) + '\n')
        example = (torch.rand(1, 3, 640, 640), *(torch.from_numpy(arrays[f'0_{n}']) for n in ['label_feats', 'task_feats', 'task_mask']))
        # Same inference architecture, input names, fixed shapes and opset as official export.py.
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            torch.onnx.export(model, example, str(output / 'omdet_tiny.onnx'), opset_version=17,
                              input_names=['img_tensor', 'label_feats', 'task_feats', 'task_mask'],
                              output_names=['boxes', 'logits'], dynamo=False)
            expected = model(*example)
        np.savez(output / 'export_reference.npz', **{n: t.numpy() for n, t in zip(['img_tensor','label_feats','task_feats','task_mask'], example)},
                 boxes=expected[0].numpy(), logits=expected[1].numpy())
        import onnx
        onnx.checker.check_model(str(output / 'omdet_tiny.onnx'))
        print('Exported official ONNX and offline prompt bank', flush=True)
    if args.stage in ('engine', 'all'):
        source_metadata = json.loads((output / 'prompts.json').read_text())
        max_labels, max_prompt_tokens = capacities(source_metadata)
        if ((args.max_labels is not None and args.max_labels != max_labels)
                or (args.max_prompt_tokens is not None and args.max_prompt_tokens != max_prompt_tokens)):
            parser.error('Capacity differs from exported ONNX; rerun --stage all or --stage export')
        onnx_path = output / 'omdet_tiny.onnx'
        conversion_metadata = {}
        if args.precision == 'fp16':
            import numpy as np
            import onnx
            from PIL import Image
            sys.path.insert(0, str(ROOT / 'src/dinov3_cases'))
            from dinov3_cases.omdet_trt import OfflinePromptBank, prepare_image
            prompt_metadata = json.loads((output / 'prompts.json').read_text())
            bank = OfflinePromptBank(output / 'prompts.npz', prompt_metadata)
            images = args.calibration_images or [
                WORKSPACE / 'dinov3_cases_data/stomach/1790733215386907348_2.png',
                WORKSPACE / 'dinov3_cases_data/stomach/1790733214108382426_0.png',
                WORKSPACE / 'dinov3_cases_data/punateng/ccc3488d975f406095947dcf03583dde_cutout.png',
            ]
            calibration = output / 'fp16_calibration'
            calibration.mkdir(exist_ok=True)
            # Remove only generated batches so changed prompt sets cannot reuse stale calibration.
            for old_batch in calibration.glob('batch_*.npz'):
                old_batch.unlink()
            for i, image_path in enumerate(images):
                with Image.open(image_path) as image:
                    pixels = prepare_image(image, prompt_metadata).numpy()
                for j, (_, features) in enumerate(bank.entries.values()):
                    np.savez(calibration / f'batch_{i}_{j}.npz', img_tensor=pixels, **features)
            onnx_path = output / 'omdet_tiny_fp16.onnx'
            sensitive_ops = ['LayerNormalization', 'Softmax', 'ReduceMean', 'ReduceSum', 'ReduceMax']
            if not args.reuse_fp16_onnx:
                subprocess.run([
                    sys.executable, '-m', 'modelopt.onnx.autocast',
                    '--onnx_path', str(output / 'omdet_tiny.onnx'), '--output_path', str(onnx_path),
                    '--low_precision_type', 'fp16', '--calibration_data', str(calibration),
                    '--keep_io_types', '--providers', 'cpu', '--op_types_to_exclude', *sensitive_ops,
                    '--nodes_to_exclude', r'^/decoder/.*',
                    '--log_level', 'INFO',
                ], check=True)
            converted = enforce_fp32_accumulation(onnx.load(str(onnx_path)))
            tensor_types = {v.name: v.type.tensor_type.elem_type for v in converted.graph.value_info}
            half_decoder_nodes = [n.name for n in converted.graph.node if n.name.startswith('/decoder/')
                                  and any(tensor_types.get(name) == onnx.TensorProto.FLOAT16 for name in n.output)]
            if half_decoder_nodes:
                raise RuntimeError('Decoder must remain FP32; rerun without --reuse-fp16-onnx')
            onnx.save(converted, str(onnx_path))
            onnx.checker.check_model(str(onnx_path))
            fp16_weights = sum(t.data_type == onnx.TensorProto.FLOAT16 for t in converted.graph.initializer)
            if not fp16_weights:
                raise RuntimeError('AutoCast did not convert any model weights to FP16')
            conversion_metadata = dict(converter='NVIDIA ModelOpt AutoCast',
                                       fp32_op_types=sensitive_ops, fp32_node_patterns=[r'^/decoder/.*'], keep_io_types=True,
                                       precision_detail='FP16 vision; FP32 decoder, sensitive ops, accumulation and I/O',
                                       fp16_initializer_count=fp16_weights,
                                       fp32_accumulation_node_count=sum(n.name.endswith('/fp32_accumulation') for n in converted.graph.node),
                                       calibration_images=[str(p) for p in images])
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, '')
        builder = trt.Builder(logger)
        network = builder.create_network()
        onnx_parser = trt.OnnxParser(network, logger)
        if not onnx_parser.parse_from_file(str(onnx_path)):
            raise RuntimeError('\n'.join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))
        expected_shapes = dict(label_feats=(max_labels, 1, 512),
                               task_feats=(max_prompt_tokens, 1, 512), task_mask=(1, max_prompt_tokens))
        actual_shapes = {network.get_input(i).name: tuple(network.get_input(i).shape)
                         for i in range(network.num_inputs)}
        if any(actual_shapes.get(name) != shape for name, shape in expected_shapes.items()):
            raise ValueError('ONNX inputs do not match metadata capacities; rerun --stage export')
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.builder_optimization_level = 3
        config.clear_flag(trt.BuilderFlag.TF32)
        timing_path = output / 'omdet_timing.cache'
        cache = config.create_timing_cache(timing_path.read_bytes() if timing_path.exists() else b'')
        config.set_timing_cache(cache, ignore_mismatch=False)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError('OmDet TensorRT build failed')
        timing_path.write_bytes(bytes(config.get_timing_cache().serialize()))
        engine_path = output / ('omdet_tiny.engine' if args.precision == 'fp16' else 'omdet_tiny_fp32.engine')
        temporary = engine_path.with_suffix('.engine.tmp')
        temporary.write_bytes(bytes(plan))
        os.replace(temporary, engine_path)
        metadata = json.loads((output / 'prompts.json').read_text())
        metadata.update(tensorrt=trt.__version__, precision=args.precision.upper(), tf32=False,
                        **conversion_metadata, source_onnx=str(onnx_path),
                        engine_sha256=hashlib.sha256(bytes(plan)).hexdigest(),
                        onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest())
        engine_path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print('Built', engine_path, flush=True)


if __name__ == '__main__':
    main()
