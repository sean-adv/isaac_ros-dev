"""Prompt-only updates preserve engines and reject incompatible input shapes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dinov3_cases.omdet_trt import OfflinePromptBank


@pytest.fixture
def builder():
    path = Path(__file__).parents[1] / 'build_omdet.py'
    spec = importlib.util.spec_from_file_location('omdet_prompt_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifacts(tmp_path, monkeypatch, builder):
    repo = tmp_path / 'repo'
    (repo / 'resources').mkdir(parents=True)
    (repo / 'resources/OmDet-Turbo_tiny_SWIN_T.pth').write_bytes(b'checkpoint')
    output = tmp_path / 'output'
    output.mkdir()
    profile = dict(prompt='cup. book.', labels=['cup', 'book'])
    arrays = {'0_label_feats': np.ones((2, 1, 512), np.float32),
              '0_task_feats': np.ones((77, 1, 512), np.float32),
              '0_task_mask': np.ones((1, 77), np.int32)}
    np.savez(output / 'prompts.npz', **arrays)
    metadata = dict(model_id=builder.MODEL_ID, class_count=2, profiles=[profile],
                    source_commit='original-source',
                    prompt_sha256=hashlib.sha256((output / 'prompts.npz').read_bytes()).hexdigest())
    (output / 'prompts.json').write_text(json.dumps(metadata))
    for precision, name in [('FP16', 'omdet_tiny'), ('FP32', 'omdet_tiny_fp32')]:
        engine = output / f'{name}.engine'
        engine.write_bytes(precision.encode())
        info = dict(metadata, precision=precision, onnx_sha256='preserved-onnx',
                    engine_sha256=hashlib.sha256(engine.read_bytes()).hexdigest())
        (output / f'{name}.json').write_text(json.dumps(info))
    monkeypatch.setattr(builder.subprocess, 'check_output', lambda *a, **kw: 'original-source\n')
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: {})
    monkeypatch.setattr(builder, 'encode_profiles', lambda *a: {
        name: value * 2 if name != '0_task_mask' else value for name, value in arrays.items()
    })
    return repo, output


def snapshot(output):
    return {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}


def test_prompt_config_preserves_actual_labels_with_explicit_capacity(builder, tmp_path):
    profiles = [dict(prompt='Detect bottle, box, cup, book.', labels=['bottle', 'box', 'cup', 'book']),
                dict(prompt='Detect bottle, box.', labels=['bottle', 'box'])]
    path = tmp_path / 'prompts.json'
    path.write_text(json.dumps(dict(profiles=profiles)))
    before = path.read_bytes()
    assert builder.load_profiles(path, max_labels=4) == profiles
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='engine maximum is 2'):
        builder.load_profiles(path, max_labels=2)


@pytest.mark.parametrize('obsolete', ['task', 'output_labels'])
def test_obsolete_config_fields_rejected(builder, tmp_path, obsolete):
    path = tmp_path / 'prompts.json'
    profile = dict(prompt='Detect box.', labels=['box'])
    profile[obsolete] = 'obsolete'
    path.write_text(json.dumps(dict(profiles=[profile])))
    with pytest.raises(ValueError, match='only accept prompt and labels'):
        builder.load_profiles(path, max_labels=4)


def test_encoding_pads_current_features_and_enforces_task_token_capacity(builder, monkeypatch):
    def tokenize(texts, truncate=False):
        assert truncate is False
        tokens = torch.zeros(len(texts), 77, dtype=torch.int64)
        for index, text in enumerate(texts):
            count = len(text.split()) + 2
            tokens[index, :count] = 1
            tokens[index, count - 1] = 99  # EOT is the largest ID.
        return tokens

    class Language:
        def eval(self):
            return self
        def float(self):
            return self
        def __call__(self, tokens, return_tokens=False):
            if return_tokens:
                count = int(tokens.argmax(dim=-1).item()) + 1
                return torch.ones(count, 1, 512), torch.ones(1, count, dtype=torch.int32)
            return torch.arange(len(tokens) * 512, dtype=torch.float32).reshape(len(tokens), 512)

    clip = SimpleNamespace(load=lambda *a, **kw: (Language(), None), tokenize=tokenize)
    monkeypatch.setitem(sys.modules, 'omdet.modeling.language_backbone.clip.models', SimpleNamespace(clip=clip))
    profiles = [dict(prompt='Detect cup, book.', labels=['cup', 'book'])]
    arrays = builder.encode_profiles(Path('.'), {}, profiles, 4, 8)
    assert arrays['0_label_feats'].shape == (4, 1, 512)
    np.testing.assert_array_equal(arrays['0_label_feats'][2], arrays['0_label_feats'][1])
    np.testing.assert_array_equal(arrays['0_label_feats'][3], arrays['0_label_feats'][1])
    assert arrays['0_task_feats'].shape == (8, 1, 512)
    assert arrays['0_task_mask'].tolist() == [[1, 1, 1, 1, 1, 0, 0, 0]]
    with pytest.raises(ValueError, match='uses 5 tokens; engine maximum is 4'):
        builder.encode_profiles(Path('.'), {}, profiles, 4, 4)
    assert builder.encode_profiles(Path('.'), {}, profiles, 2, 5)['0_task_mask'].shape == (1, 5)


@pytest.mark.parametrize('labels,tokens', [(0, 77), (True, 77), (4, 1), (4, 78), (4, True)])
def test_invalid_engine_capacities(builder, labels, tokens):
    with pytest.raises(ValueError):
        builder.validate_capacities(labels, tokens)


def test_update_preserves_engine_bytes_and_build_metadata(builder, artifacts):
    repo, output = artifacts
    before = snapshot(output)
    profile = dict(prompt='bottle. box.', labels=['bottle', 'box'])
    builder.update_prompt_bank(repo, output, [profile], 2, 77)
    for name in ['omdet_tiny', 'omdet_tiny_fp32']:
        assert (output / f'{name}.engine').read_bytes() == before[f'{name}.engine']
        info = json.loads((output / f'{name}.json').read_text())
        previous = json.loads(before[f'{name}.json'])
        for key in ['engine_sha256', 'onnx_sha256', 'precision', 'source_commit']:
            assert info[key] == previous[key]
        assert OfflinePromptBank(output / 'prompts.npz', info).resolve(profile['prompt'])[0] == profile


def test_changed_class_count_rejected_without_writes(builder, artifacts, monkeypatch):
    repo, output = artifacts
    before = snapshot(output)
    monkeypatch.setattr(builder, 'encode_profiles', lambda *a: pytest.fail('must reject before encoding'))
    with pytest.raises(ValueError, match='Engine capacity/model mismatch'):
        builder.update_prompt_bank(repo, output, [], 1, 77)
    assert snapshot(output) == before


def test_corrupt_engine_rejected_without_writes(builder, artifacts):
    repo, output = artifacts
    (output / 'omdet_tiny.engine').write_bytes(b'corrupt')
    before = snapshot(output)
    with pytest.raises(ValueError, match='checksum mismatch'):
        builder.update_prompt_bank(repo, output, [], 2, 77)
    assert snapshot(output) == before


def test_old_alternate_capacity_does_not_block_active_engine(builder, artifacts):
    repo, output = artifacts
    path = output / 'omdet_tiny_fp32.json'
    metadata = json.loads(path.read_text())
    metadata['class_count'] = 3
    path.write_text(json.dumps(metadata))
    old = path.read_bytes()
    profile = dict(prompt='Detect bottle, box.', labels=['bottle', 'box'])
    builder.update_prompt_bank(repo, output, [profile], 2, 77)
    assert path.read_bytes() == old
    active = json.loads((output / 'omdet_tiny.json').read_text())
    assert active['profiles'] == [profile]
    assert active['max_labels'] == 2


def test_invalid_new_features_do_not_replace_bank(builder, artifacts, monkeypatch):
    repo, output = artifacts
    before = snapshot(output)
    monkeypatch.setattr(builder, 'encode_profiles', lambda *a: {})
    with pytest.raises(KeyError):
        builder.update_prompt_bank(repo, output, [dict(prompt='bottle. box.', labels=['bottle', 'box'])], 2, 77)
    assert snapshot(output) == before


def test_replace_failure_rolls_back_all_files(builder, artifacts, monkeypatch):
    repo, output = artifacts
    before = snapshot(output)
    replace = builder.os.replace
    failed = False
    def fail_once(source, destination):
        nonlocal failed
        if Path(destination).name == 'omdet_tiny.json' and not failed:
            failed = True
            raise OSError('simulated replace failure')
        return replace(source, destination)
    monkeypatch.setattr(builder.os, 'replace', fail_once)
    with pytest.raises(OSError, match='simulated'):
        builder.update_prompt_bank(repo, output, [dict(prompt='bottle. box.', labels=['bottle', 'box'])], 2, 77)
    assert snapshot(output) == before
