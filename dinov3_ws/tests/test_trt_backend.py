import pytest
from dinov3_cases.trt_backend import (MODEL_ID, MODEL_VARIANTS, TRTBackbone,
                                     model_spec_for_id, model_variant, validate_shape)


@pytest.mark.parametrize('shape', [(1, 3, 32, 32), (1, 3, 53, 87),
                                  (1, 3, 720, 1280), (1, 3, 2464, 1403)])
def test_native_shapes_include_incomplete_edge_patches(shape):
    validate_shape(shape, (1, 3, 32, 32), (1, 3, 2560, 2560))


@pytest.mark.parametrize('shape', [(2, 3, 448, 448), (1, 1, 448, 448),
                                  (1, 3, 16, 32), (1, 3, 2561, 448)])
def test_unsupported_input_is_rejected_without_resize(shape):
    with pytest.raises(ValueError):
        validate_shape(shape, (1, 3, 32, 32), (1, 3, 2560, 2560))


def test_wrong_backbone_and_missing_engine_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match='model_id'):
        TRTBackbone(tmp_path / 'missing.engine', 'wrong-model')
    with pytest.raises(FileNotFoundError, match='engine not found'):
        TRTBackbone(tmp_path / 'missing.engine', MODEL_ID)


def test_fixed_engine_does_not_accept_native_rectangular_image():
    with pytest.raises(ValueError, match='outside'):
        validate_shape((1, 3, 720, 1280), (1, 3, 448, 448), (1, 3, 448, 448))


def test_selectable_vit_model_specs():
    assert set(MODEL_VARIANTS) == {'vith16plus', 'vitb16'}
    assert model_variant('vith16plus')['hidden_size'] == 1280
    assert model_variant('vitb16')['hidden_size'] == 768
    for variant, spec in MODEL_VARIANTS.items():
        assert model_spec_for_id(spec['model_id']) == (variant, spec)
        assert spec['patch_size'] == 16 and spec['num_register_tokens'] == 4
    with pytest.raises(ValueError, match='Unknown model variant'):
        model_variant('unknown')
