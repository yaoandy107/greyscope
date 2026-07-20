"""Tests for the export helpers: head-aware scalar decode, the shipped quantization
recipes, and the StandaloneScorer shim that runs exported artifacts through the
training-time eval protocol."""

import numpy as np
import pytest

from greyscope.eval import StandaloneScorer, _predict_bucket_logits
from greyscope.export import _quantization_config, _scalar_decode


def test_scalar_decode_seqcls_is_softmax_expectation():
    from greyscope.eval import compute_scalar_score

    logits = np.array([[9.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 9.0]])
    np.testing.assert_allclose(_scalar_decode("seqcls", 4)(logits),
                               compute_scalar_score(logits, 4))


def test_scalar_decode_corn_is_cumulative():
    from greyscope.corn import corn_scalar_score

    logits = np.array([[9.0, 9.0, 9.0], [-9.0, -9.0, -9.0]])
    decoded = _scalar_decode("corn", 4)(logits)
    np.testing.assert_allclose(decoded, corn_scalar_score(logits))
    assert decoded[0] == pytest.approx(1.0, abs=1e-3)
    assert decoded[1] == pytest.approx(0.0, abs=1e-3)


def test_quantization_config_fp8_excludes_score_and_gdn():
    cfg = _quantization_config("fp8")
    assert "score" in cfg.modules_to_not_convert
    # 128x128 fp8 blocks can't tile the GDN low-rank projections — Qwen's own FP8
    # checkpoints exclude these same modules.
    for pattern in ("linear_attn.conv1d", "linear_attn.in_proj_a", "linear_attn.in_proj_b"):
        assert pattern in cfg.modules_to_not_convert


def test_quantization_config_int4_tile_packed_excludes_score():
    pytest.importorskip("torchao")
    cfg = _quantization_config("int4")
    assert "score" in cfg.modules_to_not_convert
    # The GDN low-rank projections don't divide into int4 groups — same exclusions as fp8.
    for pattern in ("linear_attn.conv1d", "linear_attn.in_proj_a", "linear_attn.in_proj_b"):
        assert pattern in cfg.modules_to_not_convert
    # torchao renders this as an enum or a plain string depending on version.
    assert "tile_packed_to_4d" in str(cfg.quant_type.int4_packing_format).lower()
    assert cfg.quant_type.group_size == 128


def test_quantization_config_rejects_unknown_precision():
    with pytest.raises(ValueError, match="int4.*fp8"):
        _quantization_config("gguf")


def test_standalone_scorer_matches_trainer_predict_interface(monkeypatch):
    captured = {}

    def fake_batch_logits(model, tok, prompts, *, max_length, batch_size):
        captured.update(model=model, tok=tok, prompts=list(prompts),
                        max_length=max_length, batch_size=batch_size)
        return np.zeros((len(prompts), 3))

    import greyscope.scoring

    monkeypatch.setattr(greyscope.scoring, "batch_logits", fake_batch_logits)
    scorer = StandaloneScorer("model", "tok", max_length=512, batch_size=4)
    logits = _predict_bucket_logits(scorer, {"prompt": ["a", "b"]})

    assert logits.shape == (2, 3)
    assert captured == {"model": "model", "tok": "tok", "prompts": ["a", "b"],
                        "max_length": 512, "batch_size": 4}
