"""Decode / threshold / mode logic for `greyscope.inference.detect`.

The merged model isn't shipped in-repo, so these stub the model and tokenizer
and exercise the shipped calibration.json (fixtures/ carries a copy of the
published one): the score math, both label modes, and the output schema —
without loading any weights.
"""
import json
import math
from pathlib import Path

import pytest
import torch

import greyscope.inference as inf

CALIB = json.loads((Path(__file__).parent / "fixtures/calibration.json").read_text())


class _Enc(dict):
    def to(self, _device):
        return self


class _Tok:
    pad_token_id = 0

    def __call__(self, _prompt, **_kw):
        return _Enc(input_ids=torch.zeros(1, 4, dtype=torch.long))


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _Model:
    device = torch.device("cpu")

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, **_kw):
        return _Out(self._logits)


def _detect(monkeypatch, bucket_logits, **kw):
    logits = torch.tensor([bucket_logits], dtype=torch.bfloat16)  # bf16, like the shipped model
    monkeypatch.setattr(inf, "_load", lambda *_args: (_Model(logits), _Tok(), CALIB))
    kw.setdefault("model", "bf16")
    return inf.detect("A sample passage to classify.", **kw)


def test_output_schema(monkeypatch):
    r = _detect(monkeypatch, [20.0, 0.0, 0.0, 0.0])
    assert set(r) == {"label", "ai_involvement", "bucket_probs"}
    assert list(r["bucket_probs"]) == CALIB["bucket_descriptions"]
    assert 0.0 <= r["ai_involvement"] <= 1.0
    assert abs(sum(r["bucket_probs"].values()) - 1.0) < 0.02
    json.dumps(r)  # must be JSON-serializable


def test_ternary_human(monkeypatch):
    r = _detect(monkeypatch, [20.0, 0.0, 0.0, 0.0])
    assert r["label"] == "human"
    assert r["ai_involvement"] == 0.0


def test_ternary_generated(monkeypatch):
    r = _detect(monkeypatch, [0.0, 0.0, 0.0, 20.0])
    assert r["label"] == "AI-generated"
    assert r["ai_involvement"] == 1.0


def test_ternary_edited(monkeypatch):
    r = _detect(monkeypatch, [0.0, 0.0, 8.0, 2.0])
    assert r["label"] == "AI-edited"


def test_binary_extremes(monkeypatch):
    human = _detect(monkeypatch, [20.0, 0.0, 0.0, 0.0], mode="binary")
    ai = _detect(monkeypatch, [0.0, 0.0, 0.0, 20.0], mode="binary")
    assert human["label"] == "human"
    assert ai["label"] == "AI"
    assert "is_ai" not in human  # no contradictory dual output


def test_modes_diverge_in_grey_zone(monkeypatch):
    # A lightly-edited score (between h_thresh and the calibrated binary
    # threshold) is "AI-edited" in ternary but stays "human" in binary — the
    # exact case the two modes exist to disambiguate.
    logits = [math.log(0.30), math.log(0.45), math.log(0.20), math.log(0.05)]
    tern = _detect(monkeypatch, logits, mode="ternary")
    binr = _detect(monkeypatch, logits, mode="binary")
    assert tern["label"] == "AI-edited"
    assert binr["label"] == "human"
    assert CALIB["h_thresh"] < tern["ai_involvement"] < CALIB["binary_threshold"]


def test_invalid_mode(monkeypatch):
    with pytest.raises(ValueError):
        _detect(monkeypatch, [1.0, 0.0, 0.0, 0.0], mode="nonsense")


# --- CORN head: K−1 conditional logits, cumulative-sigmoid decode ---

CORN_CALIB = {
    "head_type": "corn", "n_buckets": 4,
    "label_names": ["human", "AI-generated", "AI-edited"],
    "bucket_descriptions": ["none", "light", "moderate", "heavy"],
    "flip": False, "score_min": 0.0, "score_max": 1.0,
    "h_thresh": 0.2, "ai_thresh": 0.8, "binary_threshold": 0.5,
    "binary_fpr_target": 0.01, "prompt_template": CALIB["prompt_template"],
    "lowercase": True, "max_length": 2048,
}


def _detect_corn(monkeypatch, cond_logits, **kw):
    logits = torch.tensor([cond_logits], dtype=torch.bfloat16)  # K−1 conditional logits
    monkeypatch.setattr(inf, "_load", lambda *_args: (_Model(logits), _Tok(), CORN_CALIB))
    kw.setdefault("model", "bf16")
    return inf.detect("A sample passage to classify.", **kw)


def test_corn_head_decodes_extremes(monkeypatch):
    hi = _detect_corn(monkeypatch, [20.0, 20.0, 20.0])
    lo = _detect_corn(monkeypatch, [-20.0, -20.0, -20.0])
    assert hi["label"] == "AI-generated" and hi["ai_involvement"] == 1.0
    assert lo["label"] == "human" and lo["ai_involvement"] == 0.0
    # 3 conditional logits still decode to 4 discrete bucket probs that sum to 1
    assert list(hi["bucket_probs"]) == CORN_CALIB["bucket_descriptions"]
    assert abs(sum(hi["bucket_probs"].values()) - 1.0) < 0.02
    json.dumps(hi)


def test_resolve_model_aliases():
    assert inf.resolve_model("bf16") == inf.BF16_MODEL
    assert inf.resolve_model("int4") == inf.INT4_MODEL
    assert inf.resolve_model("org/custom") == "org/custom"


def test_transformers_auto_alias_is_reference_bf16():
    assert inf.resolve_model() == inf.BF16_MODEL


def test_auto_uses_mlx_q4_on_apple_silicon(monkeypatch):
    import greyscope.mlx_inference as mlx_inf

    seen = {}
    monkeypatch.setattr(inf, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(inf, "_mlx_available", lambda: True)
    monkeypatch.setattr(
        mlx_inf,
        "detect_mlx",
        lambda text, mode, model: seen.update(text=text, mode=mode, model=model) or {"ok": True},
    )

    assert inf.detect("Text") == {"ok": True}
    assert seen == {"text": "Text", "mode": "ternary", "model": "q4"}


def test_auto_falls_back_to_transformers_without_mlx(monkeypatch):
    logits = torch.tensor([[20.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16)
    seen = {}

    monkeypatch.setattr(inf, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(inf, "_mlx_available", lambda: False)

    def fake_load(source, device):
        seen.update(source=source, device=device)
        return _Model(logits), _Tok(), CALIB

    monkeypatch.setattr(inf, "_load", fake_load)
    inf.detect("Text")
    assert seen == {"source": inf.BF16_MODEL, "device": "auto"}


def test_local_mlx_artifact_uses_mlx_backend(tmp_path, monkeypatch):
    import greyscope.mlx_inference as mlx_inf

    model_dir = tmp_path / "native"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_file": "mlx_model.py"}')
    seen = {}
    monkeypatch.setattr(
        mlx_inf,
        "detect_mlx",
        lambda text, mode, model: seen.update(text=text, mode=mode, model=model) or {"ok": True},
    )

    assert inf.detect("Text", model=str(model_dir)) == {"ok": True}
    assert seen["model"] == str(model_dir)


def test_detect_passes_model_and_device_to_loader(monkeypatch):
    logits = torch.tensor([[20.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16)
    seen = {}

    def fake_load(source, device):
        seen.update(source=source, device=device)
        return _Model(logits), _Tok(), CALIB

    monkeypatch.setattr(inf, "_load", fake_load)
    result = inf.detect("Text", model="int4", device="cpu")
    assert result["label"] == "human"
    assert seen == {"source": inf.INT4_MODEL, "device": "cpu"}
