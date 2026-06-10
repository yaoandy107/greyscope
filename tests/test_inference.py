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
    monkeypatch.setattr(inf, "_load", lambda: (_Model(logits), _Tok(), CALIB))
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
    # A lightly-edited score (between h_thresh and the accusation-safe binary
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
