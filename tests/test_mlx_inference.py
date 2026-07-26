import json
from pathlib import Path

import greyscope.mlx_inference as inf

CALIB = json.loads((Path(__file__).parent / "fixtures/calibration.json").read_text())


class _Array:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return _Array(self.values[index])

    def tolist(self):
        return self.values


class _Model:
    def __call__(self, _tokens):
        return _Array([[20.0, 0.0, 0.0, 0.0]])


class _Tokenizer:
    def encode(self, _prompt, **_kwargs):
        return [1, 2, 3]


def test_detect_mlx_uses_shared_calibrated_decode(monkeypatch):
    import mlx.core as mx

    monkeypatch.setattr(inf, "_load_mlx", lambda _source: (_Model(), _Tokenizer(), CALIB))
    monkeypatch.setattr(mx, "eval", lambda _value: None)
    result = inf.detect_mlx("A passage", model="local/model")
    assert result["label"] == "human"
    assert result["ai_involvement"] == 0.0


def test_default_model_is_dedicated_mlx_artifact():
    assert inf.MLX_INT4_MODEL.endswith("-mlx-4bit")


def test_mlx_precision_aliases():
    assert inf._resolve_mlx_model("q4").endswith("-mlx-4bit")
    assert inf._resolve_mlx_model("local/model") == "local/model"
