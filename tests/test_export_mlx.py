import json
import importlib.util

from greyscope.mlx_model import _quantization_for_path, _text_classifier_weight_keys
from greyscope.mlx_export import stage_model


def test_stage_model_adds_custom_classifier_without_mutating_source(tmp_path):
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    original = {
        "model_type": "qwen3_5_text",
        "n_buckets": 4,
        "quantization_config": {"bits": 4},
    }
    (source / "config.json").write_text(json.dumps(original))
    (source / "model.safetensors").write_bytes(b"weights")

    stage_model(source, stage)

    staged = json.loads((stage / "config.json").read_text())
    assert staged["model_file"] == "mlx_model.py"
    assert "quantization_config" not in staged
    assert json.loads((source / "config.json").read_text()) == original
    assert (stage / "model.safetensors").is_symlink()
    assert (stage / "mlx_model.py").is_file()


def test_custom_model_supports_mlx_dynamic_import(tmp_path):
    """Mirror mlx-lm's loader, which does not register the module in sys.modules."""
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}))
    stage_model(source, stage)

    spec = importlib.util.spec_from_file_location("custom_model", stage / "mlx_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.ModelArgs.num_labels == 3


def test_mlx_classifier_remaps_transformers_backbone_keys():
    weights = {
        "model.language_model.layers.0.mlp.up_proj.weight": "backbone",
        "score.weight": "head",
        "model.visual.patch_embed.weight": "vision",
    }

    assert _text_classifier_weight_keys(weights) == {
        "model.layers.0.mlp.up_proj.weight": "backbone",
        "score.weight": "head",
    }


def test_mlx_quantization_matches_validated_int4_boundary():
    assert _quantization_for_path("model.layers.0.mlp.up_proj") is True
    assert _quantization_for_path("model.layers.0.linear_attn.in_proj_qkv") is True
    assert _quantization_for_path("model.layers.3.self_attn.q_proj") is True
    assert _quantization_for_path("model.embed_tokens") is True
    assert _quantization_for_path("model.layers.0.linear_attn.in_proj_a") is False
    assert _quantization_for_path("score") is False
