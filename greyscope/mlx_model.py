"""MLX model definition for the Greyscope Qwen3.5 sequence classifier.

This file is copied beside converted weights so ``mlx_lm`` can load the
classifier instead of constructing Qwen's causal-language-model head.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs


def _text_classifier_weight_keys(weights):
    """Map Transformers' composite Qwen keys to MLX's text-only backbone."""
    sanitized = {}
    for key, value in weights.items():
        if key.startswith(("vision_tower.", "model.visual.")):
            continue
        if key.startswith("model.language_model."):
            key = "model." + key.removeprefix("model.language_model.")
        elif key.startswith("language_model."):
            key = "model." + key.removeprefix("language_model.")
        sanitized[key] = value
    return sanitized


def _quantization_for_path(path, _module=None):
    """Quantize the backbone to int4 while keeping sensitive GDN paths in bf16."""
    excluded = (
        "score",
        "linear_attn.conv1d",
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
    )
    return not any(name in path for name in excluded)


class ModelArgs(TextModelArgs):
    # Do not decorate this subclass with @dataclass. mlx-lm loads custom model
    # files without first registering them in sys.modules, which breaks Python's
    # dataclass annotation resolver. The inherited dataclass initializer is enough.
    num_labels: int = 3

    @classmethod
    def from_dict(cls, params):
        args = super().from_dict(params)
        args.num_labels = (
            params.get("num_labels")
            or len(params.get("id2label", {}))
            or params.get("n_buckets", 4) - 1
        )
        return args


class Model(TextModel):
    """Qwen3.5 backbone with Greyscope's last-token CORN score head."""

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        if hasattr(self, "lm_head"):
            del self.lm_head
        self.score = nn.Linear(args.hidden_size, args.num_labels, bias=False)

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden = self.model(inputs)
        return self.score(hidden[:, -1, :])

    def sanitize(self, weights):
        return super().sanitize(_text_classifier_weight_keys(weights))

    @property
    def quant_predicate(self):
        return _quantization_for_path
