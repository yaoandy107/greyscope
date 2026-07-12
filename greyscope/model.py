"""Model loading + LoRA setup for the 4-bucket seq-cls head (Unsloth FastModel, CUDA)."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _ensure_score_head(model, n_labels: int) -> None:
    """Resize the `score` head to n_labels.

    The Qwen 3.5 seq-cls loader ignores `num_labels=` and leaves a 2-class head;
    it's randomly initialized anyway, so resizing here, before get_peft_model wraps
    it via modules_to_save, is lossless.
    """
    head = model.score
    if head.out_features != n_labels:
        import torch.nn as nn

        log.info("Resizing seq-cls score head %d → %d classes.", head.out_features, n_labels)
        model.score = nn.Linear(
            head.in_features, n_labels, bias=head.bias is not None,
        ).to(device=head.weight.device, dtype=head.weight.dtype)
    model.config.num_labels = n_labels
    model.num_labels = n_labels


def load_model_and_tokenizer(cfg) -> tuple[Any, Any]:
    """Load Qwen 3.5 via Unsloth FastModel with a 4-bucket seq-cls head + LoRA.

    `modules_to_save=["score"]` is mandatory: the head is randomly initialized, so
    without it the frozen head never trains and the model predicts one class.
    """
    import torch
    from transformers import AutoModelForSequenceClassification
    from unsloth import FastModel

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[cfg.model.dtype]

    # CORN ordinal head emits K−1 conditional logits; seq-cls emits K. n_buckets and the
    # head type ride in the saved config so inference picks the right decode.
    head = getattr(cfg.model, "head", "seqcls")
    n_out = (cfg.data.n_buckets - 1) if head == "corn" else cfg.data.n_buckets

    log.info("Loading %s via FastModel head=%s (out=%d, n_buckets=%d, dtype=%s).",
             cfg.model.name, head, n_out, cfg.data.n_buckets, cfg.model.dtype)
    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
        num_labels=n_out,
        auto_model=AutoModelForSequenceClassification,
        use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # seq-cls head reads the last non-pad token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.head_type = head
    model.config.n_buckets = cfg.data.n_buckets

    _ensure_score_head(model, n_out)

    # target_modules is an explicit list or the string "all-linear" (PEFT special-cases the
    # latter to every nn.Linear).
    tm = cfg.lora.target_modules
    target_modules = tm if isinstance(tm, str) else list(tm)

    model = FastModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        target_modules=target_modules,
        use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
        task_type="SEQ_CLS",
        modules_to_save=["score"],
    )
    return model, tokenizer
