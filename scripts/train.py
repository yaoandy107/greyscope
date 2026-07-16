"""Training entrypoint (Hydra). Binds configs/train.yaml, the shipped recipe.

    python scripts/train.py                              # the shipped recipe
    python scripts/train.py training.num_train_epochs=1  # ad-hoc override

Smaller scales (smoke / ablation) are the same recipe with overrides; see the
SMOKE_OVERRIDES / ABLATION_OVERRIDES presets in modal/train.py.
"""

from __future__ import annotations

import os

# Re-parse split CSVs each run. HF `datasets` caches parsed arrow keyed by file PATH (not
# content), so a changed split at the same path (e.g. attack_paraphrase after adding human
# negatives) is otherwise served stale from the persistent hf-cache volume — the post-train
# OOD eval then reads an old AI-only slice (AUROC undefined). Point datasets at a fresh
# per-container cache. MUST precede `import unsloth`: unsloth transitively imports datasets,
# which freezes its cache-dir config at import time — setting this later has no effect.
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hfds_cache")

# unsloth must be imported before transformers/peft for its kernel patches.
# try/except so the file still imports on a Mac without unsloth.
try:
    import unsloth  # noqa: F401
except (ImportError, RuntimeError):
    pass

import json
import logging
import math
import sys
from pathlib import Path

# make the package importable when run directly (`python scripts/train.py …`)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hydra  # noqa: E402

log = logging.getLogger(__name__)


def _fmt(x) -> str:
    """Format a nullable metric (TPR@FPR is None on a single-class slice)."""
    return f"{x:.3f}" if x is not None else "n/a"


def _train(cfg) -> None:
    from transformers import EarlyStoppingCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    from greyscope.collator import DataCollatorForSeqCls
    from greyscope.data import prepare_data
    from greyscope.model import load_model_and_tokenizer
    from greyscope.trainer import (
        build_corn_trainer_class, build_sampler_trainer_class, build_weighted_trainer_class,
        make_compute_metrics, make_corn_compute_metrics,
    )

    log.info("Preparing trilingual data from %s", cfg.data.splits_dir)
    data = prepare_data(cfg.data, splits_dir=cfg.data.splits_dir)
    log.info("Train=%d  Val=%d  Test=%d", len(data.train), len(data.val), len(data.test))

    log.info("Loading model %s (seq-cls head)", cfg.model.name)
    model, tokenizer = load_model_and_tokenizer(cfg)

    collator = DataCollatorForSeqCls(tokenizer=tokenizer, max_length=cfg.model.max_seq_length)

    head = getattr(cfg.model, "head", "seqcls")
    use_sampler = getattr(cfg.training, "use_sample_weights", False)
    # Balance language AND bucket via a WeightedRandomSampler (loss stays plain CE); the
    # class-weighted-CE path balances buckets via the loss. Use one mechanism, not both. The
    # CORN head keeps the sampler but swaps cross-entropy for the ordinal conditional loss.
    if head == "corn":
        ranking_weight = getattr(cfg.model, "ranking_loss_weight", 0.0)
        log.info("CORN ordinal head: K−1 conditional logits + conditional loss%s%s.",
                 " (language+bucket sampler)" if use_sampler else "",
                 f" + hard-neg ranking loss (w={ranking_weight})" if ranking_weight > 0 else "")
        compute_metrics = make_corn_compute_metrics(cfg.data.n_buckets)
        TrainerCls = build_corn_trainer_class(
            data.sample_weights if use_sampler else None,
            ranking_weight=ranking_weight,
            ranking_margin=getattr(cfg.model, "ranking_margin", 0.25),
        )
    elif use_sampler:
        log.info("Balancing via joint language+bucket WeightedRandomSampler.")
        compute_metrics = make_compute_metrics(cfg.data.n_buckets)
        TrainerCls = build_sampler_trainer_class(data.sample_weights)
    else:
        weights = data.class_weights if cfg.training.use_class_weights else None
        log.info("Class weights: %s", weights)
        compute_metrics = make_compute_metrics(cfg.data.n_buckets)
        TrainerCls = build_weighted_trainer_class(weights)

    # HF deprecated warmup_ratio; convert it to warmup_steps.
    effective_batch = (
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps
    )
    total_steps = math.ceil(len(data.train) / effective_batch) * cfg.training.num_train_epochs
    warmup_steps = max(1, int(total_steps * cfg.training.warmup_ratio))
    log.info("Schedule: %d total steps, %d warmup steps (ratio=%.3f)",
             total_steps, warmup_steps, cfg.training.warmup_ratio)

    model_short = cfg.model.name.split("/")[-1]
    tag = os.path.basename(cfg.training.output_dir.rstrip("/")) or "run"
    cap = f"r{cfg.lora.r}"
    run_name = (
        f"{tag}-{model_short}-{cap}"
        f"-ep{cfg.training.num_train_epochs}-eb{effective_batch}"
        f"-lr{cfg.training.learning_rate:g}"
        f"-s{cfg.training.seed}"
    )
    log.info("Run name: %s", run_name)

    args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        run_name=run_name,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_steps=warmup_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        weight_decay=cfg.training.weight_decay,
        optim=cfg.training.optim,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        logging_steps=cfg.training.logging_steps,
        report_to=[cfg.training.report_to] if cfg.training.report_to != "none" else [],
        seed=cfg.training.seed,
        remove_unused_columns=False,
    )

    trainer = TrainerCls(
        model=model,
        args=args,
        train_dataset=data.train,
        eval_dataset=data.val,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.training.early_stopping_patience)],
    )

    last_ckpt = (
        get_last_checkpoint(cfg.training.output_dir)
        if cfg.training.resume_from_checkpoint and Path(cfg.training.output_dir).is_dir()
        else None
    )
    if last_ckpt:
        log.info("Resuming from checkpoint: %s", last_ckpt)
    log.info("Starting training")
    trainer.train(resume_from_checkpoint=last_ckpt)
    log.info("Training complete")

    final_metrics = trainer.evaluate()
    log.info("Final eval metrics: %s", final_metrics)
    Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(cfg.training.output_dir, "final_metrics.json"), "w") as fh:
        json.dump(final_metrics, fh, indent=2)

    from greyscope.eval import eval_ood_splits, run_ternary_eval

    log.info("Running ternary evaluation (calibrate on val, score on test)")
    ternary = run_ternary_eval(trainer, data.val, data.test, cfg.data.n_buckets, head=head)
    m = ternary["metrics"]
    d = ternary["detection"]
    log.info("Ternary macro-F1: %.4f (target ≥0.92; editlens-Llama baseline 0.895)", m["macro_f1"])
    log.info("  per-class F1 — human=%.3f / ai_generated=%.3f / ai_edited=%.3f",
             m["f1_human"], m["f1_ai_generated"], m["f1_ai_edited"])
    log.info("  DETECTION (human vs any-AI) — AUROC=%.4f  TPR@FPR1%%=%s  TPR@FPR5%%=%s",
             d["auroc"], _fmt(d["tpr@fpr1"]), _fmt(d["tpr@fpr5"]))
    log.info("  thresholds — human=%.4f / ai=%.4f (val F1s: h=%.3f, ai=%.3f)",
             ternary["h_thresh"], ternary["ai_thresh"], ternary["val_h_f1"], ternary["val_ai_f1"])
    for lang, pl in ternary.get("per_language", {}).items():
        pd = pl["detection"]
        log.info("  [%s] macro-F1=%.4f (h=%.3f / ai=%.3f / ed=%.3f) · AUROC=%.4f TPR@FPR1%%=%s (n=%d)",
                 lang, pl["macro_f1"], pl["f1_human"], pl["f1_ai_generated"], pl["f1_ai_edited"],
                 pd["auroc"], _fmt(pd["tpr@fpr1"]), pl["n"])
    m["confusion_matrix"] = m["confusion_matrix"].tolist()  # ndarray isn't JSON-serializable
    with open(os.path.join(cfg.training.output_dir, "ternary_metrics.json"), "w") as fh:
        json.dump(ternary, fh, indent=2)

    # OOD / generalization — the metric that actually decides recipe choices (selecting on
    # in-domain is the mistake this avoids). Same val-calibrated thresholds on the held-out splits.
    ood = eval_ood_splits(
        trainer, cfg.data.splits_dir,
        ["test_llama", "test_enron", "ood_generator", "attack_paraphrase"],
        cfg.data.n_buckets, head=head, flip=ternary["score_flipped"],
        h_thresh=ternary["h_thresh"], ai_thresh=ternary["ai_thresh"], limit=1200)
    log.info("=== OOD / generalization (val-calibrated thresholds) ===")
    for name, r in ood.items():
        rd = r["detection"]
        pl = " ".join(f"{k}={v['macro_f1']:.3f}(n={v['n']})" for k, v in r["per_language"].items())
        log.info("[OOD %-13s] macro-F1=%.4f · DETECTION AUROC=%s TPR@FPR1%%=%s TPR@FPR5%%=%s (n=%d)  %s",
                 name, r["macro_f1"], _fmt(rd["auroc"]), _fmt(rd["tpr@fpr1"]), _fmt(rd["tpr@fpr5"]), r["n"], pl)
    with open(os.path.join(cfg.training.output_dir, "ood_metrics.json"), "w") as fh:
        json.dump(ood, fh, indent=2)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _train(cfg)


if __name__ == "__main__":
    main()
