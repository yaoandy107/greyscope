"""Training entrypoint (Hydra). Binds configs/train.yaml, the shipped recipe.

    python scripts/train.py                              # the shipped recipe
    python scripts/train.py training.num_train_epochs=1  # ad-hoc override

Smaller scales (smoke / ablation) are the same recipe with overrides; see the
SMOKE_OVERRIDES / ABLATION_OVERRIDES presets in modal/train.py.
"""

from __future__ import annotations

# unsloth must be imported before transformers/peft for its kernel patches.
# try/except so the file still imports on a Mac without unsloth.
try:
    import unsloth  # noqa: F401
except (ImportError, RuntimeError):
    pass

import json
import logging
import math
import os
import sys
from pathlib import Path

# make the package importable when run directly (`python scripts/train.py …`)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hydra  # noqa: E402

log = logging.getLogger(__name__)


def _train(cfg) -> None:
    from transformers import EarlyStoppingCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    from greyscope.collator import DataCollatorForSeqCls
    from greyscope.data import prepare_data
    from greyscope.model import load_model_and_tokenizer
    from greyscope.trainer import build_weighted_trainer_class, make_compute_metrics

    log.info("Preparing data from %s", cfg.data.dataset)
    data = prepare_data(cfg.data)
    log.info("Train=%d  Val=%d  Test=%d", len(data.train), len(data.val), len(data.test))
    log.info("Class weights: %s", data.class_weights)

    log.info("Loading model %s (seq-cls head)", cfg.model.name)
    model, tokenizer = load_model_and_tokenizer(cfg)

    weights = data.class_weights if cfg.training.use_class_weights else None
    collator = DataCollatorForSeqCls(tokenizer=tokenizer, max_length=cfg.model.max_seq_length)
    TrainerCls = build_weighted_trainer_class(weights)
    compute_metrics = make_compute_metrics(cfg.data.n_buckets)

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
    run_name = (
        f"{tag}-{model_short}-r{cfg.lora.r}"
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

    from greyscope.eval import run_ternary_eval

    log.info("Running ternary evaluation (calibrate on val, score on test)")
    ternary = run_ternary_eval(trainer, data.val, data.test, cfg.data.n_buckets)
    m = ternary["metrics"]
    log.info("Ternary macro-F1: %.4f (target ≥0.92; editlens-Llama baseline 0.895)", m["macro_f1"])
    log.info("  per-class F1 — human=%.3f / ai_generated=%.3f / ai_edited=%.3f",
             m["f1_human"], m["f1_ai_generated"], m["f1_ai_edited"])
    log.info("  thresholds — human=%.4f / ai=%.4f (val F1s: h=%.3f, ai=%.3f)",
             ternary["h_thresh"], ternary["ai_thresh"], ternary["val_h_f1"], ternary["val_ai_f1"])
    m["confusion_matrix"] = m["confusion_matrix"].tolist()  # ndarray isn't JSON-serializable
    with open(os.path.join(cfg.training.output_dir, "ternary_metrics.json"), "w") as fh:
        json.dump(ternary, fh, indent=2)


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
