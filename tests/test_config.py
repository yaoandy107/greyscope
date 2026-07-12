"""Tests for the production config + the programmatic DataConfig."""

from pathlib import Path

import pytest

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_data_config_defaults():
    from greyscope.config import DataConfig

    cfg = DataConfig()
    assert cfg.dataset == "pangram/editlens_iclr"
    assert cfg.n_buckets == 4
    assert cfg.bucket_lo_threshold == 0.03
    assert cfg.bucket_hi_threshold == 0.15
    assert cfg.label_field == "cosine_score"


def test_train_yaml_is_the_shipped_recipe():
    """train.yaml is the single source of truth; guard the load-bearing values."""
    omegaconf = pytest.importorskip("omegaconf")

    cfg = omegaconf.OmegaConf.load(CONFIGS_DIR / "train.yaml")

    assert cfg.model.name == "unsloth/Qwen3.5-4B-Base"
    assert cfg.model.head == "corn"                       # K-1 ordinal head
    assert cfg.model.ranking_loss_weight == 0.1           # MELD hard-neg ranking loss
    assert cfg.lora.r == 32 and cfg.lora.alpha == 32      # alpha=r -> scale 1.0
    assert cfg.data.splits_dir == "data/v2/splits"
    assert cfg.data.n_buckets == 4
    assert cfg.data.train_subset is None                  # full trilingual set
    assert cfg.data.sample_weight_temperature == 0.5      # joint language+bucket sampler
    assert any("train_aug_paraphrase" in f for f in cfg.data.train_extra_files)  # paraphrase aug
    assert cfg.training.num_train_epochs == 2
    assert (
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps
        == 16
    ), "effective batch must stay at 16 for recipe parity"
    assert cfg.training.use_sample_weights is True
    assert cfg.training.metric_for_best_model == "eval_detection_auroc"  # select on detection
    assert cfg.training.warmup_ratio == 0.05              # stability fix
    assert cfg.training.max_grad_norm == 1.0              # stability fix
    assert cfg.training.weight_decay == 0.01              # stability fix
    assert cfg.training.report_to == "wandb"


def test_train_yaml_loads_via_hydra():
    """train.py binds this config with plain @hydra.main (no structured schema)."""
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_DIR)):
        cfg = compose(config_name="train")
        assert cfg.model.name.startswith("unsloth/Qwen3.5")
        assert cfg.lora.use_gradient_checkpointing == "unsloth"
