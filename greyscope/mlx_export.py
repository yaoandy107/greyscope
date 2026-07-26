"""Shared staging helpers for native MLX conversion and parity checks."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def stage_model(source: Path, stage: Path) -> None:
    """Create a cheap symlinked snapshot with the MLX classifier definition."""
    source = source.resolve()
    for item in source.iterdir():
        os.symlink(item, stage / item.name)

    config_path = stage / "config.json"
    config = json.loads(config_path.read_text())
    config["model_file"] = "mlx_model.py"
    config.pop("quantization_config", None)
    config_path.unlink()
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(Path(__file__).with_name("mlx_model.py"), stage / "mlx_model.py")
