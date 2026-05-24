from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    return data


def get_device(cfg: dict[str, Any]) -> str:
    device = str(cfg.get("device", "cpu"))
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except ImportError:
            return "cpu"
    return device
