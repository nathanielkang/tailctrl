from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def compute_split_hash(
    split_or_train: dict[str, list[int]] | list[int] | tuple[int, ...],
    val_idx: list[int] | tuple[int, ...] | None = None,
) -> str:
    if val_idx is None and isinstance(split_or_train, dict):
        payload = json.dumps(split_or_train, sort_keys=True, separators=(",", ":"))
    else:
        payload = json.dumps(
            {"train_idx": list(split_or_train), "val_idx": list(val_idx or ())},
            separators=(",", ":"),
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def compute_config_hash(cfg: dict[str, Any]) -> str:
    payload = yaml.dump(cfg, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_config_hash_from_path(path: str | Path) -> str:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    return compute_config_hash(data)


def get_git_commit(repo_root: str | Path | None = None) -> str:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
