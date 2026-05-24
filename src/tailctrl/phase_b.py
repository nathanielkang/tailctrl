from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tailctrl.bilevel import TailCtrlTrainer
from tailctrl.config import get_device
from tailctrl.data import density_from_train, fit_stratum_labels, four_way_split, load_dataset
from tailctrl.gates import evaluate_hard_gates
from tailctrl.logging_utils import append_csv_row, compute_split_hash, write_json_report
from tailctrl.types import FourWaySplit


def _filter_split_indices(idx: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    return idx[np.isin(idx, allowed)]


def _apply_train_subset_mode(
    split: FourWaySplit,
    density: np.ndarray,
    q_low: float,
    mode: str,
) -> FourWaySplit:
    if mode == "whole":
        return split
    tail_idx = np.where(density <= q_low)[0]
    if mode == "tail_train_only":
        adapted = FourWaySplit(
            meta_train=_filter_split_indices(split.meta_train, tail_idx),
            meta_val=_filter_split_indices(split.meta_val, tail_idx),
            model_val=_filter_split_indices(split.model_val, tail_idx),
            test=split.test,
        )
    elif mode == "tail_all":
        adapted = FourWaySplit(
            meta_train=_filter_split_indices(split.meta_train, tail_idx),
            meta_val=_filter_split_indices(split.meta_val, tail_idx),
            model_val=_filter_split_indices(split.model_val, tail_idx),
            test=_filter_split_indices(split.test, tail_idx),
        )
    else:
        raise ValueError(f"Unknown experiment.train_subset mode: {mode}")

    sizes = {
        "meta_train": adapted.meta_train.size,
        "meta_val": adapted.meta_val.size,
        "model_val": adapted.model_val.size,
        "test": adapted.test.size,
    }
    if min(sizes.values()) < 8:
        raise RuntimeError(f"Subset mode '{mode}' produced too-small split sizes: {sizes}")
    return adapted


def run_phase_b(cfg: dict[str, Any], out_dir: Path, seed: int) -> dict[str, Any]:
    device = get_device(cfg)
    data_cfg = cfg["data"]
    methods = list(cfg["phase_b"]["methods"])

    dataset = load_dataset(data_cfg, seed)
    split = four_way_split(dataset.x.shape[0], data_cfg["split_ratios"], seed)
    if not split.verify_disjoint():
        raise RuntimeError("Splits are not disjoint")

    labels = fit_stratum_labels(
        dataset.y[split.meta_train],
        tail_quantile=float(cfg["strata"]["tail_quantile"]),
        dense_quantile=float(cfg["strata"]["dense_quantile"]),
        bandwidth=cfg["strata"].get("kde_bandwidth"),
    )
    subset_mode = str(cfg.get("experiment", {}).get("train_subset", "whole"))
    density = density_from_train(
        dataset.y[split.meta_train],
        dataset.y,
        bandwidth=cfg["strata"].get("kde_bandwidth"),
    )
    split = _apply_train_subset_mode(split, density, labels.q_low, subset_mode)

    trainer = TailCtrlTrainer(cfg, device)
    results: dict[str, Any] = {}
    budgets: dict[str, float] = {}

    for method in methods:
        res = trainer.train(dataset, split, labels, method, seed=seed)
        results[method] = {
            "test_tail_mae": res.test_tail_mae,
            "test_dense_mae": res.test_dense_mae,
            "test_mae": res.test_mae,
            "val_tail_mae": res.val_tail_mae,
            "policy_variance": res.policy_variance,
            "train_loss_first": res.train_loss_trace[0] if res.train_loss_trace else 0.0,
            "train_loss_last": res.train_loss_trace[-1] if res.train_loss_trace else 0.0,
            "wall_clock_sec": res.budget.wall_clock_sec,
            "forward_passes": res.budget.forward_passes,
        }
        budgets[method] = res.budget.wall_clock_sec
        append_csv_row(
            out_dir / "phase_b_runs.csv",
            {
                "seed": seed,
                "method": method,
                "test_tail_mae": res.test_tail_mae,
                "test_dense_mae": res.test_dense_mae,
                "test_mae": res.test_mae,
                "policy_variance": res.policy_variance,
                "wall_clock_sec": res.budget.wall_clock_sec,
            },
        )

    baseline_time = budgets.get("fixed_floor") or budgets.get("erm") or 1.0
    tailctrl_time = budgets.get("tailctrl", baseline_time)
    budget_ratio = tailctrl_time / max(baseline_time, 1e-9)

    tail = results.get("tailctrl", {})
    gates = evaluate_hard_gates(
        split_disjoint=split.verify_disjoint(),
        policy_variance=float(tail.get("policy_variance", 0.0)),
        policy_variance_eps=float(cfg["gates"]["policy_variance_eps"]),
        budget_ratio=budget_ratio,
        budget_ratio_low=float(cfg["gates"]["budget_ratio_low"]),
        budget_ratio_high=float(cfg["gates"]["budget_ratio_high"]),
        first_loss=float(tail.get("train_loss_first", 0.0)),
        last_loss=float(tail.get("train_loss_last", 0.0)),
        min_loss_drop=float(cfg["gates"]["min_loss_drop"]),
    )

    payload = {
        "phase": "B",
        "seed": seed,
        "dataset": dataset.name,
        "experiment_train_subset": subset_mode,
        "split_sizes": {
            "meta_train": int(split.meta_train.size),
            "meta_val": int(split.meta_val.size),
            "model_val": int(split.model_val.size),
            "test": int(split.test.size),
        },
        "split_hash": compute_split_hash(split.as_dict()),
        "methods": results,
        "budget_ratio_tailctrl_vs_fixed": budget_ratio,
        "gates": gates.as_dict(),
        "tailctrl_beats_erm_tail_mae": (
            results.get("tailctrl", {}).get("test_tail_mae", 1e9)
            < results.get("erm", {}).get("test_tail_mae", -1e9)
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_report(out_dir / f"phase_b_seed{seed}.json", payload)
    policy_path = out_dir / f"policy_trace_seed{seed}.json"
    policy_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return payload
