from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tailctrl.config import load_config
from tailctrl.data import DATASET_12_DEFAULT
from tailctrl.logging_utils import append_csv_row, write_json_report
from tailctrl.phase_b import run_phase_b


def _status(path: Path, datasets: list[str], completed: list[str], failed: list[str], current: str, stage: str) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "datasets_total": len(datasets),
        "completed": completed,
        "failed": failed,
        "current": current,
        "pct": 0.0 if not datasets else 100.0 * len(completed) / len(datasets),
    }
    write_json_report(path, payload)


def _run_suite(stage_name: str, cfg: dict, datasets: list[str], out_dir: Path, seed: int) -> tuple[list[str], list[str]]:
    completed: list[str] = []
    failed: list[str] = []
    status_path = out_dir / "status.json"
    _status(status_path, datasets, completed, failed, "starting", stage_name)

    for ds in datasets:
        _status(status_path, datasets, completed, failed, ds, stage_name)
        run_cfg = copy.deepcopy(cfg)
        run_cfg["data"]["source"] = "named_tabular"
        run_cfg["data"]["dataset_name"] = ds
        try:
            result = run_phase_b(run_cfg, out_dir / ds, seed)
            completed.append(ds)
            append_csv_row(
                out_dir / "suite_summary.csv",
                {
                    "dataset": ds,
                    "seed": seed,
                    "tailctrl_tail_mae": result["methods"]["tailctrl"]["test_tail_mae"],
                    "erm_tail_mae": result["methods"]["erm"]["test_tail_mae"],
                    "tailctrl_beats_erm_tail_mae": int(result["tailctrl_beats_erm_tail_mae"]),
                    "policy_variance": result["methods"]["tailctrl"]["policy_variance"],
                    "split_leakage_flag": result["gates"]["split_leakage_flag"],
                    "subset_mode": result["experiment_train_subset"],
                },
            )
        except Exception as exc:  # pragma: no cover
            failed.append(ds)
            write_json_report(
                out_dir / f"{ds}_error.json",
                {
                    "dataset": ds,
                    "seed": seed,
                    "stage": stage_name,
                    "error": str(exc),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        _status(status_path, datasets, completed, failed, ds, stage_name)
    return completed, failed


def _apply_tuned_overrides(cfg: dict) -> None:
    cfg["training"]["outer_steps"] = 60
    cfg["training"]["inner_steps_k"] = max(int(cfg["training"]["inner_steps_k"]), 8)
    cfg["training"]["inner_lr"] = 5.0e-4
    cfg["training"]["outer_lr"] = 2.0e-4
    cfg["training"]["alpha_sparsity"] = 0.2
    cfg["training"]["m_tail"] = 1.5
    cfg["training"]["beta_gap_var"] = 0.05
    cfg["training"]["tail_mask_mid"] = 0.02
    cfg["training"]["tail_mask_dense"] = 0.02


def main() -> int:
    parser = argparse.ArgumentParser(description="Run required TailCtrl experiments: whole, tail-only, ablation sensitivity.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "tailctrl_default.yaml"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs" / "required_experiments"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    cfg_base = load_config(args.config)
    _apply_tuned_overrides(cfg_base)
    seed = int(args.seed if args.seed is not None else cfg_base["seed"])
    datasets = list(args.datasets) if args.datasets else list(cfg_base.get("benchmarks", {}).get("datasets", DATASET_12_DEFAULT))

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    # (a) whole dataset
    cfg_whole = copy.deepcopy(cfg_base)
    cfg_whole.setdefault("experiment", {})["train_subset"] = "whole"
    cfg_whole["phase_b"]["methods"] = ["erm", "tailctrl"]
    whole_dir = root / "a_whole_dataset"
    whole_completed, whole_failed = _run_suite("whole_dataset", cfg_whole, datasets, whole_dir, seed)

    # (b) imbalanced/tail data points only
    cfg_tail = copy.deepcopy(cfg_base)
    cfg_tail.setdefault("experiment", {})["train_subset"] = "tail_all"
    cfg_tail["phase_b"]["methods"] = ["erm", "tailctrl"]
    tail_dir = root / "b_tail_only"
    tail_completed, tail_failed = _run_suite("tail_only", cfg_tail, datasets, tail_dir, seed)

    # (c) sensitivity ablation (whole-dataset mode; key hyperparameters)
    ablation_dir = root / "c_ablation_sensitivity"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ("base_tuned", {}),
        ("alpha0", {"training": {"alpha_sparsity": 0.0}}),
        ("m_tail_2", {"training": {"m_tail": 2.0}}),
        ("tail_q_010", {"strata": {"tail_quantile": 0.10, "dense_quantile": 0.90}}),
        ("tail_q_020", {"strata": {"tail_quantile": 0.20, "dense_quantile": 0.80}}),
    ]
    ablation_done = 0
    for tag, delta in variants:
        for ds in datasets:
            run_cfg = copy.deepcopy(cfg_base)
            run_cfg.setdefault("experiment", {})["train_subset"] = "whole"
            run_cfg["phase_b"]["methods"] = ["erm", "tailctrl"]
            run_cfg["data"]["source"] = "named_tabular"
            run_cfg["data"]["dataset_name"] = ds
            if "training" in delta:
                run_cfg["training"].update(delta["training"])
            if "strata" in delta:
                run_cfg["strata"].update(delta["strata"])
            out_ds = ablation_dir / tag / ds
            try:
                result = run_phase_b(run_cfg, out_ds, seed)
                append_csv_row(
                    ablation_dir / "ablation_sensitivity.csv",
                    {
                        "variant": tag,
                        "dataset": ds,
                        "seed": seed,
                        "tailctrl_tail_mae": result["methods"]["tailctrl"]["test_tail_mae"],
                        "erm_tail_mae": result["methods"]["erm"]["test_tail_mae"],
                        "tailctrl_beats_erm_tail_mae": int(result["tailctrl_beats_erm_tail_mae"]),
                        "policy_variance": result["methods"]["tailctrl"]["policy_variance"],
                    },
                )
            except Exception as exc:  # pragma: no cover
                write_json_report(
                    ablation_dir / f"{tag}_{ds}_error.json",
                    {
                        "variant": tag,
                        "dataset": ds,
                        "seed": seed,
                        "error": str(exc),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            ablation_done += 1
            write_json_report(
                ablation_dir / "status.json",
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "variants": [v[0] for v in variants],
                    "datasets_total": len(datasets),
                    "jobs_total": len(variants) * len(datasets),
                    "jobs_done": ablation_done,
                },
            )

    final = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "datasets": datasets,
        "whole_dataset": {"completed": whole_completed, "failed": whole_failed},
        "tail_only": {"completed": tail_completed, "failed": tail_failed},
        "ablation_variants": [v[0] for v in variants],
    }
    write_json_report(root / "required_experiments_done.json", final)
    print(json.dumps(final, indent=2))
    print(f"done={root / 'required_experiments_done.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
