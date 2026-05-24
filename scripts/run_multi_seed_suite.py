from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tailctrl.config import load_config
from tailctrl.data import DATASET_12_DEFAULT
from tailctrl.logging_utils import append_csv_row, compute_config_hash, get_git_commit, write_json_report
from tailctrl.phase_b import run_phase_b


def _run_one_seed(
    cfg: dict,
    datasets: list[str],
    out_dir: Path,
    seed: int,
) -> tuple[list[str], list[str]]:
    completed: list[str] = []
    failed: list[str] = []
    for ds_name in datasets:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["data"]["source"] = "named_tabular"
        run_cfg["data"]["dataset_name"] = ds_name
        try:
            ds_out = out_dir / f"seed{seed}" / ds_name
            result = run_phase_b(run_cfg, ds_out, seed)
            completed.append(ds_name)
            for method, metrics in result["methods"].items():
                append_csv_row(
                    out_dir / "all_runs.csv",
                    {
                        "dataset": ds_name,
                        "seed": seed,
                        "method": method,
                        "test_tail_mae": metrics["test_tail_mae"],
                        "test_dense_mae": metrics.get("test_dense_mae"),
                        "test_mae": metrics["test_mae"],
                        "wall_clock_sec": metrics["wall_clock_sec"],
                        "forward_passes": metrics["forward_passes"],
                        "policy_variance": metrics["policy_variance"],
                        "config_hash": compute_config_hash(run_cfg),
                        "git_commit": get_git_commit(ROOT),
                    },
                )
            append_csv_row(
                out_dir / "suite_summary_by_seed.csv",
                {
                    "dataset": ds_name,
                    "seed": seed,
                    "tailctrl_tail_mae": result["methods"]["tailctrl"]["test_tail_mae"],
                    "erm_tail_mae": result["methods"]["erm"]["test_tail_mae"],
                    "tailctrl_beats_erm": int(result["tailctrl_beats_erm_tail_mae"]),
                },
            )
        except Exception as exc:  # pragma: no cover
            failed.append(f"{ds_name}_s{seed}")
            write_json_report(
                out_dir / f"seed{seed}" / f"{ds_name}_error.json",
                {"dataset": ds_name, "seed": seed, "error": str(exc)},
            )
    return completed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase-B suite across multiple seeds with unified CSV ledger.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "tailctrl_default.yaml"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs" / "multi_seed_suite"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--parallel", action="store_true", help="Launch one subprocess per seed (Windows-safe).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = list(args.datasets) if args.datasets else list(cfg.get("benchmarks", {}).get("datasets", DATASET_12_DEFAULT))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.parallel and len(args.seeds) > 1:
        procs = []
        for seed in args.seeds:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                args.config,
                "--out-dir",
                str(out_dir),
                "--seeds",
                str(seed),
            ]
            if args.datasets:
                cmd.extend(["--datasets", *args.datasets])
            procs.append(subprocess.Popen(cmd))
        rc = max(p.wait() for p in procs)
        print(f"parallel seeds done rc={rc}")
        return rc

    all_completed: list[str] = []
    all_failed: list[str] = []
    for seed in args.seeds:
        completed, failed = _run_one_seed(cfg, datasets, out_dir, seed)
        all_completed.extend(completed)
        all_failed.extend(failed)
        write_json_report(
            out_dir / f"seed{seed}_done.json",
            {
                "seed": seed,
                "completed": completed,
                "failed": failed,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    final = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds,
        "datasets": datasets,
        "n_completed_jobs": len(all_completed),
        "n_failed_jobs": len(all_failed),
        "config_hash": compute_config_hash(cfg),
        "git_commit": get_git_commit(ROOT),
    }
    write_json_report(out_dir / "multi_seed_done.json", final)
    print(json.dumps(final, indent=2))
    return 0 if not all_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
