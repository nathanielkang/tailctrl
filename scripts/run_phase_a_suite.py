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
from tailctrl.phase_a import run_phase_a


def _write_status(path: Path, *, datasets: list[str], completed: list[str], current: str, failed: list[str]) -> None:
    pct = 0.0 if not datasets else 100.0 * len(completed) / len(datasets)
    write_json_report(
        path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "A",
            "datasets_total": len(datasets),
            "completed": completed,
            "failed": failed,
            "current": current,
            "pct": pct,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TailCtrl Phase-A (ERM phenomenon) on the 12-dataset suite.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "tailctrl_default.yaml"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs" / "phase_a_suite"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    datasets = list(args.datasets) if args.datasets else list(cfg.get("benchmarks", {}).get("datasets", DATASET_12_DEFAULT))

    completed: list[str] = []
    failed: list[str] = []
    status_path = out_dir / "status.json"
    _write_status(status_path, datasets=datasets, completed=completed, current="starting", failed=failed)

    for ds_name in datasets:
        _write_status(status_path, datasets=datasets, completed=completed, current=ds_name, failed=failed)
        run_cfg = copy.deepcopy(cfg)
        run_cfg["data"]["source"] = "named_tabular"
        run_cfg["data"]["dataset_name"] = ds_name
        try:
            ds_out = out_dir / ds_name
            result = run_phase_a(run_cfg, ds_out, seed)
            completed.append(ds_name)
            append_csv_row(
                out_dir / "phase_a_suite_summary.csv",
                {
                    "dataset": ds_name,
                    "seed": seed,
                    "delta_rho": result["delta_rho"],
                    "rho_dense": result["rho_dense"],
                    "rho_tail": result["rho_tail"],
                    "loss_drop": result["loss_drop"],
                },
            )
        except Exception as exc:  # pragma: no cover
            failed.append(ds_name)
            write_json_report(
                out_dir / f"{ds_name}_error.json",
                {
                    "dataset": ds_name,
                    "seed": seed,
                    "error": str(exc),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        _write_status(status_path, datasets=datasets, completed=completed, current=ds_name, failed=failed)

    final = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "A",
        "seed": seed,
        "datasets": datasets,
        "completed": completed,
        "failed": failed,
    }
    write_json_report(out_dir / "phase_a_suite_done.json", final)
    print(json.dumps(final, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
