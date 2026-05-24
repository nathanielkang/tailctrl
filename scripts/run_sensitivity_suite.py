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


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensitivity sweeps for TailCtrl hyperparameters (Phase B).")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "tailctrl_default.yaml"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs" / "sensitivity"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="*", default=["kin8nm", "abalone", "diabetes"])
    parser.add_argument("--variants", nargs="*", default=None, help="Variant tags; default = built-in grid.")
    args = parser.parse_args()

    cfg_base = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(args.datasets)

    grid: list[tuple[str, dict]] = [
        ("K1", {"training": {"inner_steps_k": 1}}),
        ("K2", {"training": {"inner_steps_k": 2}}),
        ("K3", {"training": {"inner_steps_k": 3}}),
        ("K5", {"training": {"inner_steps_k": 5}}),
        ("K8", {"training": {"inner_steps_k": 8}}),
        ("alpha0.1", {"training": {"alpha_sparsity": 0.1}}),
        ("alpha0.5", {"training": {"alpha_sparsity": 0.5}}),
        ("alpha1.0", {"training": {"alpha_sparsity": 1.0}}),
        ("alpha2.0", {"training": {"alpha_sparsity": 2.0}}),
        ("tail_q10", {"strata": {"tail_quantile": 0.10, "dense_quantile": 0.90}}),
        ("tail_q20", {"strata": {"tail_quantile": 0.20, "dense_quantile": 0.80}}),
    ]
    if args.variants:
        grid = [g for g in grid if g[0] in set(args.variants)]

    done = 0
    total = len(grid) * len(datasets)
    for tag, delta in grid:
        for ds in datasets:
            run_cfg = copy.deepcopy(cfg_base)
            run_cfg["data"]["source"] = "named_tabular"
            run_cfg["data"]["dataset_name"] = ds
            run_cfg["phase_b"]["methods"] = ["erm", "tailctrl"]
            if "training" in delta:
                run_cfg["training"].update(delta["training"])
            if "strata" in delta:
                run_cfg["strata"].update(delta["strata"])
            ds_out = out_dir / tag / ds
            try:
                result = run_phase_b(run_cfg, ds_out, int(args.seed))
                append_csv_row(
                    out_dir / "sensitivity_summary.csv",
                    {
                        "variant": tag,
                        "dataset": ds,
                        "seed": int(args.seed),
                        "tailctrl_tail_mae": result["methods"]["tailctrl"]["test_tail_mae"],
                        "erm_tail_mae": result["methods"]["erm"]["test_tail_mae"],
                        "inner_steps_k": run_cfg["training"]["inner_steps_k"],
                        "alpha_sparsity": run_cfg["training"]["alpha_sparsity"],
                        "tail_quantile": run_cfg["strata"]["tail_quantile"],
                    },
                )
            except Exception as exc:  # pragma: no cover
                write_json_report(out_dir / f"{tag}_{ds}_error.json", {"error": str(exc), "variant": tag, "dataset": ds})
            done += 1
            write_json_report(
                out_dir / "status.json",
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "jobs_done": done,
                    "jobs_total": total,
                    "current": f"{tag}/{ds}",
                },
            )

    write_json_report(out_dir / "sensitivity_done.json", {"jobs_done": done, "jobs_total": total, "seed": args.seed})
    print(json.dumps({"done": done, "total": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
