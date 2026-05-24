from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tailctrl.config import load_config
from tailctrl.data import fit_stratum_labels, four_way_split, load_dataset
from tailctrl.gates import evaluate_hard_gates
from tailctrl.logging_utils import append_csv_row, compute_split_hash, write_json_report
from tailctrl.phase_a import run_phase_a
from tailctrl.phase_b import run_phase_b


def main() -> int:
    parser = argparse.ArgumentParser(description="TailCtrl full-stack smoke test.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "smoke.yaml"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    seed = int(cfg["seed"])

    # Phase A quick pass
    phase_a = run_phase_a(cfg, out_dir / "smoke", seed)

    # Phase B quick pass (all methods)
    phase_b = run_phase_b(cfg, out_dir / "smoke", seed)

    split = four_way_split(
        load_dataset(cfg["data"], seed).x.shape[0],
        cfg["data"]["split_ratios"],
        seed,
    )
    split_disjoint = split.verify_disjoint()
    tail = phase_b["methods"].get("tailctrl", {})
    gates = evaluate_hard_gates(
        split_disjoint=split_disjoint,
        policy_variance=float(tail.get("policy_variance", 0.0)),
        policy_variance_eps=float(cfg["gates"]["policy_variance_eps"]),
        budget_ratio=float(phase_b["budget_ratio_tailctrl_vs_fixed"]),
        budget_ratio_low=float(cfg["gates"]["budget_ratio_low"]),
        budget_ratio_high=float(cfg["gates"]["budget_ratio_high"]),
        first_loss=float(tail.get("train_loss_first", 0.0)),
        last_loss=float(tail.get("train_loss_last", 0.0)),
        min_loss_drop=float(cfg["gates"]["min_loss_drop"]),
    )

    payload = {
        "run_id": f"smoke_tailctrl_s{seed}",
        "method": "TailCtrl",
        "phase_a_delta_rho": phase_a["delta_rho"],
        "phase_b": phase_b,
        "gates": gates.as_dict(),
        "split_hash": compute_split_hash(split.as_dict()),
        "split_disjoint": split_disjoint,
        "pass": (
            split_disjoint
            and gates.loss_decrease_pass
            and gates.policy_variance_pass
            and gates.split_leakage_flag == 0
        ),
    }

    write_json_report(out_dir / "smoke_summary.json", payload)
    append_csv_row(
        out_dir / "smoke_metrics.csv",
        {
            "run_id": payload["run_id"],
            "delta_rho": phase_a["delta_rho"],
            "policy_variance": tail.get("policy_variance", 0.0),
            "tailctrl_test_tail_mae": tail.get("test_tail_mae", 0.0),
            "erm_test_tail_mae": phase_b["methods"].get("erm", {}).get("test_tail_mae", 0.0),
            "pass": int(payload["pass"]),
        },
    )

    print(json.dumps(payload, indent=2))
    print(f"json={out_dir / 'smoke_summary.json'}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
