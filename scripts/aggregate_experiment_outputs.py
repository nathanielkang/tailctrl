from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tailctrl.aggregate import (
    build_ablation_summary,
    build_compute_summary,
    build_gap_vs_gain,
    build_tradeoff_summary,
    collect_phase_a_runs,
    collect_phase_b_runs,
    run_friedman_nemenyi,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Phase A/B JSON outputs into manuscript-ready CSVs.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            str(ROOT / "outputs" / "suite"),
            str(ROOT / "outputs" / "multi_seed_suite"),
            str(ROOT / "outputs" / "phase_a_suite"),
            str(ROOT / "outputs" / "suite_12_vm"),
        ],
        help="Root directories to scan recursively for phase_* JSON files.",
    )
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs" / "aggregated"))
    parser.add_argument(
        "--friedman-methods",
        nargs="+",
        default=["erm", "fixed_floor", "tailctrl", "random_policy", "no_outer_objective", "dense_only"],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_b_frames = []
    phase_a_frames = []
    for root_str in args.inputs:
        root = Path(root_str)
        if not root.exists():
            continue
        phase_b_frames.append(collect_phase_b_runs(root))
        phase_a_frames.append(collect_phase_a_runs(root))

    import pandas as pd

    phase_b_frames = [f for f in phase_b_frames if not f.empty]
    phase_a_frames = [f for f in phase_a_frames if not f.empty]
    phase_b_df = pd.concat(phase_b_frames, ignore_index=True) if phase_b_frames else pd.DataFrame()
    phase_a_df = pd.concat(phase_a_frames, ignore_index=True) if phase_a_frames else pd.DataFrame()

    if not phase_b_df.empty:
        phase_b_df = phase_b_df.drop_duplicates(subset=["dataset", "seed", "method"], keep="last")
        phase_b_df.to_csv(out_dir / "phase_b_all_runs.csv", index=False)

    if not phase_a_df.empty:
        phase_a_df = phase_a_df.drop_duplicates(subset=["dataset", "seed"], keep="last")
        phase_a_df.to_csv(out_dir / "phase_a_all_runs.csv", index=False)

    tradeoff = build_tradeoff_summary(phase_b_df)
    if not tradeoff.empty:
        tradeoff.to_csv(out_dir / "tradeoff_summary.csv", index=False)

    ablation = build_ablation_summary(phase_b_df)
    if not ablation.empty:
        ablation.to_csv(out_dir / "ablation_summary.csv", index=False)
        macro = ablation.attrs.get("macro")
        if macro is not None and not macro.empty:
            macro.to_csv(out_dir / "ablation_macro.csv", index=False)

    compute = build_compute_summary(phase_b_df)
    if not compute.empty:
        compute.to_csv(out_dir / "compute_summary.csv", index=False)

    gap_gain = build_gap_vs_gain(phase_a_df, phase_b_df)
    if not gap_gain.empty:
        gap_gain.to_csv(out_dir / "gap_vs_gain.csv", index=False)
        if gap_gain.shape[0] >= 3:
            from scipy.stats import spearmanr

            rho, p = spearmanr(gap_gain["delta_rho"], gap_gain["tail_mae_gain_pct"])
            stats_payload = {
                "spearman_rho": float(rho),
                "spearman_p_value": float(p),
                "n_datasets": int(gap_gain.shape[0]),
            }
            (out_dir / "gap_vs_gain_stats.json").write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    stats = run_friedman_nemenyi(phase_b_df, args.friedman_methods, metric="test_tail_mae")
    if stats:
        (out_dir / "friedman_nemenyi.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    summary = {
        "phase_b_rows": int(len(phase_b_df)),
        "phase_a_rows": int(len(phase_a_df)),
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "aggregate_done.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
