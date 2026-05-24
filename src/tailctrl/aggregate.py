from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tailctrl.stats_tests import friedman_test, nemenyi_posthoc


DISPLAY_NAMES: dict[str, str] = {
    "abalone": "Abalone",
    "bike_sharing": "Bike Sharing",
    "wine_quality": "Wine Quality",
    "cpu_act": "CPU Activity",
    "energy_efficiency": "Energy Efficiency",
    "concrete": "Concrete Compressive Strength",
    "insurance": "Insurance",
    "house_16h": "House 16H",
    "superconductivity": "Superconductivity",
    "airfoil_self_noise": "Airfoil Self Noise",
    "kin8nm": "Kin8nm",
    "diabetes": "Diabetes",
}


def load_phase_b_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_phase_b_runs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for json_path in sorted(root.rglob("phase_b_seed*.json")):
        payload = load_phase_b_json(json_path)
        dataset = str(payload.get("dataset", "")).split("_openml")[0].split("_")[0]
        if dataset.startswith("abalone") or "abalone" in payload.get("dataset", ""):
            dataset = "abalone"
        for slug, display in DISPLAY_NAMES.items():
            if slug in payload.get("dataset", ""):
                dataset = slug
                break
        seed = int(payload["seed"])
        for method, metrics in payload.get("methods", {}).items():
            rows.append(
                {
                    "dataset": dataset,
                    "display_name": DISPLAY_NAMES.get(dataset, dataset),
                    "seed": seed,
                    "method": method,
                    "test_tail_mae": metrics.get("test_tail_mae"),
                    "test_dense_mae": metrics.get("test_dense_mae"),
                    "test_mae": metrics.get("test_mae"),
                    "wall_clock_sec": metrics.get("wall_clock_sec"),
                    "forward_passes": metrics.get("forward_passes"),
                    "policy_variance": metrics.get("policy_variance"),
                    "source_json": str(json_path),
                }
            )
    return pd.DataFrame(rows)


def collect_phase_a_runs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for json_path in sorted(root.rglob("phase_a_seed*.json")):
        payload = load_phase_b_json(json_path)
        dataset = payload.get("dataset", "unknown")
        slug = dataset
        for key in DISPLAY_NAMES:
            if key in dataset:
                slug = key
                break
        rows.append(
            {
                "dataset": slug,
                "display_name": DISPLAY_NAMES.get(slug, slug),
                "seed": int(payload["seed"]),
                "delta_rho": payload.get("delta_rho"),
                "rho_dense": payload.get("rho_dense"),
                "rho_tail": payload.get("rho_tail"),
                "source_json": str(json_path),
            }
        )
    return pd.DataFrame(rows)


def summarize_mean_std(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std", "count": "n_seeds"})
    )
    return agg


def build_tradeoff_summary(phase_b_df: pd.DataFrame) -> pd.DataFrame:
    if phase_b_df.empty:
        return pd.DataFrame()
    focus = phase_b_df[phase_b_df["method"].isin(["erm", "tailctrl"])].copy()
    agg = summarize_mean_std(
        focus,
        "test_tail_mae",
        ["dataset", "display_name", "method"],
    )
    dense = summarize_mean_std(
        focus,
        "test_dense_mae",
        ["dataset", "display_name", "method"],
    )
    overall = summarize_mean_std(
        focus,
        "test_mae",
        ["dataset", "display_name", "method"],
    )
    out = agg.merge(
        dense[["dataset", "method", "test_dense_mae_mean", "test_dense_mae_std"]],
        on=["dataset", "method"],
        how="left",
    ).merge(
        overall[["dataset", "method", "test_mae_mean", "test_mae_std"]],
        on=["dataset", "method"],
        how="left",
    )
    return out.sort_values(["dataset", "method"])


def build_ablation_summary(phase_b_df: pd.DataFrame) -> pd.DataFrame:
    if phase_b_df.empty:
        return pd.DataFrame()
    ablation_methods = [
        "tailctrl",
        "fixed_floor",
        "no_outer_objective",
        "dense_only",
        "random_policy",
    ]
    focus = phase_b_df[phase_b_df["method"].isin(ablation_methods)].copy()
    full = (
        focus.groupby(["dataset", "method"])["test_tail_mae"]
        .mean()
        .reset_index()
        .pivot(index="dataset", columns="method", values="test_tail_mae")
    )
    if "tailctrl" not in full.columns:
        return pd.DataFrame()
    rel_rows: list[dict[str, Any]] = []
    for dataset in full.index:
        base = float(full.loc[dataset, "tailctrl"])
        for method in ablation_methods:
            if method not in full.columns:
                continue
            val = float(full.loc[dataset, method])
            rel_rows.append(
                {
                    "dataset": dataset,
                    "display_name": DISPLAY_NAMES.get(dataset, dataset),
                    "method": method,
                    "test_tail_mae_mean": val,
                    "rel_increase_pct": 100.0 * (val - base) / max(base, 1e-9),
                }
            )
    out = pd.DataFrame(rel_rows)
    macro = (
        out.groupby("method")[["test_tail_mae_mean", "rel_increase_pct"]]
        .mean()
        .reset_index()
        .rename(columns={"test_tail_mae_mean": "macro_tail_mae", "rel_increase_pct": "macro_rel_increase_pct"})
    )
    out.attrs["macro"] = macro
    return out


def build_compute_summary(phase_b_df: pd.DataFrame) -> pd.DataFrame:
    if phase_b_df.empty:
        return pd.DataFrame()
    agg = (
        phase_b_df.groupby(["dataset", "method"])[["wall_clock_sec", "forward_passes"]]
        .mean()
        .reset_index()
    )
    erm = agg[agg["method"] == "erm"][["dataset", "wall_clock_sec", "forward_passes"]].rename(
        columns={"wall_clock_sec": "erm_wall_clock_sec", "forward_passes": "erm_forward_passes"}
    )
    tc = agg[agg["method"] == "tailctrl"].merge(erm, on="dataset", how="left")
    tc["wall_clock_ratio"] = tc["wall_clock_sec"] / tc["erm_wall_clock_sec"].clip(lower=1e-9)
    tc["forward_pass_ratio"] = tc["forward_passes"] / tc["erm_forward_passes"].clip(lower=1e-9)
    return tc


def build_gap_vs_gain(phase_a_df: pd.DataFrame, phase_b_df: pd.DataFrame) -> pd.DataFrame:
    if phase_a_df.empty or phase_b_df.empty:
        return pd.DataFrame()
    a = phase_a_df.groupby("dataset")["delta_rho"].mean().reset_index()
    erm = (
        phase_b_df[phase_b_df["method"] == "erm"]
        .groupby("dataset")["test_tail_mae"]
        .mean()
        .reset_index()
        .rename(columns={"test_tail_mae": "erm_tail_mae"})
    )
    tc = (
        phase_b_df[phase_b_df["method"] == "tailctrl"]
        .groupby("dataset")["test_tail_mae"]
        .mean()
        .reset_index()
        .rename(columns={"test_tail_mae": "tailctrl_tail_mae"})
    )
    merged = a.merge(erm, on="dataset", how="inner").merge(tc, on="dataset", how="inner")
    merged["tail_mae_gain_pct"] = 100.0 * (merged["erm_tail_mae"] - merged["tailctrl_tail_mae"]) / merged[
        "erm_tail_mae"
    ].clip(lower=1e-9)
    merged["display_name"] = merged["dataset"].map(lambda d: DISPLAY_NAMES.get(d, d))
    return merged.sort_values("delta_rho")


def run_friedman_nemenyi(
    phase_b_df: pd.DataFrame,
    methods: list[str],
    *,
    metric: str = "test_tail_mae",
) -> dict[str, Any]:
    focus = phase_b_df[phase_b_df["method"].isin(methods)].copy()
    if focus.empty:
        return {}
    pivot = focus.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")
    pivot = pivot.dropna()
    usable_methods = [m for m in methods if m in pivot.columns]
    if len(usable_methods) < 2 or pivot.shape[0] < 3:
        return {"error": "insufficient data for Friedman", "n_datasets": int(pivot.shape[0])}
    scores = pivot[usable_methods].to_numpy(dtype=np.float64)
    fr = friedman_test(scores, usable_methods, lower_is_better=True)
    nm = nemenyi_posthoc(scores, usable_methods, lower_is_better=True)
    return {
        "metric": metric,
        "friedman_statistic": fr.statistic,
        "friedman_p_value": fr.p_value,
        "mean_ranks": fr.mean_ranks,
        "nemenyi_cd": nm.critical_difference,
        "nemenyi_significant_pairs": nm.significant_pairs,
    }
