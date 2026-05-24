from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tailctrl.data import DATASET_12_DEFAULT, OPENML_REGISTRY, density_from_train, fit_stratum_labels, four_way_split, load_dataset
from tailctrl.logging_utils import append_csv_row


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

DOMAIN: dict[str, str] = {
    "abalone": "Biology",
    "bike_sharing": "Transport",
    "wine_quality": "Chemistry",
    "cpu_act": "Hardware",
    "energy_efficiency": "Physics",
    "concrete": "Materials",
    "insurance": "Finance",
    "house_16h": "Housing",
    "superconductivity": "Physics",
    "airfoil_self_noise": "Aerospace",
    "kin8nm": "Simulation",
    "diabetes": "Healthcare",
}


def summarize_one(
    dataset_slug: str,
    *,
    seed: int,
    split_ratios: dict[str, float],
    tail_quantile: float,
    dense_quantile: float,
) -> dict:
    ds = load_dataset({"source": "named_tabular", "dataset_name": dataset_slug}, seed)
    split = four_way_split(ds.x.shape[0], split_ratios, seed)
    labels = fit_stratum_labels(
        ds.y[split.meta_train],
        tail_quantile=tail_quantile,
        dense_quantile=dense_quantile,
    )
    density = density_from_train(ds.y[split.meta_train], ds.y)
    test_idx = split.test
    test_density = density[test_idx]
    n_tail_test = int((test_density <= labels.q_low).sum())
    n_dense_test = int((test_density >= labels.q_high).sum())
    openml_ids = OPENML_REGISTRY.get(dataset_slug, [])
    return {
        "dataset_slug": dataset_slug,
        "display_name": DISPLAY_NAMES.get(dataset_slug, dataset_slug),
        "domain": DOMAIN.get(dataset_slug, "Tabular"),
        "openml_id": openml_ids[0] if openml_ids else "sklearn" if dataset_slug == "diabetes" else "",
        "n_total": int(ds.x.shape[0]),
        "n_features": int(ds.x.shape[1]),
        "n_meta_train": int(split.meta_train.size),
        "n_meta_val": int(split.meta_val.size),
        "n_model_val": int(split.model_val.size),
        "n_test": int(split.test.size),
        "n_tail_test": n_tail_test,
        "n_dense_test": n_dense_test,
        "target_min": float(ds.y.min()),
        "target_max": float(ds.y.max()),
        "target_mean": float(ds.y.mean()),
        "target_std": float(ds.y.std()),
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize D1–D12 tabular dataset statistics for manuscript tables.")
    parser.add_argument("--out", type=str, default=str(ROOT / "outputs" / "dataset_stats.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    datasets = list(args.datasets) if args.datasets else list(DATASET_12_DEFAULT)
    split_ratios = {"meta_train": 0.50, "meta_val": 0.15, "model_val": 0.15, "test": 0.20}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    for slug in datasets:
        row = summarize_one(
            slug,
            seed=int(args.seed),
            split_ratios=split_ratios,
            tail_quantile=0.15,
            dense_quantile=0.85,
        )
        append_csv_row(out_path, row)
        print(f"OK {slug}: n={row['n_total']} d={row['n_features']} n_tail_test={row['n_tail_test']}")

    print(f"wrote={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
