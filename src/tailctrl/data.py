from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes
from sklearn.neighbors import KernelDensity

from tailctrl.types import FourWaySplit, STRATA, StratumLabels, TabularDataset


@dataclass
class SyntheticData:
    x: np.ndarray
    y: np.ndarray


OPENML_REGISTRY: dict[str, list[int]] = {
    "abalone": [183],
    "bike_sharing": [42712, 44063],
    "wine_quality": [287],
    "cpu_act": [573, 197],
    "energy_efficiency": [1472, 213],
    "concrete": [44959],
    "insurance": [46289],
    "house_16h": [574],
    "superconductivity": [44964, 43443],
    "airfoil_self_noise": [43919],
    "kin8nm": [189],
}


DATASET_12_DEFAULT = [
    "abalone",
    "bike_sharing",
    "wine_quality",
    "cpu_act",
    "energy_efficiency",
    "concrete",
    "insurance",
    "house_16h",
    "superconductivity",
    "airfoil_self_noise",
    "kin8nm",
    "diabetes",
]


def _to_numeric_target(y: pd.Series, dataset_name: str) -> np.ndarray:
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().any():
        bad = int(y_num.isna().sum())
        raise ValueError(
            f"Dataset '{dataset_name}' has {bad} non-numeric target values. "
            "TailCtrl requires numeric regression targets."
        )
    return y_num.to_numpy(dtype=np.float64)


def _frame_to_numeric_matrix(frame: pd.DataFrame) -> np.ndarray:
    x = frame.copy()
    for col in x.columns:
        if pd.api.types.is_bool_dtype(x[col]):
            x[col] = x[col].astype("int64")
    cat_cols = [c for c in x.columns if pd.api.types.is_object_dtype(x[c]) or pd.api.types.is_categorical_dtype(x[c])]
    if cat_cols:
        x = pd.get_dummies(x, columns=cat_cols, dummy_na=True)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    x = x.fillna(0.0)
    return x.to_numpy(dtype=np.float64)


def _load_openml_by_ids(dataset_name: str, ids: list[int]) -> TabularDataset:
    last_error: Exception | None = None
    for data_id in ids:
        try:
            bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
            frame = bunch.frame
            target_name = bunch.target_names[0] if bunch.target_names else bunch.target.name
            y = _to_numeric_target(frame[target_name], dataset_name)
            x = _frame_to_numeric_matrix(frame.drop(columns=[target_name]))
            return TabularDataset(x=x, y=y, name=f"{dataset_name}_openml_{data_id}")
        except Exception as exc:  # pragma: no cover - network/data dependent
            last_error = exc
    raise RuntimeError(f"Failed to load OpenML dataset '{dataset_name}' via IDs {ids}: {last_error}")


def _load_named_tabular(dataset_name: str) -> TabularDataset:
    name = dataset_name.strip().lower()
    if name == "california_housing":
        try:
            bunch = fetch_california_housing()
            return TabularDataset(x=bunch.data.astype(np.float64), y=bunch.target.astype(np.float64), name=name)
        except Exception:  # pragma: no cover - remote network dependent
            return _load_openml_by_ids("kin8nm", OPENML_REGISTRY["kin8nm"])
    if name == "diabetes":
        bunch = load_diabetes()
        return TabularDataset(x=bunch.data.astype(np.float64), y=bunch.target.astype(np.float64), name=name)
    if name not in OPENML_REGISTRY:
        supported = ", ".join(sorted(list(OPENML_REGISTRY.keys()) + ["california_housing", "diabetes"]))
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. Supported: {supported}")
    return _load_openml_by_ids(name, OPENML_REGISTRY[name])


def make_synthetic_regression(
    n_samples: int,
    n_features: int,
    noise_std: float,
    seed: int,
) -> SyntheticData:
    """Legacy flat-Gaussian targets (kept for backward-compatible smoke)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=(n_features, 1))
    y = x @ true_w + noise_std * rng.normal(size=(n_samples, 1))
    return SyntheticData(x=x.astype(np.float64), y=y.astype(np.float64))


def make_skewed_synthetic_regression(
    n_samples: int,
    n_features: int,
    noise_std: float,
    skew_strength: float,
    seed: int,
) -> TabularDataset:
    """Skewed continuous targets via log-normal transform (DIR-style imbalance)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    w = rng.normal(size=(n_features, 1))
    latent = x @ w + noise_std * rng.normal(size=(n_samples, 1))
    scale = np.exp(skew_strength * (latent - latent.mean()) / (latent.std() + 1e-8))
    y = scale.squeeze(-1)
    return TabularDataset(x=x, y=y.astype(np.float64), name="synthetic_skewed")


def density_from_train(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    bandwidth: float | None = None,
) -> np.ndarray:
    y_train = y_train.reshape(-1)
    y_eval = y_eval.reshape(-1)
    if bandwidth is None:
        kde = gaussian_kde(y_train)
        return kde(y_eval).astype(np.float64)
    model = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    model.fit(y_train.reshape(-1, 1))
    log_d = model.score_samples(y_eval.reshape(-1, 1))
    return np.exp(log_d).astype(np.float64)


def split_indices(
    n_samples: int,
    train_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_samples)
    rng.shuffle(all_idx)
    split = int(n_samples * train_ratio)
    train_idx = np.sort(all_idx[:split])
    val_idx = np.sort(all_idx[split:])
    return train_idx, val_idx


def four_way_split(
    n_samples: int,
    ratios: dict[str, float],
    seed: int,
) -> FourWaySplit:
    required = ("meta_train", "meta_val", "model_val", "test")
    total = sum(float(ratios[k]) for k in required)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)

    cuts = []
    running = 0
    for key in required:
        running += int(n_samples * float(ratios[key]))
        cuts.append(running)
    cuts[-1] = n_samples

    parts = np.split(idx, cuts)
    return FourWaySplit(
        meta_train=np.sort(parts[0]),
        meta_val=np.sort(parts[1]),
        model_val=np.sort(parts[2]),
        test=np.sort(parts[3]),
    )


def fit_stratum_labels(
    y_train: np.ndarray,
    tail_quantile: float = 0.15,
    dense_quantile: float = 0.85,
    bandwidth: float | None = None,
) -> StratumLabels:
    """KDE on train labels only; quantile cuts for tail/mid/dense."""
    y_train = y_train.reshape(-1)
    density = density_from_train(y_train, y_train, bandwidth=bandwidth)

    q_low = float(np.quantile(density, tail_quantile))
    q_high = float(np.quantile(density, dense_quantile))

    dense_idx = np.where(density >= q_high)[0]
    tail_idx = np.where(density <= q_low)[0]
    mid_idx = np.where((density > q_low) & (density < q_high))[0]
    return StratumLabels(
        dense_idx=dense_idx,
        mid_idx=mid_idx,
        tail_idx=tail_idx,
        density=density.astype(np.float64),
        q_low=q_low,
        q_high=q_high,
    )


def assign_stratum_from_density(density: np.ndarray, labels: StratumLabels) -> np.ndarray:
    out = np.empty(density.shape[0], dtype=object)
    out[density <= labels.q_low] = "tail"
    out[density >= labels.q_high] = "dense"
    out[(density > labels.q_low) & (density < labels.q_high)] = "mid"
    return out


def load_dataset(cfg_data: dict, seed: int) -> TabularDataset:
    source = str(cfg_data.get("source", "synthetic_skewed"))
    if source == "synthetic_skewed":
        return make_skewed_synthetic_regression(
            n_samples=int(cfg_data["n_samples"]),
            n_features=int(cfg_data["n_features"]),
            noise_std=float(cfg_data["noise_std"]),
            skew_strength=float(cfg_data.get("skew_strength", 2.0)),
            seed=seed,
        )
    if source in {"named_tabular", "openml_tabular"}:
        dataset_name = str(cfg_data["dataset_name"])
        return _load_named_tabular(dataset_name)
    raise NotImplementedError(
        f"Dataset source '{source}' not wired yet. "
        "Implement OpenML tabular regression loaders in tailctrl.data.load_dataset."
    )


def stratum_indices_in_split(
    split_idx: np.ndarray,
    density: np.ndarray,
    labels: StratumLabels,
) -> dict[str, np.ndarray]:
    d_split = density[split_idx]
    masks = labels.mask_for_density(d_split)
    return {s: split_idx[masks[s]] for s in STRATA}
