from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STRATA = ("dense", "mid", "tail")


@dataclass
class TabularDataset:
    x: np.ndarray
    y: np.ndarray
    name: str = "dataset"


@dataclass
class FourWaySplit:
    meta_train: np.ndarray
    meta_val: np.ndarray
    model_val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "meta_train": self.meta_train.tolist(),
            "meta_val": self.meta_val.tolist(),
            "model_val": self.model_val.tolist(),
            "test": self.test.tolist(),
        }

    def verify_disjoint(self) -> bool:
        sets = [
            set(self.meta_train.tolist()),
            set(self.meta_val.tolist()),
            set(self.model_val.tolist()),
            set(self.test.tolist()),
        ]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if sets[i].intersection(sets[j]):
                    return False
        return True


@dataclass
class StratumLabels:
    dense_idx: np.ndarray
    mid_idx: np.ndarray
    tail_idx: np.ndarray
    density: np.ndarray
    q_low: float
    q_high: float

    def label_from_density(self, density_value: float) -> str:
        if density_value <= self.q_low:
            return "tail"
        if density_value >= self.q_high:
            return "dense"
        return "mid"

    def mask_for_density(self, density_batch: np.ndarray) -> dict[str, np.ndarray]:
        """Strata are defined by KDE density quantiles (train labels only)."""
        return {
            "dense": density_batch >= self.q_high,
            "mid": (density_batch > self.q_low) & (density_batch < self.q_high),
            "tail": density_batch <= self.q_low,
        }


@dataclass
class GeometrySnapshot:
    epoch: int
    layer: int
    stratum: str
    participation_ratio: float
    effective_rank: float
    stable_rank: float
    nrc1: float
    batch_size: int


@dataclass
class RunArtifacts:
    run_id: str
    method: str
    phase: str
    seed: int
    metrics: dict[str, float] = field(default_factory=dict)
    logs: list[dict[str, float | int | str]] = field(default_factory=list)
