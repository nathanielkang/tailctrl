from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class FriedmanResult:
    statistic: float
    p_value: float
    n_datasets: int
    n_methods: int
    mean_ranks: dict[str, float]


@dataclass
class NemenyiResult:
    critical_difference: float
    alpha: float
    mean_ranks: dict[str, float]
    significant_pairs: list[tuple[str, str, float]]


def rank_matrix(scores: np.ndarray, *, lower_is_better: bool = True) -> np.ndarray:
    """Return rank matrix [n_datasets, n_methods] with average ranks for ties."""
    n_d, n_m = scores.shape
    ranks = np.zeros_like(scores, dtype=np.float64)
    for i in range(n_d):
        row = scores[i]
        order = np.argsort(row if lower_is_better else -row, kind="mergesort")
        sorted_vals = row[order]
        rank_vals = np.empty(n_m, dtype=np.float64)
        j = 0
        while j < n_m:
            k = j
            while k + 1 < n_m and sorted_vals[k + 1] == sorted_vals[j]:
                k += 1
            avg_rank = 0.5 * (j + k) + 1.0
            for t in range(j, k + 1):
                rank_vals[order[t]] = avg_rank
            j = k + 1
        ranks[i] = rank_vals
    return ranks


def friedman_test(
    scores: np.ndarray,
    method_names: list[str],
    *,
    lower_is_better: bool = True,
) -> FriedmanResult:
    if scores.ndim != 2:
        raise ValueError("scores must be 2D [n_datasets, n_methods]")
    ranks = rank_matrix(scores, lower_is_better=lower_is_better)
    stat, p = stats.friedmanchisquare(*[ranks[:, j] for j in range(ranks.shape[1])])
    mean_ranks = {method_names[j]: float(ranks[:, j].mean()) for j in range(len(method_names))}
    return FriedmanResult(
        statistic=float(stat),
        p_value=float(p),
        n_datasets=int(scores.shape[0]),
        n_methods=int(scores.shape[1]),
        mean_ranks=mean_ranks,
    )


def nemenyi_critical_difference(n_datasets: int, n_methods: int, *, alpha: float = 0.05) -> float:
    """Studentized range approximation (Demšar 2006 Table 5, q_alpha values)."""
    q_table = {
        (0.05, 2): 1.960,
        (0.05, 3): 2.343,
        (0.05, 4): 2.569,
        (0.05, 5): 2.728,
        (0.05, 6): 2.850,
        (0.05, 7): 2.949,
        (0.05, 8): 3.031,
        (0.05, 9): 3.102,
        (0.05, 10): 3.164,
        (0.05, 11): 3.219,
        (0.05, 12): 3.268,
    }
    key = (alpha, n_methods)
    if key not in q_table:
        q_alpha = 3.164 if n_methods <= 10 else 3.268
    else:
        q_alpha = q_table[key]
    return float(q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets)))


def nemenyi_posthoc(
    scores: np.ndarray,
    method_names: list[str],
    *,
    lower_is_better: bool = True,
    alpha: float = 0.05,
) -> NemenyiResult:
    ranks = rank_matrix(scores, lower_is_better=lower_is_better)
    mean_ranks = {method_names[j]: float(ranks[:, j].mean()) for j in range(len(method_names))}
    cd = nemenyi_critical_difference(scores.shape[0], scores.shape[1], alpha=alpha)
    sig: list[tuple[str, str, float]] = []
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            diff = abs(mean_ranks[method_names[i]] - mean_ranks[method_names[j]])
            if diff > cd:
                sig.append((method_names[i], method_names[j], diff))
    return NemenyiResult(
        critical_difference=cd,
        alpha=alpha,
        mean_ranks=mean_ranks,
        significant_pairs=sig,
    )
