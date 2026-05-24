from __future__ import annotations

import numpy as np
import torch

from tailctrl.types import STRATA, GeometrySnapshot


def _svd_singular_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    return s


def participation_ratio(singular_values: np.ndarray) -> float:
    s = singular_values
    if s.size == 0 or np.allclose(s, 0):
        return 0.0
    num = float(np.sum(s) ** 2)
    den = float(np.sum(s**2) + 1e-12)
    return num / den


def effective_rank(singular_values: np.ndarray) -> float:
    s = singular_values
    if s.size == 0:
        return 0.0
    p = s / (s.sum() + 1e-12)
    p = p[p > 1e-12]
    if p.size == 0:
        return 0.0
    return float(np.exp(-np.sum(p * np.log(p))))


def stable_rank(singular_values: np.ndarray) -> float:
    s = singular_values
    if s.size == 0:
        return 0.0
    return float((np.sum(s) ** 2) / (np.max(s) ** 2 + 1e-12))


def nrc1_noise_suppression(singular_values: np.ndarray, target_dim: int = 1) -> float:
    """Fraction of covariance energy outside top-t signal subspace (Deep NRC style)."""
    s = singular_values
    if s.size == 0:
        return 0.0
    energy = s**2
    total = float(energy.sum() + 1e-12)
    t = min(int(target_dim), energy.size)
    signal = float(energy[:t].sum())
    return 1.0 - signal / total


def layer_stratum_geometry(
    activations: np.ndarray,
    *,
    epoch: int,
    layer: int,
    stratum: str,
    target_dim: int = 1,
) -> GeometrySnapshot:
    s = _svd_singular_values(activations)
    return GeometrySnapshot(
        epoch=epoch,
        layer=layer,
        stratum=stratum,
        participation_ratio=participation_ratio(s),
        effective_rank=effective_rank(s),
        stable_rank=stable_rank(s),
        nrc1=nrc1_noise_suppression(s, target_dim=target_dim),
        batch_size=int(activations.shape[0]),
    )


def collapse_gap(
    rho_dense: float,
    rho_tail: float,
) -> float:
    return float(rho_dense - rho_tail)


def batch_geometry_by_strata(
    hidden: torch.Tensor,
    stratum_source: torch.Tensor,
    q_low: float,
    q_high: float,
    *,
    epoch: int,
    layer: int,
    target_dim: int = 1,
) -> list[GeometrySnapshot]:
    """hidden: [B, d]; stratum_source: per-sample KDE density (not raw y)."""
    src_np = stratum_source.detach().cpu().numpy().reshape(-1)
    h_np = hidden.detach().cpu().numpy()
    masks = {
        "tail": src_np <= q_low,
        "dense": src_np >= q_high,
        "mid": (src_np > q_low) & (src_np < q_high),
    }
    out: list[GeometrySnapshot] = []
    for stratum in STRATA:
        mask = masks[stratum]
        if mask.sum() < 2:
            out.append(
                GeometrySnapshot(
                    epoch=epoch,
                    layer=layer,
                    stratum=stratum,
                    participation_ratio=0.0,
                    effective_rank=0.0,
                    stable_rank=0.0,
                    nrc1=0.0,
                    batch_size=int(mask.sum()),
                )
            )
            continue
        out.append(
            layer_stratum_geometry(
                h_np[mask],
                epoch=epoch,
                layer=layer,
                stratum=stratum,
                target_dim=target_dim,
            )
        )
    return out
