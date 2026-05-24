from __future__ import annotations

import torch

from tailctrl.backbone import mse_loss, tail_mae
from tailctrl.geometry import batch_geometry_by_strata, collapse_gap
from tailctrl.types import STRATA


def hinge_sq(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) ** 2


def stratum_masks(stratum_source: torch.Tensor, q_low: float, q_high: float) -> dict[str, torch.Tensor]:
    return {
        "tail": stratum_source <= q_low,
        "mid": (stratum_source > q_low) & (stratum_source < q_high),
        "dense": stratum_source >= q_high,
    }


def participation_ratio_torch(hidden: torch.Tensor) -> torch.Tensor:
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    if centered.shape[0] < 2:
        return torch.zeros((), device=hidden.device)
    s = torch.linalg.svdvals(centered)
    num = torch.sum(s) ** 2
    den = torch.sum(s**2) + 1e-12
    return num / den


def inner_geometry_penalty(
    hidden_by_layer: dict[int, torch.Tensor],
    stratum_source: torch.Tensor,
    lam_map: dict[tuple[int, str], torch.Tensor],
    tau_map: dict[tuple[int, str], torch.Tensor],
    masks_weight: dict[str, float],
    q_low: float,
    q_high: float,
    reliability_min: int,
) -> torch.Tensor:
    penalty = torch.zeros((), device=stratum_source.device)
    stratum_masks_batch = stratum_masks(stratum_source, q_low, q_high)
    for (layer, stratum), lam in lam_map.items():
        m_s = masks_weight[stratum]
        if m_s <= 0:
            continue
        mask = stratum_masks_batch[stratum]
        n_s = int(mask.sum().item())
        d_l = hidden_by_layer[layer].shape[1]
        if n_s < max(reliability_min, 2 * d_l):
            continue
        h_s = hidden_by_layer[layer][mask]
        rho = participation_ratio_torch(h_s)
        tau = tau_map[(layer, stratum)]
        violation = hinge_sq(tau - rho)
        penalty = penalty + m_s * lam * violation
    return penalty


def inner_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    hidden_by_layer: dict[int, torch.Tensor],
    stratum_source: torch.Tensor,
    lam_map: dict,
    tau_map: dict,
    masks_weight: dict[str, float],
    q_low: float,
    q_high: float,
    alpha_sparsity: float,
    lam_mid_dense: list[torch.Tensor],
    reliability_min: int = 2,
    tail_boost_gamma: float = 0.0,
    inv_density_power: float = 0.0,
    inv_density_clip: float = 10.0,
) -> torch.Tensor:
    base = mse_loss(pred, target)
    geom = inner_geometry_penalty(
        hidden_by_layer,
        stratum_source,
        lam_map,
        tau_map,
        masks_weight,
        q_low,
        q_high,
        reliability_min=reliability_min,
    )
    sparsity = torch.zeros((), device=pred.device)
    if alpha_sparsity > 0 and lam_mid_dense:
        sparsity = alpha_sparsity * sum(torch.abs(l) for l in lam_mid_dense)
    tail_boost = torch.zeros((), device=pred.device)
    if tail_boost_gamma > 0:
        masks = stratum_masks(stratum_source, q_low, q_high)
        residual_sq = (pred - target) ** 2
        inv_d = torch.ones_like(stratum_source)
        if inv_density_power > 0:
            inv_d = (1.0 / (stratum_source.clamp_min(1.0e-12) ** inv_density_power)).clamp_max(inv_density_clip)
        weighted = (
            masks_weight.get("tail", 0.0) * inv_d * masks["tail"].float()
            + masks_weight.get("mid", 0.0) * masks["mid"].float()
            + masks_weight.get("dense", 0.0) * masks["dense"].float()
        )
        if int((weighted > 0).sum().item()) > 0:
            tail_boost = tail_boost_gamma * torch.sum(weighted * residual_sq) / torch.sum(weighted)
    return base + geom + sparsity + tail_boost


def gap_variance_from_batch(
    hidden_by_layer: dict[int, torch.Tensor],
    stratum_source: torch.Tensor,
    layer: int,
    q_low: float,
    q_high: float,
) -> torch.Tensor:
    snaps = batch_geometry_by_strata(hidden_by_layer[layer], stratum_source, q_low, q_high, epoch=0, layer=layer)
    rho = {s.stratum: s.participation_ratio for s in snaps}
    gap = collapse_gap(rho.get("dense", 0.0), rho.get("tail", 0.0))
    return torch.tensor(gap, device=stratum_source.device)


def outer_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tail_mask: torch.Tensor,
    stratum_source: torch.Tensor,
    hidden_by_layer: dict[int, torch.Tensor],
    controller: torch.nn.Module,
    gap_layer: int,
    q_low: float,
    q_high: float,
    beta_gap_var: float,
    beta_weight_decay: float,
    lam_map: dict[tuple[int, str], torch.Tensor] | None = None,
    tau_map: dict[tuple[int, str], torch.Tensor] | None = None,
    masks_weight: dict[str, float] | None = None,
) -> torch.Tensor:
    tail = tail_mae(pred, target, tail_mask)
    gap = gap_variance_from_batch(hidden_by_layer, stratum_source, gap_layer, q_low, q_high)
    policy_term = torch.zeros((), device=pred.device)
    if lam_map and tau_map:
        masks = stratum_masks(stratum_source, q_low, q_high)
        h = hidden_by_layer[gap_layer]
        for s in STRATA:
            mask = masks[s]
            if int(mask.sum().item()) < 2:
                continue
            h_s = h[mask]
            rho_s = participation_ratio_torch(h_s)
            lam_s = lam_map[(gap_layer, s)]
            tau_s = tau_map[(gap_layer, s)]
            w_s = 1.0 if masks_weight is None else float(masks_weight.get(s, 1.0))
            policy_term = policy_term + w_s * lam_s * hinge_sq(tau_s - rho_s)
    reg = torch.zeros((), device=pred.device)
    if beta_weight_decay > 0:
        for p in controller.parameters():
            reg = reg + torch.sum(p**2)
    return tail + beta_gap_var * (gap * gap + policy_term) + beta_weight_decay * reg
