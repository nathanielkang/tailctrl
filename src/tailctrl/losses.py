from __future__ import annotations

import torch

from tailctrl.backbone import mse_loss, tail_mae

# m_s in equation (inner): a stratum has batch support when it holds at least two rows,
# the minimum for a sample covariance.
STRATUM_MIN_SUPPORT = 2


def hinge_sq(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) ** 2


def stratum_masks(stratum_source: torch.Tensor, q_low: float, q_high: float) -> dict[str, torch.Tensor]:
    return {
        "tail": stratum_source <= q_low,
        "mid": (stratum_source > q_low) & (stratum_source < q_high),
        "dense": stratum_source >= q_high,
    }


def stratum_covariance(h_s: torch.Tensor) -> torch.Tensor:
    """Sample covariance Sigma_{l,s} of the rows of h_s (n_s x d_l), shape d_l x d_l."""
    n = h_s.shape[0]
    centered = h_s - h_s.mean(dim=0, keepdim=True)
    return centered.transpose(0, 1) @ centered / float(n - 1)


def participation_ratio_torch(h_s: torch.Tensor) -> torch.Tensor:
    """Equation (rho): (sum_j sigma_j)^2 / sum_j sigma_j^2 with sigma_j the singular values of Sigma_{l,s}.

    Sigma is symmetric positive semidefinite, so its singular values are its eigenvalues and
    sum_j sigma_j = tr(Sigma), sum_j sigma_j^2 = ||Sigma||_F^2. The closed form is exact and
    twice differentiable, which the unrolled hypergradient needs (an eigen-solver's second
    derivative is singular on the repeated zero eigenvalues of a rank-deficient Sigma).
    """
    if h_s.shape[0] < STRATUM_MIN_SUPPORT:
        return torch.zeros((), device=h_s.device, dtype=h_s.dtype)
    sigma = stratum_covariance(h_s)
    trace = torch.diagonal(sigma).sum()
    frob_sq = torch.sum(sigma * sigma)
    # Identical rows give Sigma = 0 and a 0/0 ratio. The manuscript sets rho := 1
    # there and drops that pair's hinge gradient. A constant 1 has no theta gradient.
    if float(frob_sq.detach()) == 0.0:
        return torch.ones((), device=h_s.device, dtype=h_s.dtype)
    return trace**2 / frob_sq


def collapse_gap_torch(
    hidden: torch.Tensor,
    stratum_source: torch.Tensor,
    q_low: float,
    q_high: float,
) -> torch.Tensor | None:
    """delta_rho = rho_dense - rho_tail at one layer; None when either stratum lacks support."""
    masks = stratum_masks(stratum_source, q_low, q_high)
    if int(masks["dense"].sum()) < STRATUM_MIN_SUPPORT or int(masks["tail"].sum()) < STRATUM_MIN_SUPPORT:
        return None
    return participation_ratio_torch(hidden[masks["dense"]]) - participation_ratio_torch(hidden[masks["tail"]])


def inner_geometry_penalty(
    hidden_by_layer: dict[int, torch.Tensor],
    stratum_source: torch.Tensor,
    lam_map: dict[tuple[int, str], torch.Tensor],
    tau_map: dict[tuple[int, str], torch.Tensor],
    stratum_on: dict[str, float],
    q_low: float,
    q_high: float,
    stats: dict[str, float] | None = None,
) -> torch.Tensor:
    """sum_{l,s} m_s lambda_{l,s} [tau_{l,s} - rho_{l,s}]_+^2.

    m_s = 1 when stratum s has at least STRATUM_MIN_SUPPORT rows in the batch. stratum_on is a
    0/1 selector used only by the dense-only ablation; the full method keeps every stratum on.
    """
    penalty = torch.zeros((), device=stratum_source.device)
    masks = stratum_masks(stratum_source, q_low, q_high)
    for (layer, stratum), lam in lam_map.items():
        if stratum_on.get(stratum, 1.0) <= 0:
            continue
        mask = masks[stratum]
        if int(mask.sum().item()) < STRATUM_MIN_SUPPORT:
            continue
        rho = participation_ratio_torch(hidden_by_layer[layer][mask])
        violation = hinge_sq(tau_map[(layer, stratum)] - rho)
        penalty = penalty + lam * violation
        if stats is not None:
            stats[f"applied_{stratum}"] = stats.get(f"applied_{stratum}", 0.0) + 1.0
            if float((lam * violation).detach()) > 0.0:
                stats[f"nonzero_{stratum}"] = stats.get(f"nonzero_{stratum}", 0.0) + 1.0
    if stats is not None:
        stats["pairs_total"] = stats.get("pairs_total", 0.0) + float(len(lam_map))
    return penalty


def inner_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    hidden_by_layer: dict[int, torch.Tensor],
    stratum_source: torch.Tensor,
    lam_map: dict,
    tau_map: dict,
    stratum_on: dict[str, float],
    q_low: float,
    q_high: float,
    stats: dict[str, float] | None = None,
) -> torch.Tensor:
    """Equation (inner): MSE + spectral hinge. The sparsity charge is in the outer loss."""
    base = mse_loss(pred, target)
    geom = inner_geometry_penalty(
        hidden_by_layer,
        stratum_source,
        lam_map,
        tau_map,
        stratum_on,
        q_low,
        q_high,
        stats=stats,
    )
    return base + geom


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
    alpha_sparsity: float = 0.0,
    lam_map: dict | None = None,
) -> torch.Tensor:
    """Equation (outer): TailMAE + beta_1 delta_rho^2 + alpha * sum of mid/dense penalties + beta_2 ||phi||^2.

    lambda and tau are the values emitted before the inner loop and held fixed, so this charge
    does not enter grad_theta. It enters grad_phi through those emitted penalties.
    """
    total = tail_mae(pred, target, tail_mask)
    if alpha_sparsity > 0 and lam_map:
        charged = [lam for (layer, stratum), lam in lam_map.items() if stratum in ("mid", "dense")]
        if charged:
            total = total + alpha_sparsity * sum(charged)
    gap = collapse_gap_torch(hidden_by_layer[gap_layer], stratum_source, q_low, q_high)
    if gap is not None and beta_gap_var > 0:
        total = total + beta_gap_var * gap * gap
    if beta_weight_decay > 0:
        reg = torch.zeros((), device=pred.device)
        for p in controller.parameters():
            reg = reg + torch.sum(p**2)
        total = total + beta_weight_decay * reg
    return total
