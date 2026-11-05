from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from tailctrl.losses import STRATUM_MIN_SUPPORT, participation_ratio_torch
from tailctrl.types import STRATA


class GeometryController(nn.Module):
    """
    Maps geometric features to per-(layer, stratum) penalty strength lambda and floor tau.
    g_phi: F -> {lambda_{l,s}, tau_{l,s}}
    """

    def __init__(
        self,
        controlled_layers: Iterable[int],
        hidden_dim: int = 64,
        lambda_max: float = 10.0,
        tau_scale: float = 1.0,
        layer_dims: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.controlled_layers = list(controlled_layers)
        self.strata = list(STRATA)
        self.lambda_max = float(lambda_max)
        self.tau_scale = float(tau_scale)
        self.layer_dims = layer_dims or {l: 1 for l in self.controlled_layers}

        # input: for each (layer, stratum): rho, stratum_loss (2 * L * S)
        in_dim = len(self.controlled_layers) * len(self.strata) * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        out_dim = len(self.controlled_layers) * len(self.strata) * 2
        self.out = nn.Linear(hidden_dim, out_dim)

    def build_features(
        self,
        hidden_by_layer: dict[int, torch.Tensor],
        target_batch: torch.Tensor,
        stratum_source: torch.Tensor,
        pred: torch.Tensor,
        q_low: float,
        q_high: float,
    ) -> torch.Tensor:
        """F(t): per (layer, stratum), rho_{l,s} from equation (rho) and the stratum MSE.

        Features are measurements at the current predictor iterate theta_t and carry no graph.
        """
        feats: list[torch.Tensor] = []
        source = stratum_source.detach()
        with torch.no_grad():
            for layer in self.controlled_layers:
                h = hidden_by_layer[layer].detach()
                for stratum in self.strata:
                    if stratum == "tail":
                        mask = source <= q_low
                    elif stratum == "dense":
                        mask = source >= q_high
                    else:
                        mask = (source > q_low) & (source < q_high)
                    if int(mask.sum().item()) < STRATUM_MIN_SUPPORT:
                        rho = torch.zeros((), device=h.device, dtype=h.dtype)
                        s_loss = torch.zeros((), device=h.device, dtype=h.dtype)
                    else:
                        rho = participation_ratio_torch(h[mask])
                        s_loss = torch.mean((pred.detach()[mask] - target_batch[mask]) ** 2)
                    feats.extend([rho, s_loss])
        return torch.stack(feats)

    def rank_caps(
        self,
        hidden_by_layer: dict[int, torch.Tensor],
        stratum_source: torch.Tensor,
        q_low: float,
        q_high: float,
    ) -> torch.Tensor:
        """Largest participation ratio a stratum covariance on this batch can attain: min(n_s-1, d_l)."""
        source = stratum_source.detach()
        caps: list[float] = []
        for layer in self.controlled_layers:
            h = hidden_by_layer[layer]
            d_l = int(h.shape[1])
            for stratum in self.strata:
                if stratum == "tail":
                    mask = source <= q_low
                elif stratum == "dense":
                    mask = source >= q_high
                else:
                    mask = (source > q_low) & (source < q_high)
                n_s = int(mask.sum().item())
                caps.append(0.0 if n_s < STRATUM_MIN_SUPPORT else float(min(n_s - 1, d_l)))
        return torch.tensor(caps, device=source.device, dtype=hidden_by_layer[self.controlled_layers[0]].dtype)

    def forward(
        self,
        features: torch.Tensor,
        rank_cap: torch.Tensor,
    ) -> tuple[dict[tuple[int, str], torch.Tensor], dict[tuple[int, str], torch.Tensor]]:
        h = self.mlp(features)
        raw = self.out(h)
        n_ls = len(self.controlled_layers) * len(self.strata)
        lam_raw = raw[:n_ls]
        tau_raw = raw[n_ls:]
        lam_map: dict[tuple[int, str], torch.Tensor] = {}
        tau_map: dict[tuple[int, str], torch.Tensor] = {}
        idx = 0
        for layer in self.controlled_layers:
            for _stratum in self.strata:
                lam = torch.nn.functional.softplus(lam_raw[idx]).clamp(max=self.lambda_max)
                tau = torch.sigmoid(tau_raw[idx]) * self.tau_scale * rank_cap[idx]
                lam_map[(layer, _stratum)] = lam
                tau_map[(layer, _stratum)] = tau
                idx += 1
        return lam_map, tau_map


class FixedFloorController:
    """Non-learned baseline: constant lambda/tau (degeneracy control comparison)."""

    def __init__(self, controlled_layers: Iterable[int], lam: float, tau: float, layer_dims: dict[int, int]) -> None:
        self.controlled_layers = list(controlled_layers)
        self.strata = list(STRATA)
        self.lam = float(lam)
        self.tau = float(tau)
        self.layer_dims = layer_dims

    def rank_caps(self, hidden_by_layer, stratum_source, q_low: float, q_high: float) -> torch.Tensor:
        probe = GeometryController(self.controlled_layers, layer_dims=self.layer_dims)
        return probe.rank_caps(hidden_by_layer, stratum_source, q_low, q_high)

    def forward(self, features: torch.Tensor, rank_cap: torch.Tensor) -> tuple[dict, dict]:
        device = features.device
        lam_map, tau_map = {}, {}
        idx = 0
        for layer in self.controlled_layers:
            for stratum in self.strata:
                lam_map[(layer, stratum)] = torch.tensor(self.lam, device=device)
                tau_map[(layer, stratum)] = self.tau * rank_cap[idx]
                idx += 1
        return lam_map, tau_map


class RandomPolicyController(nn.Module):
    """Frozen random controller initialization (novelty ablation)."""

    def __init__(self, controlled_layers: Iterable[int], layer_dims: dict[int, int], seed: int = 0) -> None:
        super().__init__()
        self.inner = GeometryController(controlled_layers, hidden_dim=32, layer_dims=layer_dims)
        torch.manual_seed(seed)
        for param in self.inner.parameters():
            nn.init.normal_(param, std=0.01)
        for param in self.inner.parameters():
            param.requires_grad_(False)

    def build_features(self, *args, **kwargs) -> torch.Tensor:
        return self.inner.build_features(*args, **kwargs)

    def rank_caps(self, *args, **kwargs) -> torch.Tensor:
        return self.inner.rank_caps(*args, **kwargs)

    def forward(self, features: torch.Tensor, rank_cap: torch.Tensor) -> tuple[dict, dict]:
        return self.inner(features, rank_cap)
