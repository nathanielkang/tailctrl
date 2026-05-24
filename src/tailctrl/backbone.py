from __future__ import annotations

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabularMLP(nn.Module):
    """Multi-layer tabular regressor; returns predictions and per-layer hidden states."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.blocks = nn.ModuleList()
        in_dim = n_features
        for _ in range(num_layers):
            self.blocks.append(MLPBlock(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        h = x
        hidden_by_layer: dict[int, torch.Tensor] = {}
        for layer_idx, block in enumerate(self.blocks):
            h = block(h)
            hidden_by_layer[layer_idx] = h
        pred = self.head(h).squeeze(-1)
        return pred, hidden_by_layer


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def tail_mae(pred: torch.Tensor, target: torch.Tensor, tail_mask: torch.Tensor) -> torch.Tensor:
    if int(tail_mask.sum().item()) == 0:
        return torch.zeros((), device=pred.device)
    return torch.mean(torch.abs(pred[tail_mask] - target[tail_mask]))
