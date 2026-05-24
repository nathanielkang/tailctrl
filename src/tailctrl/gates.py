from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GateReport:
    split_disjoint: bool
    split_leakage_flag: int
    policy_variance: float
    policy_variance_pass: bool
    budget_ratio: float
    budget_ratio_pass: bool
    loss_decrease_pass: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_disjoint": self.split_disjoint,
            "split_leakage_flag": self.split_leakage_flag,
            "policy_variance": self.policy_variance,
            "policy_variance_pass": self.policy_variance_pass,
            "budget_ratio": self.budget_ratio,
            "budget_ratio_pass": self.budget_ratio_pass,
            "loss_decrease_pass": self.loss_decrease_pass,
        }


def evaluate_hard_gates(
    *,
    split_disjoint: bool,
    policy_variance: float,
    policy_variance_eps: float,
    budget_ratio: float,
    budget_ratio_low: float,
    budget_ratio_high: float,
    first_loss: float,
    last_loss: float,
    min_loss_drop: float,
) -> GateReport:
    loss_drop = first_loss - last_loss
    return GateReport(
        split_disjoint=split_disjoint,
        split_leakage_flag=0 if split_disjoint else 1,
        policy_variance=policy_variance,
        policy_variance_pass=policy_variance > policy_variance_eps,
        budget_ratio=budget_ratio,
        budget_ratio_pass=budget_ratio_low <= budget_ratio <= budget_ratio_high,
        loss_decrease_pass=loss_drop >= min_loss_drop,
    )
