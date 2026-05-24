from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tailctrl.backbone import TabularMLP, mse_loss
from tailctrl.controller import FixedFloorController, GeometryController, RandomPolicyController
from tailctrl.data import StratumLabels, TabularDataset, density_from_train
from tailctrl.losses import inner_loss, outer_loss, stratum_masks
from tailctrl.types import FourWaySplit


@dataclass
class TrainingBudget:
    outer_steps: int = 30
    inner_steps_k: int = 5
    wall_clock_sec: float = 0.0
    forward_passes: int = 0


@dataclass
class TailCtrlTrainResult:
    method: str
    train_loss_trace: list[float] = field(default_factory=list)
    val_tail_mae: float = 0.0
    test_tail_mae: float = 0.0
    test_dense_mae: float = 0.0
    test_mae: float = 0.0
    policy_variance: float = 0.0
    policy_trace: list[float] = field(default_factory=list)
    budget: TrainingBudget = field(default_factory=TrainingBudget)


def _to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _sample_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    idx: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(idx) <= batch_size:
        sel = idx
    else:
        sel = rng.choice(idx, size=batch_size, replace=False)
    return x[sel], y[sel]


def _sample_indices(
    idx: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(idx) <= batch_size:
        return idx
    return rng.choice(idx, size=batch_size, replace=False)


class TailCtrlTrainer:
    def __init__(self, cfg: dict[str, Any], device: str) -> None:
        self.cfg = cfg
        self.device = device
        train_cfg = cfg["training"]
        self.inner_k = int(train_cfg["inner_steps_k"])
        self.inner_lr = float(train_cfg["inner_lr"])
        self.outer_lr = float(train_cfg["outer_lr"])
        self.batch_size = int(train_cfg["batch_size"])
        self.alpha = float(train_cfg["alpha_sparsity"])
        self.beta_gap = float(train_cfg["beta_gap_var"])
        self.beta_wd = float(train_cfg["beta_weight_decay"])
        self.tail_boost_gamma = float(train_cfg.get("tail_boost_gamma", 0.0))
        self.inv_density_power = float(train_cfg.get("inv_density_power", 0.0))
        self.inv_density_clip = float(train_cfg.get("inv_density_clip", 10.0))
        self.masks_weight = {
            "tail": float(train_cfg["m_tail"]),
            "mid": float(train_cfg.get("tail_mask_mid", 0.05)),
            "dense": float(train_cfg.get("tail_mask_dense", 0.05)),
        }
        model_cfg = cfg["model"]
        self.controlled_layers = list(model_cfg["controlled_layers"])
        self.gap_layer = int(cfg.get("phase_a", {}).get("penultimate_layer_index", self.controlled_layers[-1]))

    def build_model(self, n_features: int) -> TabularMLP:
        m = cfg_model = self.cfg["model"]
        return TabularMLP(
            n_features=n_features,
            hidden_dim=int(m["hidden_dim"]),
            num_layers=int(m["num_layers"]),
            dropout=float(m["dropout"]),
        ).to(self.device)

    def build_controller(self, model: TabularMLP, method: str, seed: int) -> nn.Module | FixedFloorController:
        layer_dims = {l: model.hidden_dim for l in self.controlled_layers}
        if method in {"tailctrl", "no_outer_objective", "dense_only"}:
            c_cfg = self.cfg["controller"]
            return GeometryController(
                controlled_layers=self.controlled_layers,
                hidden_dim=int(c_cfg["hidden_dim"]),
                lambda_max=float(c_cfg["lambda_max"]),
                tau_scale=float(c_cfg["tau_scale"]),
                layer_dims=layer_dims,
            ).to(self.device)
        if method == "fixed_floor":
            b_cfg = self.cfg["phase_b"]
            return FixedFloorController(
                self.controlled_layers,
                lam=float(b_cfg["fixed_floor_lambda"]),
                tau=float(b_cfg["fixed_floor_tau"]),
                layer_dims=layer_dims,
            )
        if method == "random_policy":
            return RandomPolicyController(self.controlled_layers, layer_dims, seed=seed).to(self.device)
        raise ValueError(f"Unknown method for controller: {method}")

    def train(
        self,
        dataset: TabularDataset,
        split: FourWaySplit,
        labels: StratumLabels,
        method: str,
        *,
        seed: int,
        outer_steps: int | None = None,
    ) -> TailCtrlTrainResult:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        outer_steps = outer_steps or int(self.cfg["training"]["outer_steps"])

        x = _to_tensor(dataset.x, self.device)
        y_raw = _to_tensor(dataset.y, self.device)
        y_mean = y_raw[split.meta_train].mean()
        y_std = y_raw[split.meta_train].std().clamp_min(1e-6)
        y = (y_raw - y_mean) / y_std
        density_np = density_from_train(
            dataset.y[split.meta_train],
            dataset.y,
            bandwidth=self.cfg["strata"].get("kde_bandwidth"),
        )
        density = _to_tensor(density_np, self.device)
        q_low, q_high = labels.q_low, labels.q_high
        masks_weight = dict(self.masks_weight)
        if method == "dense_only":
            masks_weight = {"tail": 0.0, "mid": 0.0, "dense": 1.0}

        model = self.build_model(dataset.x.shape[1])
        if method == "erm":
            controller: nn.Module | FixedFloorController | None = None
        else:
            controller = self.build_controller(model, method, seed)

        if method in {"erm", "fixed_floor"}:
            return self._train_erm_or_fixed(
                model, controller, x, y, density, split, labels, method, masks_weight, rng, outer_steps
            )
        return self._train_bilevel(
            model, controller, x, y, density, split, labels, method, masks_weight, rng, outer_steps
        )

    def _train_erm_or_fixed(
        self,
        model: TabularMLP,
        controller: nn.Module | FixedFloorController | None,
        x: torch.Tensor,
        y: torch.Tensor,
        density: torch.Tensor,
        split: FourWaySplit,
        labels: StratumLabels,
        method: str,
        masks_weight: dict[str, float],
        rng: np.random.Generator,
        outer_steps: int,
    ) -> TailCtrlTrainResult:
        opt = torch.optim.Adam(model.parameters(), lr=self.inner_lr)
        budget = TrainingBudget(outer_steps=outer_steps, inner_steps_k=1)
        t0 = time.perf_counter()
        trace: list[float] = []
        policy_vals: list[float] = []

        for step in range(outer_steps):
            sel = _sample_indices(split.meta_train, self.batch_size, rng)
            x_b, y_b = x[sel], y[sel]
            d_b = density[sel]
            pred, hidden = model(x_b)
            if method == "fixed_floor":
                dummy = torch.zeros(1, device=self.device)
                lam_map, tau_map = controller.forward(dummy)
            else:
                lam_map, tau_map = {}, {}
            if method == "erm":
                loss = mse_loss(pred, y_b)
            else:
                lam_mid_dense = [
                    lam_map[(l, s)] for l in self.controlled_layers for s in ("mid", "dense") if (l, s) in lam_map
                ]
                loss = inner_loss(
                    pred,
                    y_b,
                    hidden,
                    d_b,
                    lam_map,
                    tau_map,
                    masks_weight,
                    labels.q_low,
                    labels.q_high,
                    self.alpha,
                    lam_mid_dense,
                    tail_boost_gamma=self.tail_boost_gamma,
                    inv_density_power=self.inv_density_power,
                    inv_density_clip=self.inv_density_clip,
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            trace.append(float(loss.detach().cpu()))
            budget.forward_passes += 1
            if lam_map:
                policy_vals.append(float(torch.stack(list(lam_map.values())).var().detach().cpu()))

        budget.wall_clock_sec = time.perf_counter() - t0
        return self._finalize(model, x, y, density, split, labels, method, trace, policy_vals, budget)

    def _train_bilevel(
        self,
        model: TabularMLP,
        controller: GeometryController,
        x: torch.Tensor,
        y: torch.Tensor,
        density: torch.Tensor,
        split: FourWaySplit,
        labels: StratumLabels,
        method: str,
        masks_weight: dict[str, float],
        rng: np.random.Generator,
        outer_steps: int,
    ) -> TailCtrlTrainResult:
        opt_outer = torch.optim.Adam(controller.parameters(), lr=self.outer_lr)
        budget = TrainingBudget(outer_steps=outer_steps, inner_steps_k=self.inner_k)
        t0 = time.perf_counter()
        trace: list[float] = []
        policy_vals: list[float] = []

        for _ in range(outer_steps):
            fast_model = copy.deepcopy(model)
            opt_inner = torch.optim.SGD(fast_model.parameters(), lr=self.inner_lr)

            inner_losses: list[float] = []
            for _k in range(self.inner_k):
                sel = _sample_indices(split.meta_train, self.batch_size, rng)
                x_b, y_b = x[sel], y[sel]
                d_b = density[sel]
                pred, hidden = fast_model(x_b)
                feats = controller.build_features(hidden, y_b, d_b, pred, labels.q_low, labels.q_high)
                lam_map, tau_map = controller(feats)
                lam_mid_dense = [lam_map[(l, s)] for l in self.controlled_layers for s in ("mid", "dense")]
                loss_in = inner_loss(
                    pred,
                    y_b,
                    hidden,
                    d_b,
                    lam_map,
                    tau_map,
                    masks_weight,
                    labels.q_low,
                    labels.q_high,
                    self.alpha,
                    lam_mid_dense,
                    tail_boost_gamma=self.tail_boost_gamma,
                    inv_density_power=self.inv_density_power,
                    inv_density_clip=self.inv_density_clip,
                )
                opt_inner.zero_grad()
                loss_in.backward()
                opt_inner.step()
                inner_losses.append(float(loss_in.detach().cpu()))
                budget.forward_passes += 1

            trace.append(float(np.mean(inner_losses)))

            sel_mv = _sample_indices(split.meta_val, self.batch_size, rng)
            x_mv, y_mv = x[sel_mv], y[sel_mv]
            d_mv = density[sel_mv]
            pred_mv, hidden_mv = fast_model(x_mv)
            feats_mv = controller.build_features(hidden_mv, y_mv, d_mv, pred_mv, labels.q_low, labels.q_high)
            lam_mv, tau_mv = controller(feats_mv)
            policy_vals.append(float(torch.stack(list(lam_mv.values())).var().detach().cpu()))

            tail_mask = stratum_masks(d_mv, labels.q_low, labels.q_high)["tail"]
            if method != "no_outer_objective":
                loss_out = outer_loss(
                    pred_mv,
                    y_mv,
                    tail_mask,
                    d_mv,
                    hidden_mv,
                    controller,
                    self.gap_layer,
                    labels.q_low,
                    labels.q_high,
                    self.beta_gap,
                    self.beta_wd,
                    lam_map=lam_mv,
                    tau_map=tau_mv,
                    masks_weight=masks_weight,
                )
                opt_outer.zero_grad()
                loss_out.backward()
                opt_outer.step()
            budget.forward_passes += 1

            # sync backbone toward inner solution (first-order coupling)
            model.load_state_dict(fast_model.state_dict())

        budget.wall_clock_sec = time.perf_counter() - t0
        return self._finalize(model, x, y, density, split, labels, method, trace, policy_vals, budget)

    def _finalize(
        self,
        model: TabularMLP,
        x: torch.Tensor,
        y: torch.Tensor,
        density: torch.Tensor,
        split: FourWaySplit,
        labels: StratumLabels,
        method: str,
        trace: list[float],
        policy_vals: list[float],
        budget: TrainingBudget,
    ) -> TailCtrlTrainResult:
        model.eval()
        with torch.no_grad():
            pred_test, _ = model(x[split.test])
            pred_mv, _ = model(x[split.model_val])
            masks_test = stratum_masks(density[split.test], labels.q_low, labels.q_high)
            masks_mv = stratum_masks(density[split.model_val], labels.q_low, labels.q_high)
            test_tail = float(torch.mean(torch.abs(pred_test[masks_test["tail"]] - y[split.test][masks_test["tail"]])).cpu())
            val_tail = float(torch.mean(torch.abs(pred_mv[masks_mv["tail"]] - y[split.model_val][masks_mv["tail"]])).cpu())
            dense_mask = masks_test["dense"]
            test_dense = (
                float(torch.mean(torch.abs(pred_test[dense_mask] - y[split.test][dense_mask])).cpu())
                if int(dense_mask.sum()) > 0
                else test_tail
            )
            test_mae = float(torch.mean(torch.abs(pred_test - y[split.test])).cpu())

        pol_var = float(np.var(policy_vals)) if policy_vals else 0.0
        return TailCtrlTrainResult(
            method=method,
            train_loss_trace=trace,
            val_tail_mae=val_tail,
            test_tail_mae=test_tail,
            test_dense_mae=test_dense,
            test_mae=test_mae,
            policy_variance=pol_var,
            policy_trace=policy_vals,
            budget=budget,
        )
