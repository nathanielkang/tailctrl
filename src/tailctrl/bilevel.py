from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from tailctrl.backbone import TabularMLP, mse_loss, tail_mae
from tailctrl.controller import FixedFloorController, GeometryController, RandomPolicyController
from tailctrl.data import StratumLabels, TabularDataset, density_from_train
from tailctrl.losses import inner_loss, outer_loss, stratum_masks
from tailctrl.types import STRATA, FourWaySplit

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


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
    outer_steps_run: int = 0
    best_outer_step: int = 0
    hypergrad_norm_trace: list[float] = field(default_factory=list)
    geometry_gate: dict[str, float] = field(default_factory=dict)


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


def predictor_forward(
    model: TabularMLP,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    x: torch.Tensor,
    *,
    train: bool,
    dropout: bool = True,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """f_theta(x) evaluated at an explicit parameter dict (keeps the unrolled graph).

    train=True uses batch statistics in BatchNorm (and updates the running buffers passed in);
    dropout=False switches the dropout layers off while BatchNorm stays on batch statistics.
    """
    model.train(train)
    if train and not dropout:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
    try:
        return functional_call(model, (params, buffers), (x,))
    finally:
        model.train(train)


def unrolled_inner_map(
    model: TabularMLP,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    controller: nn.Module,
    batches: list[Batch],
    *,
    q_low: float,
    q_high: float,
    controlled_layers: list[int],
    stratum_on: dict[str, float],
    alpha: float,
    inner_lr: float,
    create_graph: bool,
    stats: dict[str, float] | None = None,
) -> tuple[dict[str, torch.Tensor], list[float], dict, dict]:
    """theta^(K)(phi): K steps theta <- theta - eta_theta grad_theta L_in(theta; phi) (Algorithm 1).

    F(t) and (lambda, tau) = g_phi(F(t)) are formed once, from the first meta-train batch at
    theta_t, and held for all K steps. With create_graph=True every step stays in the autograd
    graph, so a backward pass from a loss on theta^(K) reaches phi through each inner update.
    The sparsity charge alpha is not part of L_in. outer_objective adds it to L_out.
    """
    if not torch.isfinite(torch.tensor(alpha)):
        raise ValueError("alpha must be finite")
    names = list(params.keys())
    x0, y0, d0 = batches[0]
    pred, hidden = predictor_forward(model, params, buffers, x0, train=True)
    feats = controller.build_features(hidden, y0, d0, pred, q_low, q_high)
    lam_map, tau_map = controller(feats, controller.rank_caps(hidden, d0, q_low, q_high))

    losses: list[float] = []
    for k, (x_b, y_b, d_b) in enumerate(batches):
        if k > 0:
            pred, hidden = predictor_forward(model, params, buffers, x_b, train=True)
        step_stats: dict[str, float] = {}
        loss_in = inner_loss(
            pred,
            y_b,
            hidden,
            d_b,
            lam_map,
            tau_map,
            stratum_on,
            q_low,
            q_high,
            stats=step_stats,
        )
        grads = torch.autograd.grad(
            loss_in,
            [params[n] for n in names],
            create_graph=create_graph,
            allow_unused=True,
        )
        new_params: dict[str, torch.Tensor] = {}
        for n, g in zip(names, grads):
            p = params[n] if g is None else params[n] - inner_lr * g
            new_params[n] = p if create_graph else p.detach().requires_grad_(True)
        params = new_params
        losses.append(float(loss_in.detach().cpu()))
        if stats is not None:
            _merge_gate_stats(stats, step_stats)
    return params, losses, lam_map, tau_map


def outer_objective(
    model: TabularMLP,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    controller: nn.Module,
    batch: Batch,
    *,
    q_low: float,
    q_high: float,
    gap_layer: int,
    beta_gap: float,
    beta_wd: float,
    alpha: float = 0.0,
    lam_map: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L_out on a meta-val batch at theta^(K)(phi); returns (L_out, TailMAE term).

    BatchNorm uses the meta-val batch statistics on a throwaway copy of the buffers. Eval-mode
    running statistics depend on phi through the unroll but are updated outside autograd, so
    using them would drop part of d L_out / d phi; batch statistics keep L_out differentiable
    in theta^(K) end to end. Dropout is off. Model selection and test use eval mode.
    """
    x_mv, y_mv, d_mv = batch
    scratch = {n: b.clone() for n, b in buffers.items()}
    pred, hidden = predictor_forward(model, params, scratch, x_mv, train=True, dropout=False)
    tail_mask = stratum_masks(d_mv, q_low, q_high)["tail"]
    loss = outer_loss(
        pred,
        y_mv,
        tail_mask,
        d_mv,
        hidden,
        controller,
        gap_layer,
        q_low,
        q_high,
        beta_gap,
        beta_wd,
        alpha_sparsity=alpha,
        lam_map=lam_map,
    )
    return loss, tail_mae(pred, y_mv, tail_mask)


def _merge_gate_stats(total: dict[str, float], step: dict[str, float]) -> None:
    total["inner_steps"] = total.get("inner_steps", 0.0) + 1.0
    nonzero = sum(v for k, v in step.items() if k.startswith("nonzero_"))
    if nonzero > 0:
        total["steps_with_nonzero_hinge"] = total.get("steps_with_nonzero_hinge", 0.0) + 1.0
    for k, v in step.items():
        total[k] = total.get(k, 0.0) + v


def _summarize_gate_stats(stats: dict[str, float], n_layers: int) -> dict[str, float]:
    steps = stats.get("inner_steps", 0.0)
    if steps <= 0:
        return {}
    out = {
        "inner_steps": steps,
        "frac_steps_hinge_nonzero": stats.get("steps_with_nonzero_hinge", 0.0) / steps,
    }
    for s in STRATA:
        out[f"frac_{s}_pairs_applied"] = stats.get(f"applied_{s}", 0.0) / (steps * n_layers)
        out[f"frac_{s}_pairs_nonzero"] = stats.get(f"nonzero_{s}", 0.0) / (steps * n_layers)
    return out


class _EarlyStopping:
    """Model selection on D_modelval TailMAE: stop after `patience` outer steps without improvement."""

    def __init__(self, patience: int) -> None:
        self.patience = int(patience)
        self.best = float("inf")
        self.best_step = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.bad = 0

    def update(self, model: nn.Module, value: float | None, step: int) -> bool:
        if value is None:
            return False
        if value < self.best:
            self.best = value
            self.best_step = step
            self.best_state = copy.deepcopy(model.state_dict())
            self.bad = 0
            return False
        self.bad += 1
        return self.patience > 0 and self.bad >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class TailCtrlTrainer:
    def __init__(self, cfg: dict[str, Any], device: str) -> None:
        self.cfg = cfg
        self.device = device
        train_cfg = cfg["training"]
        for knob in ("tail_boost_gamma", "inv_density_power"):
            if float(train_cfg.get(knob, 0.0)) != 0.0:
                raise ValueError(
                    f"training.{knob} must be 0: the density-weighted residual is not part of equation (inner)."
                )
        self.inner_k = int(train_cfg["inner_steps_k"])
        self.inner_lr = float(train_cfg["inner_lr"])
        self.outer_lr = float(train_cfg["outer_lr"])
        self.batch_size = int(train_cfg["batch_size"])
        self.alpha = float(train_cfg["alpha_sparsity"])
        self.beta_gap = float(train_cfg["beta_gap_var"])
        self.beta_wd = float(train_cfg["beta_weight_decay"])
        self.predictor_wd = float(train_cfg.get("predictor_weight_decay", 0.0))
        self.patience = int(train_cfg.get("early_stop_patience", 0))
        self.stratum_on = {
            "tail": float(train_cfg.get("m_tail", 1.0)),
            "mid": float(train_cfg.get("tail_mask_mid", 1.0)),
            "dense": float(train_cfg.get("tail_mask_dense", 1.0)),
        }
        for s, v in self.stratum_on.items():
            if v not in (0.0, 1.0):
                raise ValueError(f"m_s for stratum '{s}' must be 0 or 1 (presence flag in equation (inner)); got {v}.")
        model_cfg = cfg["model"]
        self.controlled_layers = list(model_cfg["controlled_layers"])
        self.gap_layer = int(cfg.get("phase_a", {}).get("penultimate_layer_index", self.controlled_layers[-1]))

    def build_model(self, n_features: int) -> TabularMLP:
        m = self.cfg["model"]
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

        x_raw = _to_tensor(dataset.x, self.device)
        x_mean = x_raw[split.meta_train].mean(dim=0, keepdim=True)
        x_std = x_raw[split.meta_train].std(dim=0, keepdim=True).clamp_min(1e-6)
        x = (x_raw - x_mean) / x_std
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
        stratum_on = dict(self.stratum_on)
        if method == "dense_only":
            stratum_on = {"tail": 0.0, "mid": 0.0, "dense": 1.0}

        model = self.build_model(dataset.x.shape[1])
        if method == "erm":
            controller: nn.Module | FixedFloorController | None = None
        else:
            controller = self.build_controller(model, method, seed)

        if method in {"erm", "fixed_floor"}:
            return self._train_erm_or_fixed(
                model, controller, x, y, density, split, labels, method, stratum_on, rng, outer_steps
            )
        return self._train_bilevel(
            model, controller, x, y, density, split, labels, method, stratum_on, rng, outer_steps
        )

    def _model_val_tail_mae(
        self,
        model: TabularMLP,
        x: torch.Tensor,
        y: torch.Tensor,
        density: torch.Tensor,
        split: FourWaySplit,
        labels: StratumLabels,
    ) -> float | None:
        mask = stratum_masks(density[split.model_val], labels.q_low, labels.q_high)["tail"]
        if int(mask.sum()) == 0:
            return None
        was_training = model.training
        model.eval()
        with torch.no_grad():
            pred, _ = model(x[split.model_val])
            value = float(torch.mean(torch.abs(pred[mask] - y[split.model_val][mask])).cpu())
        model.train(was_training)
        return value

    def _train_erm_or_fixed(
        self,
        model: TabularMLP,
        controller: FixedFloorController | None,
        x: torch.Tensor,
        y: torch.Tensor,
        density: torch.Tensor,
        split: FourWaySplit,
        labels: StratumLabels,
        method: str,
        stratum_on: dict[str, float],
        rng: np.random.Generator,
        outer_steps: int,
    ) -> TailCtrlTrainResult:
        opt = torch.optim.Adam(model.parameters(), lr=self.inner_lr, weight_decay=self.predictor_wd)
        budget = TrainingBudget(outer_steps=outer_steps, inner_steps_k=1)
        t0 = time.perf_counter()
        trace: list[float] = []
        policy_vals: list[float] = []
        gate_stats: dict[str, float] = {}
        stopper = _EarlyStopping(self.patience)
        steps_run = 0

        for step in range(outer_steps):
            model.train()
            sel = _sample_indices(split.meta_train, self.batch_size, rng)
            x_b, y_b = x[sel], y[sel]
            d_b = density[sel]
            pred, hidden = model(x_b)
            if method == "erm":
                loss = mse_loss(pred, y_b)
            else:
                lam_map, tau_map = controller.forward(
                    torch.zeros(1, device=self.device),
                    controller.rank_caps(hidden, d_b, labels.q_low, labels.q_high),
                )
                step_stats: dict[str, float] = {}
                loss = inner_loss(
                    pred,
                    y_b,
                    hidden,
                    d_b,
                    lam_map,
                    tau_map,
                    stratum_on,
                    labels.q_low,
                    labels.q_high,
                    stats=step_stats,
                )
                _merge_gate_stats(gate_stats, step_stats)
                policy_vals.append(float(torch.stack(list(lam_map.values())).var().detach().cpu()))
            opt.zero_grad()
            loss.backward()
            opt.step()
            trace.append(float(loss.detach().cpu()))
            budget.forward_passes += 1
            steps_run = step + 1
            if stopper.update(model, self._model_val_tail_mae(model, x, y, density, split, labels), steps_run):
                break

        stopper.restore(model)
        budget.wall_clock_sec = time.perf_counter() - t0
        result = self._finalize(model, x, y, density, split, labels, method, trace, policy_vals, budget)
        result.outer_steps_run = steps_run
        result.best_outer_step = stopper.best_step
        result.geometry_gate = _summarize_gate_stats(gate_stats, len(self.controlled_layers))
        return result

    def _train_bilevel(
        self,
        model: TabularMLP,
        controller: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        density: torch.Tensor,
        split: FourWaySplit,
        labels: StratumLabels,
        method: str,
        stratum_on: dict[str, float],
        rng: np.random.Generator,
        outer_steps: int,
    ) -> TailCtrlTrainResult:
        trainable = [p for p in controller.parameters() if p.requires_grad]
        hypergrad = method != "no_outer_objective" and len(trainable) > 0
        opt_outer = torch.optim.Adam(trainable, lr=self.outer_lr) if hypergrad else None
        budget = TrainingBudget(outer_steps=outer_steps, inner_steps_k=self.inner_k)
        t0 = time.perf_counter()
        trace: list[float] = []
        policy_vals: list[float] = []
        hg_norms: list[float] = []
        gate_stats: dict[str, float] = {}
        stopper = _EarlyStopping(self.patience)
        steps_run = 0
        buffers = dict(model.named_buffers())

        for step in range(outer_steps):
            # theta_t is the warm start; the unroll treats it as a constant.
            params = {n: p.detach().clone().requires_grad_(True) for n, p in model.named_parameters()}
            batches: list[Batch] = []
            for _k in range(self.inner_k):
                sel = _sample_indices(split.meta_train, self.batch_size, rng)
                batches.append((x[sel], y[sel], density[sel]))

            params_k, inner_losses, lam_map, _tau_map = unrolled_inner_map(
                model,
                params,
                buffers,
                controller,
                batches,
                q_low=labels.q_low,
                q_high=labels.q_high,
                controlled_layers=self.controlled_layers,
                stratum_on=stratum_on,
                alpha=self.alpha,
                inner_lr=self.inner_lr,
                create_graph=hypergrad,
                stats=gate_stats,
            )
            budget.forward_passes += self.inner_k
            trace.append(float(np.mean(inner_losses)))
            policy_vals.append(float(torch.stack(list(lam_map.values())).var().detach().cpu()))

            if hypergrad:
                sel_mv = _sample_indices(split.meta_val, self.batch_size, rng)
                loss_out, _tail = outer_objective(
                    model,
                    params_k,
                    buffers,
                    controller,
                    (x[sel_mv], y[sel_mv], density[sel_mv]),
                    q_low=labels.q_low,
                    q_high=labels.q_high,
                    gap_layer=self.gap_layer,
                    beta_gap=self.beta_gap,
                    beta_wd=self.beta_wd,
                    alpha=self.alpha,
                    lam_map=lam_map,
                )
                opt_outer.zero_grad()
                if loss_out.requires_grad:
                    loss_out.backward(inputs=trainable)
                    sq = [torch.sum(p.grad**2) for p in trainable if p.grad is not None]
                    hg_norms.append(float(torch.sqrt(torch.stack(sq).sum()).cpu()) if sq else 0.0)
                    opt_outer.step()
                else:
                    hg_norms.append(0.0)
                budget.forward_passes += 1

            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(params_k[n].detach())
            model.train()
            del params_k, params

            steps_run = step + 1
            if stopper.update(model, self._model_val_tail_mae(model, x, y, density, split, labels), steps_run):
                break

        stopper.restore(model)
        budget.wall_clock_sec = time.perf_counter() - t0
        result = self._finalize(model, x, y, density, split, labels, method, trace, policy_vals, budget)
        result.outer_steps_run = steps_run
        result.best_outer_step = stopper.best_step
        result.hypergrad_norm_trace = hg_norms
        result.geometry_gate = _summarize_gate_stats(gate_stats, len(self.controlled_layers))
        return result

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
