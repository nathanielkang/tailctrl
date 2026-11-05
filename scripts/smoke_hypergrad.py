"""Smoke test for the repaired TailCtrl trainer (unrolled hypergradient, equation (rho), m_s gate).

Runs on CPU in about a minute. Exit code 0 only if every check passes.
"""
from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tailctrl.backbone import TabularMLP
from tailctrl.bilevel import TailCtrlTrainer, outer_objective, unrolled_inner_map
from tailctrl.config import load_config
from tailctrl.controller import GeometryController
from tailctrl.data import density_from_train, fit_stratum_labels, four_way_split, load_dataset
from tailctrl.geometry import covariance_singular_values, layer_stratum_geometry, participation_ratio
from tailctrl.losses import inner_geometry_penalty, stratum_masks

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def make_data(cfg: dict, seed: int):
    dataset = load_dataset(cfg["data"], seed)
    split = four_way_split(dataset.x.shape[0], cfg["data"]["split_ratios"], seed)
    labels = fit_stratum_labels(
        dataset.y[split.meta_train],
        tail_quantile=float(cfg["strata"]["tail_quantile"]),
        dense_quantile=float(cfg["strata"]["dense_quantile"]),
    )
    density = density_from_train(dataset.y[split.meta_train], dataset.y)
    x = dataset.x
    x = (x - x[split.meta_train].mean(0)) / (x[split.meta_train].std(0) + 1e-6)
    y = (dataset.y - dataset.y[split.meta_train].mean()) / (dataset.y[split.meta_train].std() + 1e-6)
    return dataset, split, labels, density, x, y


def test_rho_definition() -> None:
    rng = np.random.default_rng(0)
    h = rng.normal(size=(40, 16)) @ rng.normal(size=(16, 16))
    from tailctrl.losses import participation_ratio_torch

    rho_code = float(participation_ratio_torch(torch.as_tensor(h, dtype=torch.float64)))
    rho_sigma = participation_ratio(covariance_singular_values(h))
    c = h - h.mean(0, keepdims=True)
    s_act = np.linalg.svd(c, compute_uv=False)
    rho_old = float(s_act.sum() ** 2 / (s_act**2).sum())
    snap = layer_stratum_geometry(h, epoch=0, layer=0, stratum="tail")
    ok = abs(rho_code - rho_sigma) < 1e-8 * rho_sigma and abs(snap.participation_ratio - rho_sigma) < 1e-8 * rho_sigma
    check(
        "rho = (sum sigma_j(Sigma))^2 / sum sigma_j(Sigma)^2",
        ok and abs(rho_old - rho_sigma) > 1e-3,
        f"torch={rho_code:.6f}, numpy svd(Sigma)={rho_sigma:.6f}, geometry.py={snap.participation_ratio:.6f}; "
        f"old activation-SVD value={rho_old:.6f} (different spectrum)",
    )


def test_gate(width: int, batch: int, layers: int, controlled: list[int], cfg: dict, label: str) -> None:
    torch.manual_seed(0)
    _, split, labels, density, x, y = make_data(cfg, 42)
    rng = np.random.default_rng(1)
    sel = rng.choice(split.meta_train, size=min(batch, len(split.meta_train)), replace=False)
    xb = torch.as_tensor(x[sel], dtype=torch.float32)
    yb = torch.as_tensor(y[sel], dtype=torch.float32)
    db = torch.as_tensor(density[sel], dtype=torch.float32)
    model = TabularMLP(x.shape[1], hidden_dim=width, num_layers=layers, dropout=0.1)
    ctrl = GeometryController(controlled, hidden_dim=64, layer_dims={l: width for l in controlled})
    model.train()
    pred, hidden = model(xb)
    feats = ctrl.build_features(hidden, yb, db, pred, labels.q_low, labels.q_high)
    caps = ctrl.rank_caps(hidden, db, labels.q_low, labels.q_high)
    lam, tau = ctrl(feats, caps)
    stats: dict[str, float] = {}
    pen = inner_geometry_penalty(hidden, db, lam, tau, {"tail": 1.0, "mid": 1.0, "dense": 1.0}, labels.q_low, labels.q_high, stats)
    n_tail = int(stratum_masks(db, labels.q_low, labels.q_high)["tail"].sum())
    tail_hinge = sum(
        float((lam[(l, "tail")] * torch.relu(tau[(l, "tail")] - _rho(hidden[l], db, labels, "tail")) ** 2).detach())
        for l in controlled
    )
    tail_cap = float(min(n_tail - 1, width)) if n_tail >= 2 else 0.0
    tau_ok = all(float(tau[(l, "tail")].detach()) <= tail_cap + 1e-4 for l in controlled)
    ok = float(pen.detach()) > 0 and stats.get("applied_tail", 0) == len(controlled) and tau_ok
    check(
        f"geometry penalty fires ({label})",
        ok,
        f"width={width}, batch={len(sel)}, tail rows={n_tail}, old gate needed >= {2 * width}; "
        f"penalty={float(pen):.4g}, tail pairs applied={int(stats.get('applied_tail', 0))}/{len(controlled)}, "
        f"tail hinge sum={tail_hinge:.4g}",
    )


def _rho(h: torch.Tensor, d: torch.Tensor, labels, stratum: str) -> torch.Tensor:
    from tailctrl.losses import participation_ratio_torch

    return participation_ratio_torch(h[stratum_masks(d, labels.q_low, labels.q_high)[stratum]])


def test_hypergradient(cfg: dict) -> None:
    torch.manual_seed(3)
    _, split, labels, density, x, y = make_data(cfg, 42)
    dt = torch.float64
    rng = np.random.default_rng(7)
    k_steps = 3
    batches = []
    for _ in range(k_steps):
        sel = rng.choice(split.meta_train, size=64, replace=False)
        batches.append(
            (torch.as_tensor(x[sel], dtype=dt), torch.as_tensor(y[sel], dtype=dt), torch.as_tensor(density[sel], dtype=dt))
        )
    mv = split.meta_val
    val = (torch.as_tensor(x[mv], dtype=dt), torch.as_tensor(y[mv], dtype=dt), torch.as_tensor(density[mv], dtype=dt))
    n_tail_val = int(stratum_masks(val[2], labels.q_low, labels.q_high)["tail"].sum())

    controlled = [1, 2]
    model = TabularMLP(x.shape[1], hidden_dim=32, num_layers=3, dropout=0.0).to(dt)
    ctrl = GeometryController(controlled, hidden_dim=16, lambda_max=5.0, tau_scale=0.5, layer_dims={l: 32 for l in controlled}).to(dt)
    theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    buf0 = {n: b.detach().clone() for n, b in model.named_buffers()}
    kw = dict(
        q_low=labels.q_low,
        q_high=labels.q_high,
        controlled_layers=controlled,
        stratum_on={"tail": 1.0, "mid": 1.0, "dense": 1.0},
        alpha=0.5,
        inner_lr=0.05,
    )

    def tail_at_theta_k(create_graph: bool):
        params = {n: t.clone().requires_grad_(True) for n, t in theta0.items()}
        buffers = {n: b.clone() for n, b in buf0.items()}
        params_k, _, _, _ = unrolled_inner_map(model, params, buffers, ctrl, batches, create_graph=create_graph, **kw)
        _, tail = outer_objective(
            model, params_k, buffers, ctrl, val, q_low=labels.q_low, q_high=labels.q_high, gap_layer=2, beta_gap=0.0, beta_wd=0.0
        )
        return tail, params_k, buffers

    phi = [p for p in ctrl.parameters()]

    tail, params_k, buffers = tail_at_theta_k(create_graph=True)
    grads = torch.autograd.grad(tail, phi, allow_unused=True, retain_graph=True)
    g_norm = math.sqrt(sum(float((g**2).sum()) for g in grads if g is not None))
    check(
        "controller gets d TailMAE_val / d phi through theta^(K)(phi)",
        g_norm > 0 and math.isfinite(g_norm) and n_tail_val > 0,
        f"K={k_steps}, val tail rows={n_tail_val}, ||grad_phi TailMAE|| = {g_norm:.3e} "
        "(TailMAE contains no lambda/tau term, so this path is the unroll only)",
    )

    tail_det, _, _ = tail_at_theta_k(create_graph=False)
    if tail_det.requires_grad:
        g_det = torch.autograd.grad(tail_det, phi, allow_unused=True)
        none_all = all(g is None for g in g_det)
    else:
        none_all = True
    check(
        "negative control: detached unroll gives no phi-gradient for TailMAE",
        none_all,
        "create_graph=False (the old optimizer.step / load_state_dict behaviour) leaves TailMAE disconnected from phi",
    )

    # Full outer objective (TailMAE + beta_1 delta_rho^2 + beta_2 ||phi||^2) also backpropagates.
    loss_out, _ = outer_objective(
        model, params_k, buffers, ctrl, val, q_low=labels.q_low, q_high=labels.q_high, gap_layer=2, beta_gap=0.1, beta_wd=0.0
    )
    g_out = torch.autograd.grad(loss_out, phi, allow_unused=True)
    g_out_norm = math.sqrt(sum(float((g**2).sum()) for g in g_out if g is not None))
    check("outer objective (beta_2 = 0) has nonzero phi-gradient", g_out_norm > 0, f"||grad|| = {g_out_norm:.3e}")

    # Central finite difference along a random direction in phi.
    torch.manual_seed(11)
    v = [torch.randn_like(p) for p in phi]
    v_norm = math.sqrt(sum(float((t**2).sum()) for t in v))
    v = [t / v_norm for t in v]
    analytic = sum(float((g * t).sum()) for g, t in zip(grads, v) if g is not None)
    eps = 1e-6
    base = [p.detach().clone() for p in phi]

    def f_at(shift: float) -> float:
        with torch.no_grad():
            for p, b, t in zip(phi, base, v):
                p.copy_(b + shift * t)
        val_t, _, _ = tail_at_theta_k(create_graph=False)
        return float(val_t)

    fd = (f_at(eps) - f_at(-eps)) / (2 * eps)
    with torch.no_grad():
        for p, b in zip(phi, base):
            p.copy_(b)
    rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-12)
    check(
        "hypergradient matches central finite difference (float64)",
        rel < 1e-4 and abs(analytic) > 0,
        f"<grad, v> = {analytic:.6e}, FD = {fd:.6e}, relative error = {rel:.2e}",
    )


def test_trainer(cfg: dict) -> None:
    cfg = copy.deepcopy(cfg)
    cfg["training"]["outer_steps"] = 2
    cfg["training"]["inner_steps_k"] = 3
    dataset, split, labels, _, _, _ = make_data(cfg, 42)
    trainer = TailCtrlTrainer(cfg, "cpu")
    for method in ["tailctrl", "erm", "fixed_floor", "random_policy", "no_outer_objective", "dense_only"]:
        t0 = time.perf_counter()
        res = trainer.train(dataset, split, labels, method, seed=42)
        finite = all(math.isfinite(v) for v in [res.test_tail_mae, res.test_mae, *res.train_loss_trace])
        gate = res.geometry_gate
        detail = (
            f"{time.perf_counter() - t0:.1f}s, outer steps={res.outer_steps_run}, test TailMAE={res.test_tail_mae:.4f}, "
            f"hypergrad norms={[round(v, 4) for v in res.hypergrad_norm_trace]}, "
            f"steps with nonzero hinge={gate.get('frac_steps_hinge_nonzero', 'n/a')}, "
            f"tail pairs applied={gate.get('frac_tail_pairs_applied', 'n/a')}"
        )
        ok = finite
        if method == "tailctrl":
            ok = (
                finite
                and len(res.hypergrad_norm_trace) == 2
                and all(v > 0 for v in res.hypergrad_norm_trace)
                and gate.get("frac_tail_pairs_applied", 0) > 0
            )
        check(f"trainer end-to-end: {method}", ok, detail)


def test_paper_config() -> None:
    cfg = load_config(ROOT / "config" / "tailctrl_default.yaml")
    t = cfg["training"]
    ok = (
        int(cfg["model"]["hidden_dim"]) == 256
        and list(cfg["model"]["controlled_layers"]) == [0, 1, 2, 3]
        and float(t["tail_boost_gamma"]) == 0.0
        and float(t["inv_density_power"]) == 0.0
        and int(t["batch_size"]) == 256
    )
    check(
        "paper config",
        ok,
        f"hidden_dim={cfg['model']['hidden_dim']}, controlled_layers={cfg['model']['controlled_layers']}, "
        f"tail_boost_gamma={t['tail_boost_gamma']}, inv_density_power={t['inv_density_power']}",
    )
    bad = copy.deepcopy(cfg)
    bad["training"]["tail_boost_gamma"] = 0.5
    try:
        TailCtrlTrainer(bad, "cpu")
        raised = False
    except ValueError:
        raised = True
    check("trainer rejects a nonzero density-weighted residual", raised, "tail_boost_gamma=0.5 -> ValueError")

    # One outer iteration at the printed width (256, four blocks, K=5, batch 256).
    cfg["data"]["n_samples"] = 2000
    cfg["training"]["outer_steps"] = 1
    dataset, split, labels, _, _, _ = make_data(cfg, 42)
    t0 = time.perf_counter()
    res = TailCtrlTrainer(cfg, "cpu").train(dataset, split, labels, "tailctrl", seed=42)
    gate = res.geometry_gate
    ok = (
        math.isfinite(res.test_tail_mae)
        and res.hypergrad_norm_trace
        and res.hypergrad_norm_trace[0] > 0
        and math.isfinite(res.hypergrad_norm_trace[0])
        and gate.get("frac_tail_pairs_applied", 0) == 1.0
    )
    check(
        "paper width: one outer step (K=5, width 256, batch 256)",
        bool(ok),
        f"{time.perf_counter() - t0:.1f}s, inner loss trace={[round(v, 2) for v in res.train_loss_trace]}, "
        f"hypergrad norm={res.hypergrad_norm_trace}, gate={gate}",
    )


def main() -> int:
    smoke_cfg = load_config(ROOT / "config" / "smoke.yaml")
    test_rho_definition()
    test_gate(64, 64, 3, [1, 2], smoke_cfg, "smoke width 64, batch 64")
    paper_cfg = load_config(ROOT / "config" / "tailctrl_default.yaml")
    test_gate(256, 256, 4, [0, 1, 2, 3], paper_cfg, "paper width 256, batch 256")
    test_hypergradient(smoke_cfg)
    test_trainer(smoke_cfg)
    test_paper_config()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\nSUMMARY: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
