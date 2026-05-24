from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tailctrl.backbone import TabularMLP
from tailctrl.bilevel import _sample_batch, _sample_indices, _to_tensor
from tailctrl.config import get_device
from tailctrl.data import density_from_train, fit_stratum_labels, four_way_split, load_dataset
from tailctrl.geometry import batch_geometry_by_strata, collapse_gap
from tailctrl.logging_utils import append_csv_row, compute_split_hash, write_json_report


def run_phase_a(cfg: dict[str, Any], out_dir: Path, seed: int) -> dict[str, Any]:
    device = get_device(cfg)
    data_cfg = cfg["data"]
    phase_cfg = cfg["phase_a"]

    dataset = load_dataset(data_cfg, seed)
    split = four_way_split(dataset.x.shape[0], data_cfg["split_ratios"], seed)
    if not split.verify_disjoint():
        raise RuntimeError("Splits are not disjoint")

    strata_cfg = cfg["strata"]
    labels = fit_stratum_labels(
        dataset.y[split.meta_train],
        tail_quantile=float(strata_cfg["tail_quantile"]),
        dense_quantile=float(strata_cfg["dense_quantile"]),
        bandwidth=strata_cfg.get("kde_bandwidth"),
    )

    x = _to_tensor(dataset.x, device)
    y_raw = _to_tensor(dataset.y, device)
    y_mean = y_raw[split.meta_train].mean()
    y_std = y_raw[split.meta_train].std().clamp_min(1e-6)
    y = (y_raw - y_mean) / y_std
    density_np = density_from_train(
        dataset.y[split.meta_train],
        dataset.y,
        bandwidth=cfg["strata"].get("kde_bandwidth"),
    )
    density = _to_tensor(density_np, device)
    model = TabularMLP(
        n_features=dataset.x.shape[1],
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["inner_lr"]))
    rng = np.random.default_rng(seed)

    log_epochs = set(int(e) for e in phase_cfg["log_epochs"])
    gap_layer = int(phase_cfg["penultimate_layer_index"])
    geometry_logs: list[dict[str, float | int | str]] = []
    loss_trace: list[float] = []

    epochs = int(phase_cfg["epochs"])
    batch_size = int(cfg["training"]["batch_size"])

    for epoch in range(1, epochs + 1):
        x_b, y_b = _sample_batch(x, y, split.meta_train, batch_size, rng)
        pred, hidden = model(x_b)
        loss = torch.mean((pred - y_b) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_trace.append(float(loss.detach().cpu()))

        if epoch in log_epochs:
            model.eval()
            with torch.no_grad():
                sel_log = _sample_indices(split.meta_train, batch_size, rng)
                x_log, y_log = x[sel_log], y[sel_log]
                d_log = density[sel_log]
                _, hidden_log = model(x_log)
                snaps = batch_geometry_by_strata(
                    hidden_log[gap_layer],
                    d_log,
                    labels.q_low,
                    labels.q_high,
                    epoch=epoch,
                    layer=gap_layer,
                )
                for snap in snaps:
                    geometry_logs.append(
                        {
                            "epoch": snap.epoch,
                            "layer": snap.layer,
                            "stratum": snap.stratum,
                            "participation_ratio": snap.participation_ratio,
                            "effective_rank": snap.effective_rank,
                            "stable_rank": snap.stable_rank,
                            "nrc1": snap.nrc1,
                            "batch_size": snap.batch_size,
                        }
                    )
            model.train()

    # primary gap at phenomenon epoch
    t_star = int(phase_cfg["phenomenon_epoch"])
    epoch_snaps = [g for g in geometry_logs if g["epoch"] == t_star and g["layer"] == gap_layer]
    rho_dense = next((g["participation_ratio"] for g in epoch_snaps if g["stratum"] == "dense"), 0.0)
    rho_tail = next((g["participation_ratio"] for g in epoch_snaps if g["stratum"] == "tail"), 0.0)
    delta_rho = collapse_gap(rho_dense, rho_tail)

    split_hash = compute_split_hash(split.as_dict())

    payload = {
        "phase": "A",
        "method": "ERM",
        "seed": seed,
        "dataset": dataset.name,
        "split_hash": split_hash,
        "split_disjoint": split.verify_disjoint(),
        "phenomenon_epoch": t_star,
        "gap_layer": gap_layer,
        "rho_dense": rho_dense,
        "rho_tail": rho_tail,
        "delta_rho": delta_rho,
        "first_train_loss": loss_trace[0],
        "last_train_loss": loss_trace[-1],
        "loss_drop": loss_trace[0] - loss_trace[-1],
        "n_geometry_logs": len(geometry_logs),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_report(out_dir / f"phase_a_seed{seed}.json", payload)

    geom_path = out_dir / f"phase_a_geometry_seed{seed}.json"
    geom_path.write_text(json.dumps(geometry_logs, indent=2), encoding="utf-8")

    append_csv_row(
        out_dir / "phase_a_summary.csv",
        {
            "seed": seed,
            "delta_rho": delta_rho,
            "rho_dense": rho_dense,
            "rho_tail": rho_tail,
            "loss_drop": payload["loss_drop"],
        },
    )
    return payload
