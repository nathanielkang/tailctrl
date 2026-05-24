# TailCtrl

Reference implementation for **TailCtrl**: bilevel, geometry-aware training for deep tabular regression under imbalanced target densities. The method couples a standard MLP predictor with a controller that maps stratum-wise spectral diagnostics to per-layer penalty strengths and spectral floors.

This repository contains **source code only**. It does not ship datasets (loaded at runtime via OpenML or scikit-learn), pretrained weights, or precomputed result tables. Running the scripts creates local logs under `outputs/`, which is excluded from version control.

## Requirements

- Python 3.10+
- Dependencies: see `requirements.txt` (PyTorch, NumPy, SciPy, scikit-learn, pandas, PyYAML).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick verification (smoke test)

Run a short end-to-end check on synthetic data before any full benchmark:

```bash
python scripts/smoke_tailctrl.py --config config/smoke.yaml --out-dir outputs/smoke
```

A successful run writes JSON summaries under `outputs/smoke/` and exits with code 0.

## Reproducing the paper pipeline

Experiments use a **four-way split** (meta-train, meta-validation, model-selection validation, test) and **frozen KDE strata** on the target axis. Default hyperparameters are in `config/tailctrl_default.yaml`.

```bash
# Optional: dataset counts for manuscript tables (writes outputs/dataset_stats.csv locally)
python scripts/summarize_datasets.py --out outputs/dataset_stats.csv

# Phase A — ERM collapse diagnostic (12 tabular benchmarks, one seed)
python scripts/run_phase_a_suite.py --config config/tailctrl_default.yaml --seed 42

# Phase B — TailCtrl and in-repo ablations (ERM, fixed floor, dense-only, etc.)
python scripts/run_benchmark_suite.py --config config/tailctrl_default.yaml --seed 42 --out-dir outputs/suite

# Multi-seed runs (example: five seeds)
python scripts/run_multi_seed_suite.py --config config/tailctrl_default.yaml --seeds 0 1 2 3 4

# Aggregate JSON logs into summary CSVs (local outputs/aggregated/)
python scripts/aggregate_experiment_outputs.py --out-dir outputs/aggregated
```

Optional sensitivity sweeps over inner unroll depth, sparsity weight, and tail quantile:

```bash
python scripts/run_sensitivity_suite.py --config config/tailctrl_default.yaml --seed 42
```

## Package layout

| Path | Role |
|------|------|
| `src/tailctrl/data.py` | Dataset loaders, four-way splits, KDE stratum labels |
| `src/tailctrl/backbone.py` | Tabular MLP regressor |
| `src/tailctrl/geometry.py` | Participation ratio and stratum-conditioned geometry |
| `src/tailctrl/controller.py` | Geometry controller and ablation controllers |
| `src/tailctrl/losses.py` | Inner / outer objectives |
| `src/tailctrl/bilevel.py` | Bilevel training loop (truncated unroll) |
| `src/tailctrl/phase_a.py` | Phase A phenomenon diagnostic |
| `src/tailctrl/phase_b.py` | Phase B method comparison |
| `src/tailctrl/aggregate.py` | Post-hoc CSV aggregation helpers |
| `src/tailctrl/stats_tests.py` | Friedman / Nemenyi tests on ablation ranks |
| `config/*.yaml` | Data splits, model, training, and phase settings |
| `scripts/` | CLI entry points listed above |

## Scope

**Included:** TailCtrl, ERM, and ablation variants implemented in this tree (fixed floor, random policy, no outer loop, dense-only, etc.) on the shared MLP backbone.

## Citation

If you use this code, please cite the associated TailCtrl manuscript when available.

## License

Code is provided for research reproduction. See the repository license file if present.
