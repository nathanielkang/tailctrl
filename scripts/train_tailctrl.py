from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tailctrl.config import get_device, load_config
from tailctrl.data import fit_stratum_labels, four_way_split, load_dataset
from tailctrl.bilevel import TailCtrlTrainer
from tailctrl.logging_utils import write_json_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TailCtrl on configured dataset.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "tailctrl_default.yaml"))
    parser.add_argument("--method", type=str, default="tailctrl")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs" / "train"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    device = get_device(cfg)

    dataset = load_dataset(cfg["data"], seed)
    split = four_way_split(dataset.x.shape[0], cfg["data"]["split_ratios"], seed)
    labels = fit_stratum_labels(
        dataset.y[split.meta_train],
        tail_quantile=float(cfg["strata"]["tail_quantile"]),
        dense_quantile=float(cfg["strata"]["dense_quantile"]),
        bandwidth=cfg["strata"].get("kde_bandwidth"),
    )

    trainer = TailCtrlTrainer(cfg, device)
    result = trainer.train(dataset, split, labels, args.method, seed=seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json_report(
        out / f"{args.method}_seed{seed}.json",
        {
            "method": result.method,
            "test_tail_mae": result.test_tail_mae,
            "test_mae": result.test_mae,
            "val_tail_mae": result.val_tail_mae,
            "policy_variance": result.policy_variance,
            "train_loss_trace": result.train_loss_trace,
            "wall_clock_sec": result.budget.wall_clock_sec,
            "forward_passes": result.budget.forward_passes,
        },
    )
    print(f"test_tail_mae={result.test_tail_mae:.6f}")
    print(f"policy_variance={result.policy_variance:.6f}")
    print(f"wrote={out / f'{args.method}_seed{seed}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
