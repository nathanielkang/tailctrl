from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tailctrl.config import load_config
from tailctrl.phase_a import run_phase_a
from tailctrl.phase_b import run_phase_b


def main() -> int:
    parser = argparse.ArgumentParser(description="TailCtrl Phase A/B experiment runner.")
    parser.add_argument("--phase", choices=["A", "B", "both"], required=True)
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "tailctrl_default.yaml"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "seed": seed,
    }

    if args.phase in {"A", "both"}:
        results["phase_a"] = run_phase_a(cfg, out_dir / "phase_a", seed)

    if args.phase in {"B", "both"}:
        results["phase_b"] = run_phase_b(cfg, out_dir / "phase_b", seed)

    out_path = out_dir / f"run_{args.phase.lower()}_seed{seed}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
