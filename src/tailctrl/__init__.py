"""TailCtrl: bilevel tail-geometry control for deep imbalanced regression."""

from tailctrl.bilevel import TailCtrlTrainer, TailCtrlTrainResult
from tailctrl.phase_a import run_phase_a
from tailctrl.phase_b import run_phase_b

__all__ = [
    "TailCtrlTrainer",
    "TailCtrlTrainResult",
    "run_phase_a",
    "run_phase_b",
]
