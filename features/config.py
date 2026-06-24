"""Minimal PATHS stub for world_cup_predictor_metrics.py.

The original underdog repo had a config.py that supplied PATHS. This stub
provides the same dict so that ``from config import PATHS`` does not crash
when world_cup_predictor_metrics is imported. Only PATHS["processed"] is
used by that module's main() function.
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

PATHS: dict[str, Path] = {
    "raw":       _ROOT / "data" / "raw",
    "processed": _ROOT / "data" / "processed",
    "external":  _ROOT / "data" / "external",
}
