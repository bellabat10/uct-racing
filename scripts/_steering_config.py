"""
Loader for configs/steering_rack.yaml - the single source of truth for rack
values shared between scripts/steering_rack_optimisation.py and
scripts/motion_study.py. Edit the YAML, not this file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

STEERING_RACK_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "steering_rack.yaml"


def _load() -> dict:
    with open(STEERING_RACK_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def rack_half_stroke_mm() -> float:
    """Rack travel from centre to full lock (one direction), mm."""
    return float(_load()["half_stroke_mm"])


def rack_half_stroke_is_placeholder() -> bool:
    return bool(_load().get("half_stroke_is_placeholder", True))
