"""
Shared geometry-loading and solver-plumbing helpers for the mount/KPI sweep
(scripts/mount_kpi_sweep.py) and the final confirmatory solve
(scripts/build_optimal_geometry.py). Both need "build an axle-scope
suspension, run BUMP + ROLL, extract flat metric rows" - this module is the
one place that logic lives, instead of being copy-pasted a third time.

Coordinates: ISO 8855 - X forward, Y left, Z up, millimetres.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

MOTION_STEPS = 11          # must be odd so a step lands exactly on static
BUMP_TRAVEL = 30.0         # +/- mm, both wheels together
ROLL_TRAVEL = 25.0         # +/- mm, wheels opposed (~2.4 deg roll on 1200 track)


def load_generated_geometry(axle_position: str = "front") -> dict:
    path = CONFIGS_DIR / f"uct_{axle_position}_geometry_generated.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def hp(hardpoints: dict, name: str) -> tuple[float, float, float]:
    p = hardpoints[name]
    return (float(p["x"]), float(p["y"]), float(p["z"]))


def pt(t) -> dict:
    return {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])}


def cad_geometry_to_axle_dict(cad_geometry: dict, name: str = "axle",
                               axle_position: str = "front") -> dict:
    """Convert a geometry file in the CAD schema (flat `hardpoints:`, a
    `config:` block - configs/uct_front_geometry_generated.yaml's schema,
    also used by scripts/mount_kpi_sweep.py's best_zkpi_geometry.yaml
    handoff) into the solver's axle-scope input schema."""
    hp_flat = cad_geometry["hardpoints"]
    cfg = cad_geometry["config"]
    cg = cfg["cg_position"]
    tire_cfg = cfg["wheel"]["tire"]
    return {
        "type": "double_wishbone",
        "scope": "axle",
        "name": name,
        "version": "0.0.1",
        "units": "millimeters",
        "vehicle_config": {
            "cg_position": {"x": float(cg["x"]), "y": float(cg["y"]), "z": float(cg["z"])},
            "wheelbase": float(cfg["wheelbase"]),
        },
        "axle_config": {
            "axle_position": axle_position,
            "steering": {"type": "rack"},
            "actuation": {"type": "direct", "mount": "lower_wishbone"},
            "spring": {"type": "none"},
            "anti_roll": {"type": "none"},
            "heave_link": {"type": "none"},
            "wheel": {
                "offset": float(cfg["wheel"].get("offset") or 0),
                "tire": {
                    "aspect_ratio": float(tire_cfg["aspect_ratio"]),
                    "section_width": float(tire_cfg["section_width"]),
                    "rim_diameter": float(tire_cfg["rim_diameter"]),
                },
            },
        },
        "hardpoints": {
            "left": {n: {"x": float(v["x"]), "y": float(v["y"]), "z": float(v["z"])}
                     for n, v in hp_flat.items()}
        },
    }


def converged(info) -> bool:
    for attr in ("converged", "success", "solver_converged"):
        if hasattr(info, attr):
            return bool(getattr(info, attr))
    return True


def run_sweep(geometry: dict, sweep_spec: dict) -> list[dict]:
    """Solve one motion sweep, return flat metric rows (or raise)."""
    from kinematics.core.input import build_suspension, build_sweep
    from kinematics.core.sweep import solve_evaluated_sweep

    suspension = build_suspension(geometry)
    sweep = build_sweep(sweep_spec, suspension)
    ev = solve_evaluated_sweep(suspension, sweep)
    if not all(converged(s) for s in ev.solver_stats):
        raise RuntimeError("solver did not converge at every step")
    # Axle scope: each row is an AxleMetricRows. flat_row() gives one dict per
    # step - corner metrics carry a _left / _right suffix, axle metrics don't.
    return [r.flat_row() for r in ev.metrics.rows]


def slope(x, y) -> float | None:
    """Least-squares slope, ignoring None entries. None if too few points."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    if max(xs) - min(xs) < 1e-9:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def span(values) -> float | None:
    vals = [v for v in values if v is not None]
    return (max(vals) - min(vals)) if len(vals) >= 2 else None


def build_motion_sweep(left_start, left_stop, right_start, right_stop) -> dict:
    """Axle-scope motion sweep spec. Every actuator DOF must be pinned at
    every step, which is why the rack appears even when held at zero."""
    return {
        "version": 1,
        "steps": MOTION_STEPS,
        "targets": [
            {"point": "wheel_center", "side": "left", "direction": {"axis": "z"},
             "mode": "relative", "start": left_start, "stop": left_stop},
            {"point": "wheel_center", "side": "right", "direction": {"axis": "z"},
             "mode": "relative", "start": right_start, "stop": right_stop},
            {"point": "trackrod_inboard", "side": "left", "direction": {"axis": "y"},
             "mode": "relative", "start": 0.0, "stop": 0.0},
        ],
    }


BUMP_SWEEP = build_motion_sweep(-BUMP_TRAVEL, BUMP_TRAVEL, -BUMP_TRAVEL, BUMP_TRAVEL)
ROLL_SWEEP = build_motion_sweep(-ROLL_TRAVEL, ROLL_TRAVEL, ROLL_TRAVEL, -ROLL_TRAVEL)


def build_axle_suspension(geometry: dict):
    """Thin wrapper kept for call-site symmetry with run_sweep - builds the
    Suspension object without immediately solving anything."""
    from kinematics.core.input import build_suspension

    return build_suspension(geometry)


# ===========================================================================
# Shared scoring targets (Stage C restores the full 9; Stage A drops
# caster_static/mechanical_trail_static - see scripts/mount_kpi_sweep.py).
# name -> (accept_min, accept_max, optimal_lo, optimal_hi)
# Straight from iteration2_kpi_sweep.py's TARGETS/WEIGHTS (team priority doc).
# ===========================================================================

TARGETS_FULL = {
    "bump_steer_max":               (0.0,   0.05,  0.0,   0.0),
    "camber_gain_roll_pct":         (0.5,   0.9,   0.6,   0.8),
    "roll_centre_z_static":         (20.0,  40.0,  30.0,  30.0),
    "kpi_static":                   (5.0,   10.0,  6.0,   7.0),
    "scrub_radius_static":          (5.0,   20.0,  10.0,  10.0),
    "roll_steer_front":             (-0.02, 0.02,  0.0,   0.0),
    "caster_static":                (3.0,   7.0,   5.0,   6.0),
    "camber_gain_bump_per25mm":     (-0.8,  -0.2,  -0.5,  -0.3),
    "mechanical_trail_static":      (5.0,   20.0,  10.0,  15.0),
    "camber_static":                (-4.0,  -2.0,  -3.5,  -3.5),
}

WEIGHTS_FULL = {
    "bump_steer_max":           1.00,
    "camber_gain_roll_pct":     0.96,
    "roll_centre_z_static":     0.92,
    "kpi_static":               0.83,
    "scrub_radius_static":      0.83,
    "roll_steer_front":         0.83,
    "caster_static":            0.79,
    "camber_gain_bump_per25mm": 0.79,
    "mechanical_trail_static":  0.75,
    # camber_static is Tier 4: checked against its band in TARGETS_FULL (so
    # it still shows PASS/FAIL in reporting) but deliberately has no weight
    # here, so weighted_cost() below skips it - it's driven directly by
    # mount_kpi_sweep.py's own camber-correction rotation now, not by score
    # pressure, and double-counting it here would fight that rotation.
}

MAX_NORMALIZED_ERROR = 5.0  # cap, in units of "acceptance-band-widths past optimal"


def distance_from_optimal(value, opt_lo, opt_hi, band_lo, band_hi) -> float:
    if value is None or not np.isfinite(value):
        return MAX_NORMALIZED_ERROR
    if opt_lo <= value <= opt_hi:
        return 0.0
    span_ = max(band_hi - band_lo, 1e-9)
    raw = min(abs(value - opt_lo), abs(value - opt_hi)) / span_
    return min(raw, MAX_NORMALIZED_ERROR)


def weighted_cost(row: dict, targets: dict, weights: dict) -> float:
    """Targets present in `targets` but absent from `weights` (e.g.
    camber_static - Tier 4) are still checkable against their band
    elsewhere, but don't contribute to this sum."""
    return sum(
        weights[name] * distance_from_optimal(row.get(name), opt_lo, opt_hi, lo, hi)
        for name, (lo, hi, opt_lo, opt_hi) in targets.items()
        if name in weights
    )


def in_band(value, lo, hi) -> bool:
    return value is not None and lo <= value <= hi


def evaluate_candidate(geometry: dict) -> dict:
    """Run BUMP + ROLL on one axle-scope geometry and extract every metric
    used anywhere in TARGETS_FULL, plus reported-only and arm-length columns.
    Does not apply weighted_cost itself - Stage A scores against a 7-target
    subset, Stage C against the full 9, so that stays candidate-set-specific.
    """
    bump = run_sweep(geometry, BUMP_SWEEP)
    mid = len(bump) // 2  # static ride height

    static = bump[mid]

    heave_mm = [r.get("heave") for r in bump]
    camber_bump = [r.get("camber_left") for r in bump]
    rc_z_bump = [r.get("roll_center_z") for r in bump]
    camber_gain_heave = slope(heave_mm, camber_bump)
    camber_gain_bump_per25mm = camber_gain_heave * 25.0 if camber_gain_heave is not None else None
    rc_migration_heave = slope(heave_mm, rc_z_bump)

    # Physical front-leg length of the upper wishbone at this candidate -
    # both the chassis mount (z_up) and the ball joint (KPI-rotated) move,
    # so this is measured directly from the candidate's own geometry, not
    # against any fixed CAD reference.
    hp_left = geometry["hardpoints"]["left"]
    mount = hp(hp_left, "upper_wishbone_inboard_front")
    outboard = hp(hp_left, "upper_wishbone_outboard")
    upper_arm_length = float(np.linalg.norm(np.array(outboard) - np.array(mount)))

    bump_steer_max = span([r.get("roadwheel_angle_left") for r in bump])

    roll = run_sweep(geometry, ROLL_SWEEP)

    roll_deg = [r.get("roll") for r in roll]
    camber_l = [r.get("camber_left") for r in roll]
    rc_y_r = [r.get("roll_center_y") for r in roll]
    rc_z_r = [r.get("roll_center_z") for r in roll]
    roadwheel_l = [r.get("roadwheel_angle_left") for r in roll]

    dcam_droll = slope(roll_deg, camber_l)
    camber_gain_roll_pct = (1.0 - abs(dcam_droll)) if dcam_droll is not None else None
    rc_migration_roll = slope(roll_deg, rc_z_r)
    rc_migration_roll_y = span(rc_y_r)
    roll_steer_front = slope(roll_deg, roadwheel_l)

    return {
        "camber_static": static.get("camber_left"),
        "kpi_static": static.get("kpi_left"),
        "caster_static": static.get("caster_left"),
        "scrub_radius_static": static.get("scrub_radius_left"),
        "mechanical_trail_static": static.get("mechanical_trail_left"),
        "roll_centre_z_static": static.get("roll_center_z"),
        "bump_steer_max": bump_steer_max,
        "camber_gain_bump_per25mm": camber_gain_bump_per25mm,
        "camber_gain_roll_pct": camber_gain_roll_pct,
        "roll_steer_front": roll_steer_front,
        # reported-only (not in either Stage's scored target set)
        "rc_migration_roll": rc_migration_roll,
        "rc_migration_heave": rc_migration_heave,
        "rc_migration_roll_y": rc_migration_roll_y,
        "fvsa_length_left": static.get("fvsa_length_left"),
        "upper_arm_length": upper_arm_length,
    }
