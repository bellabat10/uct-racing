"""
Steering rack / tie-rod optimisation - front axle, standalone Stage D.

Solves the two tie-rod hardpoints (trackrod_inboard, trackrod_outboard) on
top of the wishbone/kingpin geometry that Stages A/B/C have already locked.
Never touches a wishbone hardpoint.

===========================================================================
Before running
===========================================================================

Stage C must have run first: the default input is
outputs/front/optimal/optimal_geometry.yaml (written by
scripts/build_optimal_geometry.py, e.g. via scripts/run_front_pipeline.py).
This script is not part of run_front_pipeline.py - run it separately.

===========================================================================
Inputs
===========================================================================

  --geometry                        locked Stage C geometry (the only file read
                                    besides configs/steering_rack.yaml)
  --rack-height-z                   rack inboard (tie-rod inner) height, ISO z,
                                    mm above ground. Absolute, not an offset.
                                    Default RACK_HEIGHT_Z_MM = 188.24.
                                    From SolidWorks: ISO z = SolidWorks Y + 108.24
  TRACKROD_INBOARD_Y                rack inboard Y = 305mm, half-rack (fixed,
                                    awaiting confirmation - no CLI flag on purpose)
  --steering-arm-rearward-offset    how far the tie-rod outboard point sits behind
                                    the kingpin axis. Fixed, not swept.
                                    Default STEERING_ARM_REARWARD_OFFSET_MM = 60.
  --bump-travel / --droop-travel    wheel travel for the bump-steer sweep, mm up
                                    and down from ride height. Default 25 / 25:
                                    bump is the real damper-stroke limit, droop
                                    is a PLACEHOLDER until confirmed.
  configs/steering_rack.yaml        rack half-stroke (PLACEHOLDER 40mm) - sets
                                    the max steer angle

===========================================================================
Solving method
===========================================================================

Zero bump steer at the static position requires the tie rod to lie along a
line through the instant axis, as seen in the front view (Y-Z plane): the
line from the rack inboard point through the FVIC (front-view instant
centre, computed from the fixed wishbones).

  1. Rack inboard point: Y = TRACKROD_INBOARD_Y, Z = --rack-height-z,
     X = the kingpin axis's own X at that Z (see "SVIC note" below).

  2. Front-view line: through the rack inboard point and the FVIC. Fixed for
     the whole run.

  3. Steering arm INSET (the one swept design variable, default 40-200mm in
     5mm steps): how far the tie-rod outboard point sits laterally (in Y)
     inboard of the kingpin axis, at the outboard point's own Z. Combined
     with the front-view line this pins down (Y_out, Z_out) in closed form
     (see solve_outboard_yz()).

  4. X_out (fore-aft): the kingpin axis's X at Z_out minus the fixed
     rearward offset (rear-steer).

The inset is NOT the steering arm length. The true steering arm length -
the perpendicular distance from the kingpin axis to the tie-rod outboard
point - combines the inset and the rearward offset, and is reported in its
own column (steering_arm_length_mm).

SVIC note: the side-view instant centre (SVIC) is not used to solve any
coordinate here. On this car the SVIC sits tens of metres away in X (the
wishbones are nearly parallel in side view), so a literal line solve through
it is not buildable. Its alignment error is still reported
(svic_alignment_error_mm) as a diagnostic - never used to score or reject.

===========================================================================
Evaluation and scoring
===========================================================================

Each candidate is solved through three sweeps:
  - bump:  -DROOP_TRAVEL_MM to +BUMP_TRAVEL_MM (25 / 25mm), both wheels
           together. Stages A/C use +/-30mm, so this bump steer number is
           NOT directly comparable with Stage C's bump_steer_max.
  - roll:  +/-ROLL_TRAVEL_MM (25mm, = Stages A/C), wheels opposed
  - steer: +/-5mm of rack, only to measure the steering ratio (deg/mm)

Max steer angle = steering ratio x rack half-stroke - a linear
extrapolation, and the half-stroke is still a placeholder.

Roll steer is reported two ways: the slope of toe vs roll (scored, same
metric as Stage C) and the peak toe change from static anywhere in the roll
sweep (roll_toe_change_peak_deg, reported only). A curved, symmetric
response can have a near-zero slope and still a large peak.

  - Hard reject (not scored): max steer angle < MIN_STEER_ANGLE_DEG (25 deg).
  - Cost (lower is better): how far bump steer and |roll steer slope| sit
    beyond their targets. Steering ratio and roll toe peak are reported,
    never scored.
  - PASS: bump steer <= 0.05 deg AND |roll steer slope| <= 0.02 deg/deg AND
    not hard-rejected. The targets are read from _sweep_common.TARGETS_FULL,
    so a steering PASS means the same thing as a Stage C pass.

The recommended candidate is the lowest-cost one that isn't hard-rejected,
whether or not it passes.

===========================================================================
Outputs (in --out, default outputs/front/steering_rack_optimisation/)
===========================================================================

  summary.csv          one row per inset candidate (infeasible ones kept,
                       feasible=False)
  best_candidate/      bump/roll/steer sweep CSVs, geometry.yaml (locked
                       Stage C geometry + solved tie rod), linkage_4view.png

Plot with scripts/plot_steering_rack_optimisation.py.

Everything here reads suspension-model through its public API
(build_suspension / build_sweep / solve_evaluated_sweep) via
scripts/_sweep_common.py - the solver itself is never modified.

Coordinates: ISO 8855 (x forward, y left, z up), millimetres.

Usage:
    python scripts/steering_rack_optimisation.py --smoke
    python scripts/steering_rack_optimisation.py
    python scripts/steering_rack_optimisation.py --rack-height-z 220
    python scripts/steering_rack_optimisation.py --droop-travel 35
    python scripts/steering_rack_optimisation.py --arm-inset-min 40 --arm-inset-max 120
"""

from __future__ import annotations

import argparse
import copy
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _steering_config  # noqa: E402
import _sweep_common as sc  # noqa: E402

from kinematics.cli.visualization.api import visualize_geometry  # noqa: E402
from kinematics.core.input import build_suspension, build_sweep  # noqa: E402
from kinematics.core.sweep import solve_evaluated_sweep  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GEOMETRY_PATH = REPO_ROOT / "outputs" / "front" / "optimal" / "optimal_geometry.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "front" / "steering_rack_optimisation"


class SteeringOptimisationError(RuntimeError):
    """A run that can't produce a recommendation - reported by main() as a
    plain error message rather than a traceback."""


# ===========================================================================
# Fixed spec constants
# ===========================================================================

TRACKROD_INBOARD_Y = 305.0     # mm, ISO - half-rack, fixed per spec (awaiting confirmation)

# Rack inboard (tie-rod inner) height, ISO z, mm above ground - absolute.
# 188.24 = the previous "30mm above lower_wishbone_inboard_front" (158.24 + 30),
# kept as the default so results don't change until VD supplies a new value.
#
# Converting a SolidWorks height:  ISO z = SolidWorks Y + 108.24
# (the offset configs/convert_coordinates.py applies to the front corner: it
# moves the SolidWorks wheel centre, Y = 165.36, to the tyre free radius,
# 273.6mm. If the wheel centre or tyre in solidworks_points.yaml changes,
# this offset changes with it.)
RACK_HEIGHT_Z_MM = 188.24

# How far the tie-rod outboard point sits BEHIND the kingpin axis (rear-steer),
# measured in X at the outboard point's own Z. A fixed, buildable choice - not
# swept. It is half of the true steering arm length (the other half is the
# swept inset), so changing it changes steering ratio, max steer angle and
# the tie rod's plan-view angle.
STEERING_ARM_REARWARD_OFFSET_MM = 60.0

# Pass thresholds = the Stage A/C targets, read (not copied) from
# _sweep_common.TARGETS_FULL: (accept_min, accept_max, optimal_lo, optimal_hi).
BUMP_STEER_TARGET = sc.TARGETS_FULL["bump_steer_max"][1]      # 0.05 deg
ROLL_STEER_TARGET = sc.TARGETS_FULL["roll_steer_front"][1]    # 0.02 deg/deg (magnitude)
MIN_STEER_ANGLE_DEG = 25.0     # hard reject below this - not scored

# Rack half-stroke - PLACEHOLDER, single source of truth in
# configs/steering_rack.yaml (shared with scripts/motion_study.py).
RACK_HALF_STROKE_MM = _steering_config.rack_half_stroke_mm()
RACK_HALF_STROKE_IS_PLACEHOLDER = _steering_config.rack_half_stroke_is_placeholder()

# Swept variable: steering arm inset (see module docstring), mm.
ARM_INSET_MIN_MM = 40.0
ARM_INSET_MAX_MM = 200.0
ARM_INSET_STEP_MM = 5.0

# ===========================================================================
# Sweep travel constants
# ===========================================================================

# Bump-steer sweep travel, mm from ride height, both wheels together. Set to
# the car's real wheel travel rather than Stages A/C's +/-30mm
# (sc.BUMP_TRAVEL), so bump steer here is NOT directly comparable with
# Stage C's bump_steer_max - it covers less travel.
BUMP_TRAVEL_MM = 25.0    # upward (bump) - about 25mm, limited by the damper stroke
DROOP_TRAVEL_MM = 25.0   # downward (droop) - PLACEHOLDER until the real droop travel is confirmed
DROOP_TRAVEL_IS_PLACEHOLDER = True

ROLL_TRAVEL_MM = sc.ROLL_TRAVEL   # +/- mm, wheels opposed - same as Stages A/C
STEER_HALF_STROKE_VALIDATION_MM = 5.0   # small sweep just to measure the steering-ratio slope

# Fallback travel if the solver can't converge at the full amount: each
# attempt scales the requested travel down (1.0 = as requested).
TRAVEL_FALLBACK_SCALES = (1.0, 0.8, 0.4, 0.2)

STEPS_HALF = 25   # per half-sweep, as specified (25 steps each direction)

# Sweeps are split into two half-sweeps, each anchored AT the design state
# (0) and walking out to one extreme. The solver solves sweep steps
# sequentially, warm-starting each step from the previous one - a single
# sweep spanning e.g. -30mm to +30mm therefore cold-starts its very FIRST
# step directly from the design state with no warm-up, which some candidate
# geometries fail to converge from. Starting every half-sweep at 0 makes
# step 0 trivial (already the design state) and every subsequent step a
# small increment, which converges reliably across the whole candidate set.


# ===========================================================================
# IC-line / instant-axis solving
# ===========================================================================

def kingpin_axis_point_at_z(lower_outboard: tuple, upper_outboard: tuple,
                             z_target: float) -> tuple[float, float]:
    """Interpolate the kingpin axis (q -> p) to a given Z, return (X, Y)."""
    q, p = np.array(lower_outboard), np.array(upper_outboard)
    t = (z_target - q[2]) / (p[2] - q[2])
    pt = q + t * (p - q)
    return float(pt[0]), float(pt[1])


def solve_outboard_yz(rack_y: float, rack_z: float, fvic_y: float, fvic_z: float,
                       kingpin_lower: tuple, kingpin_upper: tuple,
                       arm_inset: float) -> tuple[float, float, float]:
    """Closed-form solve of the outboard point's (Y, Z), given:
      - the front-view instant-axis line (through the rack inboard point and
        FVIC) - the zero-bump-steer condition
      - the steering arm inset: the outboard point sits `arm_inset` mm
        laterally inboard of the kingpin axis's own Y at the outboard Z

    Two linear equations in (Y_out, Z_out):
        Z_out = rack_z + m_fv * (Y_out - rack_y)                  [front-view line]
        Y_out = Y_kingpin(Z_out) = kp_y0 + s*(Z_out - kp_z0) - arm_inset  [inset]

    Substituting gives one linear equation in Z_out. Returns (Y_out, Z_out, m_fv).
    """
    m_fv = (rack_z - fvic_z) / (rack_y - fvic_y)
    kp_y0, kp_z0 = kingpin_lower[1], kingpin_lower[2]
    s = (kingpin_upper[1] - kingpin_lower[1]) / (kingpin_upper[2] - kingpin_lower[2])

    denom = 1.0 - m_fv * s
    if abs(denom) < 1e-9:
        raise ValueError("front-view line is parallel to the kingpin axis in Y-Z - "
                          "no unique outboard point for this geometry")
    z_out = (rack_z + m_fv * (kp_y0 - s * kp_z0 - arm_inset - rack_y)) / denom
    y_out = kp_y0 + s * (z_out - kp_z0) - arm_inset
    return y_out, z_out, m_fv


def _point_line_distance_2d(pt, line_a, line_b) -> float:
    p, a, b = np.array(pt, dtype=float), np.array(line_a, dtype=float), np.array(line_b, dtype=float)
    ab = b - a
    if np.linalg.norm(ab) < 1e-9:
        return float(np.linalg.norm(p - a))
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def steering_arm_length(outboard: tuple, kingpin_lower: tuple, kingpin_upper: tuple) -> float:
    """True steering arm length: perpendicular 3D distance from the tie-rod
    outboard point to the kingpin axis (lower -> upper ball joint)."""
    p, a, b = (np.array(v, dtype=float) for v in (outboard, kingpin_lower, kingpin_upper))
    axis = (b - a) / np.linalg.norm(b - a)
    return float(np.linalg.norm(np.cross(p - a, axis)))


# ===========================================================================
# Geometry candidate construction
# ===========================================================================

def build_candidate_cad_geometry(baseline_cad: dict, x_in: float, rack_y: float, rack_z: float,
                                  x_out: float, y_out: float, z_out: float) -> dict:
    """Deep-copy the baseline (Stage C locked) geometry, overwrite
    trackrod_inboard/trackrod_outboard - the only hardpoints this stage solves for."""
    geometry = copy.deepcopy(baseline_cad)
    geometry["hardpoints"]["trackrod_inboard"] = {"x": x_in, "y": rack_y, "z": rack_z}
    geometry["hardpoints"]["trackrod_outboard"] = {"x": x_out, "y": y_out, "z": z_out}
    return geometry


# ===========================================================================
# Sweep specs
# ===========================================================================

def axle_sweep(steps: int, left_z=(0.0, 0.0), right_z=(0.0, 0.0),
                rack_y=(0.0, 0.0)) -> dict:
    return {
        "version": 1,
        "steps": steps,
        "targets": [
            {"point": "wheel_center", "side": "left", "direction": {"axis": "z"},
             "mode": "relative", "start": left_z[0], "stop": left_z[1]},
            {"point": "wheel_center", "side": "right", "direction": {"axis": "z"},
             "mode": "relative", "start": right_z[0], "stop": right_z[1]},
            {"point": "trackrod_inboard", "side": "left", "direction": {"axis": "y"},
             "mode": "relative", "start": rack_y[0], "stop": rack_y[1]},
        ],
    }


def bump_sweep_specs(bump: float = BUMP_TRAVEL_MM,
                      droop: float = DROOP_TRAVEL_MM) -> tuple[dict, dict]:
    neg = axle_sweep(STEPS_HALF, left_z=(0.0, -droop), right_z=(0.0, -droop))
    pos = axle_sweep(STEPS_HALF, left_z=(0.0, bump), right_z=(0.0, bump))
    return neg, pos


def roll_sweep_specs(travel: float = ROLL_TRAVEL_MM) -> tuple[dict, dict]:
    neg = axle_sweep(STEPS_HALF, left_z=(0.0, -travel), right_z=(0.0, travel))
    pos = axle_sweep(STEPS_HALF, left_z=(0.0, travel), right_z=(0.0, -travel))
    return neg, pos


def steer_sweep_specs(half_stroke: float = STEER_HALF_STROKE_VALIDATION_MM) -> tuple[dict, dict]:
    neg = axle_sweep(STEPS_HALF, rack_y=(0.0, -half_stroke))
    pos = axle_sweep(STEPS_HALF, rack_y=(0.0, half_stroke))
    return neg, pos


def run(suspension, spec: dict) -> pd.DataFrame:
    sweep = build_sweep(spec, suspension)
    ev = solve_evaluated_sweep(suspension, sweep)
    rows = [r.flat_row() for r in ev.metrics.rows]
    df = pd.DataFrame(rows)
    df["_converged"] = [sc.converged(s) for s in ev.solver_stats]
    return df


def run_bidirectional(suspension, specs: tuple[dict, dict]) -> pd.DataFrame:
    """Run the (negative-half, positive-half) sweep pair (see STEPS_HALF
    docstring) and stitch them into one DataFrame ordered from the negative
    extreme to the positive extreme, sharing a single row at 0."""
    neg_spec, pos_spec = specs
    neg_df = run(suspension, neg_spec).iloc[::-1].reset_index(drop=True)   # 0 -> -extreme, reversed
    pos_df = run(suspension, pos_spec).iloc[1:].reset_index(drop=True)     # skip duplicate 0 row
    return pd.concat([neg_df, pos_df], ignore_index=True)


def run_bidirectional_with_fallback(suspension, spec_fn,
                                     attempts: tuple[tuple[float, ...], ...]) -> tuple[pd.DataFrame, tuple[float, ...]]:
    """Try each argument tuple for `spec_fn` in turn (the requested travel
    first, then smaller ones); some candidate geometries hit genuine solver
    non-convergence at the full travel. Falling back keeps the pipeline
    producing a usable, honestly-labelled result instead of crashing.
    Returns (dataframe, arguments_actually_used)."""
    for args in attempts:
        try:
            return run_bidirectional(suspension, spec_fn(*args)), args
        except RuntimeError:
            continue
    raise RuntimeError(f"no travel in {attempts} converged")


def scaled_attempts(*travel: float, scales=TRAVEL_FALLBACK_SCALES) -> tuple[tuple[float, ...], ...]:
    """(travel...) at each fallback scale, e.g. scaled_attempts(25, 25) ->
    ((25, 25), (20, 20), (10, 10), (5, 5))."""
    return tuple(tuple(s * t for t in travel) for s in scales)


# ===========================================================================
# Per-candidate evaluation
# ===========================================================================

def roll_toe_change_peak(roll_deg: np.ndarray, roll_toe: np.ndarray) -> float:
    """Largest toe change from static (roll = 0) anywhere in the roll sweep,
    signed, deg. Reported alongside the slope because a symmetric (curved)
    toe-vs-roll response fits a near-zero slope even when the toe change
    at full roll is large."""
    static_toe = roll_toe[int(np.nanargmin(np.abs(roll_deg)))]
    change = roll_toe - static_toe
    return float(change[int(np.nanargmax(np.abs(change)))])


def evaluate_candidate(baseline_cad: dict, arm_inset: float,
                        x_in: float, rack_y: float, rack_z: float,
                        x_out: float, y_out: float, z_out: float,
                        svic_x: float, svic_z: float,
                        bump_travel: float, droop_travel: float) -> dict:
    geometry_cad = build_candidate_cad_geometry(baseline_cad, x_in, rack_y, rack_z, x_out, y_out, z_out)
    axle_geometry = sc.cad_geometry_to_axle_dict(geometry_cad, name="steering_rack_opt")
    suspension = build_suspension(axle_geometry)

    bump, (bump_used, droop_used) = run_bidirectional_with_fallback(
        suspension, bump_sweep_specs, scaled_attempts(bump_travel, droop_travel))
    roll, (roll_amp_used,) = run_bidirectional_with_fallback(
        suspension, roll_sweep_specs, scaled_attempts(ROLL_TRAVEL_MM))
    steer, (steer_amp_used,) = run_bidirectional_with_fallback(
        suspension, steer_sweep_specs,
        scaled_attempts(STEER_HALF_STROKE_VALIDATION_MM, scales=(1.0, 0.4, 0.2)))

    n_nonconverged = int((~bump["_converged"]).sum() + (~roll["_converged"]).sum()
                          + (~steer["_converged"]).sum())

    bump_toe = bump["roadwheel_angle_left"].to_numpy(dtype=float)
    bump_steer_deg = float(np.nanmax(bump_toe) - np.nanmin(bump_toe))

    roll_toe = roll["roadwheel_angle_left"].to_numpy(dtype=float)
    roll_deg = roll["roll"].to_numpy(dtype=float)
    roll_steer_slope = float(np.polyfit(roll_deg, roll_toe, 1)[0])
    roll_toe_peak = roll_toe_change_peak(roll_deg, roll_toe)
    roll_max_deg = float(np.nanmax(np.abs(roll_deg)))

    rack_disp = steer["rack_displacement"].to_numpy(dtype=float)
    steer_toe = steer["roadwheel_angle_left"].to_numpy(dtype=float)
    steering_ratio = float(np.polyfit(rack_disp, steer_toe, 1)[0])
    max_steer_angle_deg = abs(steering_ratio) * RACK_HALF_STROKE_MM

    tie_rod_vec = np.array([x_out, y_out, z_out]) - np.array([x_in, rack_y, rack_z])
    tie_rod_length = float(np.linalg.norm(tie_rod_vec))

    svic_alignment_error = _point_line_distance_2d(
        (x_out, z_out), (x_in, rack_z), (svic_x, svic_z))

    return {
        "arm_inset_mm": arm_inset,
        "trackrod_inboard_x": x_in,
        "trackrod_inboard_y": rack_y,
        "trackrod_inboard_z": rack_z,
        "trackrod_outboard_x": x_out,
        "trackrod_outboard_y": y_out,
        "trackrod_outboard_z": z_out,
        "tie_rod_length_mm": tie_rod_length,
        "bump_steer_deg": bump_steer_deg,
        "roll_steer_slope_deg_per_deg": roll_steer_slope,
        "roll_toe_change_peak_deg": roll_toe_peak,
        "roll_max_deg": roll_max_deg,
        "steering_ratio_deg_per_mm": steering_ratio,
        "max_steer_angle_deg": max_steer_angle_deg,
        "svic_alignment_error_mm": svic_alignment_error,
        "n_nonconverged_steps": n_nonconverged,
        "bump_travel_used_mm": bump_used,
        "droop_travel_used_mm": droop_used,
        "roll_amplitude_used_mm": roll_amp_used,
        "steer_amplitude_used_mm": steer_amp_used,
    }


# ===========================================================================
# Scoring
# ===========================================================================

def score_candidate(row: pd.Series) -> float:
    """Lower is better. Penalises bump steer and roll steer slope beyond
    target; steering ratio is reported only, never penalised."""
    cost = 0.0
    cost += 3.0 * max(0.0, row["bump_steer_deg"] - BUMP_STEER_TARGET) / BUMP_STEER_TARGET
    cost += 2.0 * max(0.0, abs(row["roll_steer_slope_deg_per_deg"]) - ROLL_STEER_TARGET) / ROLL_STEER_TARGET
    return cost


def pass_fail(row: pd.Series) -> str:
    ok = (row["bump_steer_deg"] <= BUMP_STEER_TARGET
          and abs(row["roll_steer_slope_deg_per_deg"]) <= ROLL_STEER_TARGET
          and row["max_steer_angle_deg"] >= MIN_STEER_ANGLE_DEG)
    return "PASS" if ok else "FAIL"


# ===========================================================================
# Main pipeline
# ===========================================================================

def load_locked_geometry(path: Path) -> dict:
    if not path.exists():
        raise SteeringOptimisationError(
            f"locked geometry not found: {path}\n"
            "Run Stages A/B/C first (python scripts/run_front_pipeline.py), "
            "or pass --geometry.")
    with open(path) as f:
        return yaml.safe_load(f)


def extract_static_ics(baseline_cad: dict) -> dict:
    """Solve the locked (Stage C) geometry once (placeholder trackrod,
    doesn't matter - FVIC/SVIC/kingpin only depend on the fixed wishbones)
    and pull the static FVIC/SVIC/kingpin points needed by the solver,
    straight from the CSV column names confirmed against suspension-model's
    metric catalog (kinematics/core/metrics/catalog.py: fvic_y, fvic_z,
    svic_x, svic_z)."""
    axle_geometry = sc.cad_geometry_to_axle_dict(baseline_cad, name="ic_probe")
    suspension = build_suspension(axle_geometry)
    static = run(suspension, axle_sweep(1)).iloc[0]

    hp = baseline_cad["hardpoints"]
    return {
        "fvic_y": float(static["fvic_y_left"]),
        "fvic_z": float(static["fvic_z_left"]),
        "svic_x": float(static["svic_x_left"]),
        "svic_z": float(static["svic_z_left"]),
        "kingpin_lower": sc.hp(hp, "lower_wishbone_outboard"),
        "kingpin_upper": sc.hp(hp, "upper_wishbone_outboard"),
    }


def _evaluate_arm_inset_candidate(args: tuple) -> dict:
    """Top-level (picklable) wrapper for ProcessPoolExecutor.map - one inset
    candidate in, one result row out. Never raises: a solver RuntimeError
    becomes a feasible=False row so one kinematically infeasible candidate
    doesn't take the whole pool down."""
    (arm_inset, baseline_cad, x_in, rack_y, rack_z, x_out, y_out, z_out,
     svic_x, svic_z, bump_travel, droop_travel) = args
    try:
        row = evaluate_candidate(baseline_cad, arm_inset, x_in, rack_y, rack_z,
                                  x_out, y_out, z_out, svic_x, svic_z,
                                  bump_travel, droop_travel)
    except RuntimeError as exc:
        return {"arm_inset_mm": arm_inset, "feasible": False, "_error": str(exc)}
    row["feasible"] = True
    return row


def _to_plain(obj):
    """Recursively cast numpy scalars to plain Python types so yaml.dump
    produces a file yaml.safe_load can read back."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def arm_inset_values(inset_min: float, inset_max: float, inset_step: float) -> np.ndarray:
    """Inclusive min..max in inset_step increments."""
    return np.arange(inset_min, inset_max + 0.5 * inset_step, inset_step)


def run_pipeline(geometry_path: Path, rack_height_z: float,
                  arm_inset_list: np.ndarray, rearward_offset: float,
                  out_dir: Path, bump_travel: float = BUMP_TRAVEL_MM,
                  droop_travel: float = DROOP_TRAVEL_MM) -> tuple[pd.DataFrame, dict]:
    baseline_cad = load_locked_geometry(geometry_path)
    ics = extract_static_ics(baseline_cad)

    print(f"Loaded locked geometry: {geometry_path}")
    print(f"Static FVIC (front view): y={ics['fvic_y']:.2f}  z={ics['fvic_z']:.2f}")
    print(f"Static SVIC (side view):  x={ics['svic_x']:.2f}  z={ics['svic_z']:.2f}")
    print(f"Kingpin axis: lower={ics['kingpin_lower']}  upper={ics['kingpin_upper']}")

    hp = baseline_cad["hardpoints"]
    rack_z = float(rack_height_z)
    rack_y = TRACKROD_INBOARD_Y
    x_in = kingpin_axis_point_at_z(ics["kingpin_lower"], ics["kingpin_upper"], rack_z)[0]
    lwb_front_z = float(hp["lower_wishbone_inboard_front"]["z"])
    print(f"\nRack inboard: x={x_in:.2f}  y={rack_y:.2f}  z={rack_z:.2f} "
          f"({rack_z - lwb_front_z:+.1f}mm vs lower_wishbone_inboard_front)")
    print(f"Steering arm rearward offset: {rearward_offset:.1f}mm (fixed)")
    droop_note = ("  ** PLACEHOLDER **" if DROOP_TRAVEL_IS_PLACEHOLDER
                  and droop_travel == DROOP_TRAVEL_MM else "")
    print(f"Bump-steer sweep: {bump_travel:.1f}mm bump, {droop_travel:.1f}mm droop{droop_note} "
          f"(Stage C uses +/-{sc.BUMP_TRAVEL:g}mm - not directly comparable)")
    print(f"Rack half-stroke: {RACK_HALF_STROKE_MM:.1f}mm"
          + ("  ** PLACEHOLDER - see configs/steering_rack.yaml **"
             if RACK_HALF_STROKE_IS_PLACEHOLDER else ""))

    worker_args = []
    arm_length_by_inset = {}
    for arm_inset in arm_inset_list:
        arm_inset = float(arm_inset)
        y_out, z_out, m_fv = solve_outboard_yz(
            rack_y, rack_z, ics["fvic_y"], ics["fvic_z"],
            ics["kingpin_lower"], ics["kingpin_upper"], arm_inset)
        x_out = kingpin_axis_point_at_z(ics["kingpin_lower"], ics["kingpin_upper"], z_out)[0] - rearward_offset
        arm_length_by_inset[arm_inset] = steering_arm_length(
            (x_out, y_out, z_out), ics["kingpin_lower"], ics["kingpin_upper"])
        worker_args.append((arm_inset, baseline_cad, x_in, rack_y, rack_z,
                             x_out, y_out, z_out, ics["svic_x"], ics["svic_z"],
                             bump_travel, droop_travel))

    print(f"\nFront-view instant-axis line slope dz/dy = {m_fv:.5f} (fixed across all candidates)")
    print(f"evaluating {len(worker_args)} steering arm inset candidates in parallel ...")
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(_evaluate_arm_inset_candidate, worker_args, chunksize=4))

    errors = []
    for i, row in enumerate(rows):
        arm_inset = row["arm_inset_mm"]
        row["steering_arm_length_mm"] = arm_length_by_inset[arm_inset]
        row["steering_arm_rearward_offset_mm"] = rearward_offset
        label = (f"  candidate {i+1}/{len(rows)}: inset={arm_inset:.1f}mm "
                 f"(arm length {row['steering_arm_length_mm']:.1f}mm) ... ")
        if row["feasible"]:
            print(label + f"bump_steer={row['bump_steer_deg']:.4f} deg, "
                  f"roll_steer_slope={row['roll_steer_slope_deg_per_deg']:.4f} deg/deg, "
                  f"roll_toe_peak={row['roll_toe_change_peak_deg']:+.3f} deg, "
                  f"max_steer_angle={row['max_steer_angle_deg']:.1f} deg")
        else:
            print(label + f"INFEASIBLE ({row['_error']})")
            errors.append(f"inset={arm_inset:.1f}mm: {row.pop('_error')}")

    if not rows or len(errors) == len(rows):
        raise SteeringOptimisationError(
            f"every one of the {len(rows)} candidates was kinematically infeasible - "
            "nothing to score or recommend. Solver errors:\n  " + "\n  ".join(errors)
            + "\nCheck --rack-height-z, the inset range and --steering-arm-rearward-offset.")
    if errors:
        print(f"\n{len(errors)}/{len(rows)} candidates were kinematically "
              f"infeasible - excluded from scoring, kept in the CSV with feasible=False.")

    df = pd.DataFrame(rows)
    lead = ["arm_inset_mm", "steering_arm_length_mm", "steering_arm_rearward_offset_mm"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df["cost"] = np.nan
    df["pass_fail"] = "FAIL"

    feasible_mask = df["feasible"].astype(bool)
    hard_reject_mask = feasible_mask & (df["max_steer_angle_deg"] < MIN_STEER_ANGLE_DEG)
    scoreable_mask = feasible_mask & ~hard_reject_mask

    df.loc[scoreable_mask, "cost"] = df.loc[scoreable_mask].apply(score_candidate, axis=1)
    df.loc[feasible_mask, "pass_fail"] = df.loc[feasible_mask].apply(pass_fail, axis=1)

    n_hard_rejected = int(hard_reject_mask.sum())
    if n_hard_rejected:
        print(f"\n{n_hard_rejected} candidate(s) hard-rejected: max_steer_angle_deg < "
              f"{MIN_STEER_ANGLE_DEG:.0f} deg (not scored).")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    criteria = (f"bump_steer<={BUMP_STEER_TARGET}, |roll_steer_slope|<={ROLL_STEER_TARGET}, "
                f"max_steer_angle>={MIN_STEER_ANGLE_DEG}")
    n_pass = int((df.loc[feasible_mask, "pass_fail"] == "PASS").sum())
    if n_pass:
        print(f"\n{n_pass}/{int(feasible_mask.sum())} feasible candidates passed all constraints ({criteria}).")
    else:
        print(f"\nNO candidates passed all constraints ({criteria}). "
              "The recommendation below is the least-bad candidate, not a pass.")

    ranked = df.loc[scoreable_mask].sort_values("cost")
    print("\nTop 5 candidates by cost:")
    print(ranked.head(5)[["arm_inset_mm", "steering_arm_length_mm", "bump_steer_deg",
                           "roll_steer_slope_deg_per_deg", "roll_toe_change_peak_deg",
                           "steering_ratio_deg_per_mm", "max_steer_angle_deg",
                           "cost", "pass_fail"]].to_string(index=False))

    if ranked.empty:
        raise SteeringOptimisationError(
            f"no candidate reached the {MIN_STEER_ANGLE_DEG:.0f} deg max steer angle "
            f"(rack half-stroke {RACK_HALF_STROKE_MM:.1f}mm) - lower --arm-inset-min "
            "or adjust --steering-arm-rearward-offset")

    best_row = ranked.iloc[0]
    print(f"\nRecommended inset = {best_row['arm_inset_mm']:.1f} mm "
          f"(steering arm length {best_row['steering_arm_length_mm']:.1f} mm, "
          f"cost={best_row['cost']:.3f}, {best_row['pass_fail']})")
    if best_row["arm_inset_mm"] <= float(np.min(arm_inset_list)):
        print("  NOTE: the recommendation is at the bottom of the swept inset range - "
              "the optimum may lie below it.")

    detail = save_best_candidate_detail(baseline_cad, best_row, out_dir)

    meta = {
        "baseline_cad": baseline_cad,
        "ics": ics,
        "rack_x_in": x_in,
        "rack_y": rack_y,
        "rack_z": rack_z,
        "best_row": best_row,
        "out_dir": out_dir,
        "detail": detail,
    }
    return df, meta


def save_best_candidate_detail(baseline_cad: dict, best_row: pd.Series, out_dir: Path) -> dict:
    """Re-solve the recommended candidate, save its full bump/roll/steer
    sweep DataFrames and the resolved geometry YAML."""
    x_in = float(best_row["trackrod_inboard_x"])
    rack_y = float(best_row["trackrod_inboard_y"])
    rack_z = float(best_row["trackrod_inboard_z"])
    x_out = float(best_row["trackrod_outboard_x"])
    y_out = float(best_row["trackrod_outboard_y"])
    z_out = float(best_row["trackrod_outboard_z"])

    geometry_cad = build_candidate_cad_geometry(baseline_cad, x_in, rack_y, rack_z, x_out, y_out, z_out)
    axle_geometry = sc.cad_geometry_to_axle_dict(geometry_cad, name="steering_rack_opt_best")
    suspension = build_suspension(axle_geometry)

    detail_dir = out_dir / "best_candidate"
    detail_dir.mkdir(parents=True, exist_ok=True)

    detail_scales = (1.0, 0.5, 0.25)
    bump, _ = run_bidirectional_with_fallback(
        suspension, bump_sweep_specs,
        scaled_attempts(float(best_row["bump_travel_used_mm"]), float(best_row["droop_travel_used_mm"]),
                        scales=detail_scales))
    roll, _ = run_bidirectional_with_fallback(
        suspension, roll_sweep_specs,
        scaled_attempts(float(best_row["roll_amplitude_used_mm"]), scales=detail_scales))
    steer, _ = run_bidirectional_with_fallback(
        suspension, steer_sweep_specs,
        scaled_attempts(float(best_row["steer_amplitude_used_mm"]), scales=(1.0, 0.2)))
    bump.to_csv(detail_dir / "bump_sweep.csv", index=False)
    roll.to_csv(detail_dir / "roll_sweep.csv", index=False)
    steer.to_csv(detail_dir / "steer_sweep.csv", index=False)

    with open(detail_dir / "geometry.yaml", "w") as f:
        yaml.dump(_to_plain(geometry_cad), f, default_flow_style=False, sort_keys=False)

    linkage_png = detail_dir / "linkage_4view.png"
    render_result = visualize_geometry(suspension, linkage_png)
    print(f"wrote {render_result.output_path}  (contact patch on ground: "
          f"{render_result.contact_patch_on_ground})")

    print("\n" + "=" * 70)
    print("RECOMMENDED HARDPOINTS (ISO 8855, mm)")
    print("=" * 70)
    print(f"  trackrod_inboard:  x={x_in:.2f}  y={rack_y:.2f}  z={rack_z:.2f}")
    print(f"  trackrod_outboard: x={x_out:.2f}  y={y_out:.2f}  z={z_out:.2f}")
    print(f"  steering arm: inset {best_row['arm_inset_mm']:.1f}mm, rearward offset "
          f"{best_row['steering_arm_rearward_offset_mm']:.1f}mm, "
          f"length {best_row['steering_arm_length_mm']:.1f}mm")
    print(f"  roll steer: slope {best_row['roll_steer_slope_deg_per_deg']:+.4f} deg/deg, "
          f"peak toe change {best_row['roll_toe_change_peak_deg']:+.3f} deg "
          f"over +/-{best_row['roll_max_deg']:.1f} deg roll (reported only)")

    return {"inboard": (x_in, rack_y, rack_z), "outboard": (x_out, y_out, z_out),
            "bump": bump, "roll": roll, "steer": steer}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Steering rack / tie-rod optimisation (Stage D). Needs Stage C's "
                    "outputs/front/optimal/optimal_geometry.yaml.")
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY_PATH,
                         help="locked Stage C geometry YAML")
    parser.add_argument("--rack-height-z", type=float, default=RACK_HEIGHT_Z_MM,
                         help="rack inboard (tie-rod inner) height, ISO z in mm above ground - "
                              f"absolute, not an offset (default {RACK_HEIGHT_Z_MM}). "
                              "From SolidWorks: ISO z = SolidWorks Y + 108.24")
    parser.add_argument("--arm-inset-min", type=float, default=ARM_INSET_MIN_MM,
                         help="smallest steering arm inset, mm - lateral distance of the "
                              "tie-rod outboard point inboard of the kingpin axis "
                              f"(default {ARM_INSET_MIN_MM:g})")
    parser.add_argument("--arm-inset-max", type=float, default=ARM_INSET_MAX_MM,
                         help=f"largest steering arm inset, mm (default {ARM_INSET_MAX_MM:g})")
    parser.add_argument("--arm-inset-step", type=float, default=ARM_INSET_STEP_MM,
                         help=f"inset step, mm (default {ARM_INSET_STEP_MM:g})")
    parser.add_argument("--steering-arm-rearward-offset", type=float,
                         default=STEERING_ARM_REARWARD_OFFSET_MM,
                         help="how far the tie-rod outboard point sits behind the kingpin "
                              "axis, mm, at the outboard Z - fixed, not swept "
                              f"(default {STEERING_ARM_REARWARD_OFFSET_MM:g})")
    parser.add_argument("--bump-travel", type=float, default=BUMP_TRAVEL_MM,
                         help="bump (upward) wheel travel for the bump-steer sweep, mm from ride "
                              f"height (default {BUMP_TRAVEL_MM:g}, the damper-stroke limit)")
    parser.add_argument("--droop-travel", type=float, default=DROOP_TRAVEL_MM,
                         help="droop (downward) wheel travel for the bump-steer sweep, mm from ride "
                              f"height (default {DROOP_TRAVEL_MM:g} - PLACEHOLDER until the real "
                              "droop travel is confirmed)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true",
                         help="wiring check: 3 candidates (min, mid and max of the inset range) "
                              "with the normal full-resolution sweeps; writes to <out>/smoke")
    args = parser.parse_args()

    if args.arm_inset_step <= 0 or args.arm_inset_min > args.arm_inset_max:
        parser.error("need --arm-inset-step > 0 and --arm-inset-min <= --arm-inset-max")
    if args.bump_travel <= 0 or args.droop_travel <= 0:
        parser.error("need --bump-travel > 0 and --droop-travel > 0")

    if args.smoke:
        insets = np.array([args.arm_inset_min, 0.5 * (args.arm_inset_min + args.arm_inset_max),
                           args.arm_inset_max])
        out_dir = args.out / "smoke"
    else:
        insets = arm_inset_values(args.arm_inset_min, args.arm_inset_max, args.arm_inset_step)
        out_dir = args.out

    try:
        run_pipeline(args.geometry, args.rack_height_z, insets,
                     args.steering_arm_rearward_offset, out_dir,
                     args.bump_travel, args.droop_travel)
    except SteeringOptimisationError as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
