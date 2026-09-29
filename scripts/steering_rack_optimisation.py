"""
Steering rack / tie-rod optimisation - front axle, standalone Stage D.

Runs AFTER Stage A/B/C lock the wishbone/kingpin geometry
(outputs/front/optimal/optimal_geometry.yaml). That geometry is treated as
fixed input here - this script never touches wishbone hardpoints, it only
solves for trackrod_inboard/trackrod_outboard.

===========================================================================
Solving method
===========================================================================

Zero bump steer at the static position requires the tie rod to lie along a
line through the instant axis, as seen in the front view (Y-Z plane): the
line from the rack inboard point through the FVIC (front-view instant
centre, computed from the fixed wishbones).

  1. Rack inboard point: Y = 305mm (half-rack, spec), Z = rack height
     (a constant, RACK_HEIGHT_OFFSET_MM above the lower wishbone chassis
     node), X = the kingpin axis's own X at that Z (see below - this
     replaces a literal side-view/SVIC line solve for X, which is
     degenerate on this car, see "SVIC note" below).

  2. Front-view line: through the rack inboard point (Y, Z fixed) and FVIC.
     This line is fixed for the whole run, independent of steering arm
     length - walking along it trades Y for Z.

  3. Steering arm length (the swept design variable, 80-200mm): the
     perpendicular/lateral distance from the kingpin axis to the tie rod's
     outboard attachment point. Combined with the front-view line, this
     pins down (Y_out, Z_out) in closed form (two linear equations, see
     solve_outboard_yz()).

  4. X_out (fore-aft): the outboard point sits behind the kingpin axis
     (rear-steer). Set from the kingpin axis's own X at Z_out, minus a
     fixed, physically-buildable rearward offset (--steering-arm-rearward-offset,
     default 60mm - tune this if bump steer / steering ratio trade badly).

SVIC note: the side-view instant centre (SVIC) is not used to solve any
coordinate here. On this car the SVIC sits tens of metres away in X (the
wishbones are nearly parallel in side view - a 100mm tie-rod height
difference implies a ~26m X shift on the literal SVIC line, verified
algebraically, not a numerical fluke). A literal 2-point line solve through
it is therefore not buildable. Its alignment error is still reported
(svic_alignment_error_mm) as an honest diagnostic - never zeroed, never
used to score or reject candidates.

Everything here reads suspension-model through its public API
(build_suspension / build_sweep / solve_evaluated_sweep) via
scripts/_sweep_common.py - the solver itself is never modified.

Coordinates: ISO 8855 (x forward, y left, z up), millimetres.

Usage:
    python scripts/steering_rack_optimisation.py --smoke
    python scripts/steering_rack_optimisation.py
    python scripts/steering_rack_optimisation.py --arm-length-min 100 --arm-length-max 180
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
import _sweep_common as sc  # noqa: E402

from kinematics.cli.visualization.api import visualize_geometry  # noqa: E402
from kinematics.core.input import build_suspension, build_sweep  # noqa: E402
from kinematics.core.sweep import solve_evaluated_sweep  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GEOMETRY_PATH = REPO_ROOT / "outputs" / "front" / "optimal" / "optimal_geometry.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "front" / "steering_rack_optimisation"

# ===========================================================================
# Fixed spec constants
# ===========================================================================

TRACKROD_INBOARD_Y = 305.0     # mm, ISO - half-rack, fixed per spec

BUMP_STEER_TARGET, BUMP_STEER_ACCEPT = 0.05, 0.50     # deg, matches _sweep_common.py TARGETS_FULL["bump_steer_max"]
ROLL_STEER_TARGET, ROLL_STEER_ACCEPT = 0.02, 0.05      # deg/deg
MIN_STEER_ANGLE_DEG = 25.0     # hard reject below this - not scored

# Rack half-stroke for the steer-ratio validation sweep. PLACEHOLDER pending
# a real number from the steering team. Shares this placeholder value with
# RACK_MM in scripts/motion_study.py and RACK_STEER_HALF_STROKE_MM's old
# value here - update together, never just one.
RACK_HALF_STROKE_MM = 40.0

# ===========================================================================
# Sweep travel constants (brief section 6/7)
# ===========================================================================

BUMP_TRAVEL_MM = 25.0     # +/- mm, both wheels together
ROLL_TRAVEL_MM = 25.0     # +/- mm, wheels opposed - matches
# _sweep_common.py's build_motion_sweep(-25, 25, 25, -25)
STEER_HALF_STROKE_VALIDATION_MM = 5.0   # small sweep just to measure the steering-ratio slope

STEPS_HALF = 25   # per half-sweep, as specified (25 steps each direction)

# Sweeps are split into two half-sweeps, each anchored AT the design state
# (0) and walking out to one extreme. The solver solves sweep steps
# sequentially, warm-starting each step from the previous one - a single
# sweep spanning e.g. -25mm to +25mm therefore cold-starts its very FIRST
# step directly from the design state with no warm-up, which some candidate
# geometries fail to converge from. Starting every half-sweep at 0 makes
# step 0 trivial (already the design state) and every subsequent step a
# small increment, which converges reliably across the whole candidate set.
# (Carried over from the previous version of this script - verified pattern.)


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
                       arm_length: float) -> tuple[float, float, float]:
    """Closed-form solve of the outboard point's (Y, Z), given:
      - the front-view instant-axis line (through the rack inboard point and
        FVIC) - the zero-bump-steer condition
      - the steering arm length: the outboard point sits `arm_length` mm
        laterally inboard of the kingpin axis's own Y at the outboard Z

    Two linear equations in (Y_out, Z_out):
        Z_out = rack_z + m_fv * (Y_out - rack_y)                  [front-view line]
        Y_out = Y_kingpin(Z_out) = kp_y0 + s*(Z_out - kp_z0) - arm_length  [arm length]

    Substituting gives one linear equation in Z_out. Returns (Y_out, Z_out, m_fv).
    """
    m_fv = (rack_z - fvic_z) / (rack_y - fvic_y)
    kp_y0, kp_z0 = kingpin_lower[1], kingpin_lower[2]
    s = (kingpin_upper[1] - kingpin_lower[1]) / (kingpin_upper[2] - kingpin_lower[2])

    denom = 1.0 - m_fv * s
    if abs(denom) < 1e-9:
        raise ValueError("front-view line is parallel to the kingpin axis in Y-Z - "
                          "no unique outboard point for this geometry")
    z_out = (rack_z + m_fv * (kp_y0 - s * kp_z0 - arm_length - rack_y)) / denom
    y_out = kp_y0 + s * (z_out - kp_z0) - arm_length
    return y_out, z_out, m_fv


def _point_line_distance_2d(pt, line_a, line_b) -> float:
    p, a, b = np.array(pt, dtype=float), np.array(line_a, dtype=float), np.array(line_b, dtype=float)
    ab = b - a
    if np.linalg.norm(ab) < 1e-9:
        return float(np.linalg.norm(p - a))
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


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


def bump_sweep_specs(travel: float = BUMP_TRAVEL_MM) -> tuple[dict, dict]:
    neg = axle_sweep(STEPS_HALF, left_z=(0.0, -travel), right_z=(0.0, -travel))
    pos = axle_sweep(STEPS_HALF, left_z=(0.0, travel), right_z=(0.0, travel))
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


def run_bidirectional_with_fallback(suspension, spec_fn, amplitude: float,
                                     fallback_amplitudes: tuple[float, ...]) -> tuple[pd.DataFrame, float]:
    """Try `amplitude` first; some candidate geometries hit genuine solver
    non-convergence at the full travel. Falls back to smaller amplitudes so
    the pipeline still produces a usable, honestly-labelled result instead
    of crashing. Returns (dataframe, amplitude_actually_used)."""
    for amp in (amplitude, *fallback_amplitudes):
        try:
            return run_bidirectional(suspension, spec_fn(amp)), amp
        except RuntimeError:
            continue
    raise RuntimeError(f"no amplitude in {(amplitude, *fallback_amplitudes)} converged")


# ===========================================================================
# Per-candidate evaluation
# ===========================================================================

def evaluate_candidate(baseline_cad: dict, arm_length: float,
                        x_in: float, rack_y: float, rack_z: float,
                        x_out: float, y_out: float, z_out: float,
                        svic_x: float, svic_z: float) -> dict:
    geometry_cad = build_candidate_cad_geometry(baseline_cad, x_in, rack_y, rack_z, x_out, y_out, z_out)
    axle_geometry = sc.cad_geometry_to_axle_dict(geometry_cad, name="steering_rack_opt")
    suspension = build_suspension(axle_geometry)

    bump, bump_amp_used = run_bidirectional_with_fallback(
        suspension, bump_sweep_specs, BUMP_TRAVEL_MM, (20.0, 10.0, 5.0))
    roll, roll_amp_used = run_bidirectional_with_fallback(
        suspension, roll_sweep_specs, ROLL_TRAVEL_MM, (20.0, 10.0, 5.0))
    steer, steer_amp_used = run_bidirectional_with_fallback(
        suspension, steer_sweep_specs, STEER_HALF_STROKE_VALIDATION_MM, (2.0, 1.0))

    n_nonconverged = int((~bump["_converged"]).sum() + (~roll["_converged"]).sum()
                          + (~steer["_converged"]).sum())

    bump_toe = bump["roadwheel_angle_left"].to_numpy(dtype=float)
    bump_steer_deg = float(np.nanmax(bump_toe) - np.nanmin(bump_toe))

    roll_toe = roll["roadwheel_angle_left"].to_numpy(dtype=float)
    roll_deg = roll["roll"].to_numpy(dtype=float)
    roll_steer_slope = float(np.polyfit(roll_deg, roll_toe, 1)[0])

    rack_disp = steer["rack_displacement"].to_numpy(dtype=float)
    steer_toe = steer["roadwheel_angle_left"].to_numpy(dtype=float)
    steering_ratio = float(np.polyfit(rack_disp, steer_toe, 1)[0])
    max_steer_angle_deg = abs(steering_ratio) * RACK_HALF_STROKE_MM

    tie_rod_vec = np.array([x_out, y_out, z_out]) - np.array([x_in, rack_y, rack_z])
    tie_rod_length = float(np.linalg.norm(tie_rod_vec))

    svic_alignment_error = _point_line_distance_2d(
        (x_out, z_out), (x_in, rack_z), (svic_x, svic_z))

    return {
        "arm_length_mm": arm_length,
        "trackrod_inboard_x": x_in,
        "trackrod_inboard_y": rack_y,
        "trackrod_inboard_z": rack_z,
        "trackrod_outboard_x": x_out,
        "trackrod_outboard_y": y_out,
        "trackrod_outboard_z": z_out,
        "tie_rod_length_mm": tie_rod_length,
        "bump_steer_deg": bump_steer_deg,
        "roll_steer_slope_deg_per_deg": roll_steer_slope,
        "steering_ratio_deg_per_mm": steering_ratio,
        "max_steer_angle_deg": max_steer_angle_deg,
        "svic_alignment_error_mm": svic_alignment_error,
        "n_nonconverged_steps": n_nonconverged,
        "bump_amplitude_used_mm": bump_amp_used,
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
    ok = (row["bump_steer_deg"] <= BUMP_STEER_ACCEPT
          and abs(row["roll_steer_slope_deg_per_deg"]) <= ROLL_STEER_ACCEPT
          and row["max_steer_angle_deg"] >= MIN_STEER_ANGLE_DEG)
    return "PASS" if ok else "FAIL"


# ===========================================================================
# Main pipeline
# ===========================================================================

def load_locked_geometry(path: Path) -> dict:
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


def _evaluate_arm_length_candidate(args: tuple) -> dict:
    """Top-level (picklable) wrapper for ProcessPoolExecutor.map - one arm
    length candidate in, one result row out. Never raises: a solver
    RuntimeError becomes a feasible=False row so one kinematically
    infeasible candidate doesn't take the whole pool down."""
    (arm_length, baseline_cad, x_in, rack_y, rack_z, x_out, y_out, z_out,
     svic_x, svic_z) = args
    try:
        row = evaluate_candidate(baseline_cad, arm_length, x_in, rack_y, rack_z,
                                  x_out, y_out, z_out, svic_x, svic_z)
    except RuntimeError as exc:
        return {"arm_length_mm": arm_length, "feasible": False, "_error": str(exc)}
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


def run_pipeline(geometry_path: Path, rack_height_offset: float,
                  arm_length_min: float, arm_length_max: float, arm_length_steps: int,
                  rearward_offset: float, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    baseline_cad = load_locked_geometry(geometry_path)
    ics = extract_static_ics(baseline_cad)

    print(f"Loaded locked geometry: {geometry_path}")
    print(f"Static FVIC (front view): y={ics['fvic_y']:.2f}  z={ics['fvic_z']:.2f}")
    print(f"Static SVIC (side view):  x={ics['svic_x']:.2f}  z={ics['svic_z']:.2f}")
    print(f"Kingpin axis: lower={ics['kingpin_lower']}  upper={ics['kingpin_upper']}")

    hp = baseline_cad["hardpoints"]
    rack_z = float(hp["lower_wishbone_inboard_front"]["z"]) + rack_height_offset
    rack_y = TRACKROD_INBOARD_Y
    x_in = kingpin_axis_point_at_z(ics["kingpin_lower"], ics["kingpin_upper"], rack_z)[0]
    print(f"\nRack inboard: x={x_in:.2f}  y={rack_y:.2f}  z={rack_z:.2f} "
          f"(rack_height_offset={rack_height_offset:.1f}mm above lower_wishbone_inboard_front)")

    arm_length_values = np.linspace(arm_length_min, arm_length_max, arm_length_steps)

    worker_args = []
    for arm_length in arm_length_values:
        y_out, z_out, m_fv = solve_outboard_yz(
            rack_y, rack_z, ics["fvic_y"], ics["fvic_z"],
            ics["kingpin_lower"], ics["kingpin_upper"], float(arm_length))
        x_out = kingpin_axis_point_at_z(ics["kingpin_lower"], ics["kingpin_upper"], z_out)[0] - rearward_offset
        worker_args.append((float(arm_length), baseline_cad, x_in, rack_y, rack_z,
                             x_out, y_out, z_out, ics["svic_x"], ics["svic_z"]))

    print(f"\nFront-view instant-axis line slope dz/dy = {m_fv:.5f} (fixed across all candidates)")
    print(f"evaluating {len(worker_args)} steering arm length candidates in parallel ...")
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(_evaluate_arm_length_candidate, worker_args, chunksize=4))

    n_infeasible = 0
    for i, row in enumerate(rows):
        arm_length = row["arm_length_mm"]
        if row["feasible"]:
            print(f"  candidate {i+1}/{len(rows)}: arm_length={arm_length:.1f}mm ... "
                  f"bump_steer={row['bump_steer_deg']:.4f} deg, "
                  f"roll_steer_slope={row['roll_steer_slope_deg_per_deg']:.4f} deg/deg, "
                  f"max_steer_angle={row['max_steer_angle_deg']:.1f} deg")
        else:
            print(f"  candidate {i+1}/{len(rows)}: arm_length={arm_length:.1f}mm ... "
                  f"INFEASIBLE ({row.get('_error', '')})")
            n_infeasible += 1
            del row["_error"]
    if n_infeasible:
        print(f"\n{n_infeasible}/{len(arm_length_values)} candidates were kinematically "
              f"infeasible - excluded from scoring, kept in the CSV with feasible=False.")

    df = pd.DataFrame(rows)
    df["cost"] = np.nan
    df["pass_fail"] = "FAIL"

    feasible_mask = df["feasible"]
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

    n_pass = int((df.loc[feasible_mask, "pass_fail"] == "PASS").sum())
    print(f"\n{n_pass}/{int(feasible_mask.sum())} feasible candidates passed all constraints "
          f"(bump_steer<={BUMP_STEER_ACCEPT}, |roll_steer_slope|<={ROLL_STEER_ACCEPT}, "
          f"max_steer_angle>={MIN_STEER_ANGLE_DEG})." if n_pass else
          f"\nNO candidates passed all constraints "
          f"(bump_steer<={BUMP_STEER_ACCEPT}, |roll_steer_slope|<={ROLL_STEER_ACCEPT}, "
          f"max_steer_angle>={MIN_STEER_ANGLE_DEG}).")

    ranked = df.loc[scoreable_mask].sort_values("cost")
    print("\nTop 5 candidates by cost:")
    print(ranked.head(5)[["arm_length_mm", "bump_steer_deg", "roll_steer_slope_deg_per_deg",
                           "steering_ratio_deg_per_mm", "max_steer_angle_deg",
                           "cost", "pass_fail"]].to_string(index=False))

    if ranked.empty:
        raise RuntimeError("no candidate survived the hard steer-angle rejection - "
                            "widen --arm-length-min/--arm-length-max or adjust "
                            "--steering-arm-rearward-offset")

    best_row = ranked.iloc[0]
    print(f"\nRecommended arm length = {best_row['arm_length_mm']:.1f} mm (cost={best_row['cost']:.3f})")

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

    bump, _ = run_bidirectional_with_fallback(suspension, bump_sweep_specs,
                                               float(best_row["bump_amplitude_used_mm"]), (10.0, 5.0))
    roll, _ = run_bidirectional_with_fallback(suspension, roll_sweep_specs,
                                               float(best_row["roll_amplitude_used_mm"]), (10.0, 5.0))
    steer, _ = run_bidirectional_with_fallback(suspension, steer_sweep_specs,
                                                float(best_row["steer_amplitude_used_mm"]), (1.0,))
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

    return {"inboard": (x_in, rack_y, rack_z), "outboard": (x_out, y_out, z_out),
            "bump": bump, "roll": roll, "steer": steer}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY_PATH,
                         help="locked Stage C geometry YAML")
    parser.add_argument("--rack-height-offset", type=float, default=30.0,
                         help="mm above lower_wishbone_inboard_front.z")
    parser.add_argument("--arm-length-min", type=float, default=80.0)
    parser.add_argument("--arm-length-max", type=float, default=200.0)
    parser.add_argument("--arm-length-steps", type=int, default=25)
    parser.add_argument("--steering-arm-rearward-offset", type=float, default=60.0,
                         help="mm rearward of the kingpin axis, at the outboard Z")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true",
                         help="3 candidates, coarse sweeps, for wiring checks")
    args = parser.parse_args()

    if args.smoke:
        run_pipeline(args.geometry, args.rack_height_offset,
                     args.arm_length_min, args.arm_length_max, 3,
                     args.steering_arm_rearward_offset, args.out / "smoke")
        return

    run_pipeline(args.geometry, args.rack_height_offset,
                 args.arm_length_min, args.arm_length_max, args.arm_length_steps,
                 args.steering_arm_rearward_offset, args.out)


if __name__ == "__main__":
    main()
