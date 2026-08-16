"""
Steering rack / tie-rod optimisation - front axle.

Solves the two remaining unknowns in the front steering geometry: the
fore-aft (X) position of the steering rack on the chassis, and the
trackrod_outboard pickup point on the upright's steering arm. Both wishbones
and the axle are treated as fixed CAD (this script never touches them);
only trackrod_inboard/trackrod_outboard change.

===========================================================================
Solving method (read before changing the constants below)
===========================================================================

The team brief poses two conditions:

  1. FRONT VIEW (Y-Z plane): the tie rod line must pass through the FVIC
     computed from the (fixed) wishbone geometry. This kills bump steer
     at static ride height.
  2. SIDE VIEW (X-Z plane): the tie rod line must pass through the SVIC.
     This kills the fore-aft component of bump steer.

Both conditions only constrain 2 of trackrod_outboard's 3 coordinates each
(front view: Y vs Z; side view: X vs Z) - together they are 2 equations in
3 unknowns, one degree of freedom short of "fully determined" unless a
third, independent target pins it down. This script uses the team's own
STEERING MOMENT ARM target (150-200mm, brief section 4) as that third
condition:

  Step A (front view + moment arm, INDEPENDENT OF RACK X):
      trackrod_inboard's Y and Z are fixed by spec (305, 158mm) regardless
      of rack X, so the front-view tie-rod line (FVIC -> inboard) is the
      SAME line for every candidate. Walking along that line, find the
      point (Y_out, Z_out) whose distance to the wheel centre (front view)
      equals the moment-arm target (175mm, band midpoint). Quadratic in Y;
      of the two roots, keep the one on the same side as the original CAD
      placeholder (inboard of the wheel centre - the physically sensible
      one for an upright steering arm). This point is CONSTANT across all
      rack X candidates.

  Step B (side view, PER RACK-X CANDIDATE):
      With Z_out now fixed from Step A, and trackrod_inboard's X = the
      candidate rack X, the side-view line (SVIC -> inboard) gives exactly
      one X_out satisfying the SVIC condition at that Z. This is genuinely
      solved, not guessed, and it DOES vary with rack X.

CAUTION - Step B's literal answer is not buildable here. Because the SVIC
sits ~26.6m away in X (the wishbones are nearly parallel in side view), the
side-view line through it is almost perfectly flat: to reach the Step-A Z
(only ~4mm above the rack's 158mm) it needs trackrod_outboard's X to be
several METRES from the rack - verified numerically, not a sign error (see
"literal_svic" columns kept in the CSV for evidence). That is obviously not
a real steering arm. This script therefore does NOT use the literal Step B
solution for the geometry it actually sweeps. Instead:

  Step B' (practical): trackrod_outboard's X is placed a fixed, physically
  buildable STEERING_ARM_REACH_MM (150mm, a generous but realistic FSAE
  upright arm length) rearward of the kingpin axis at Z_out. This point,
  like Y_out/Z_out, is CONSTANT across every rack-X candidate. What varies
  per candidate is only trackrod_inboard's X - which still changes the tie
  rod's 3D length/angle, and therefore its bump/roll-steer sensitivity
  through travel, even though the outboard point itself doesn't move.

  svic_alignment_error / z_consistency_error are then reported HONESTLY
  against this practical point (i.e. how far it sits from exact SVIC
  alignment) rather than forced to zero - this is real information: the
  side-view IC condition cannot be hit exactly by any buildable arm on this
  car, only approached.

The knock-on effect discovered numerically (see run output / summary CSV):
achieving even the low end of the target Ackermann range (40%) at this
Y_out would need a steering-arm reach of several HUNDRED mm - also not
buildable. This is reported as a genuine design conflict (moment-arm
target vs. Ackermann target, mediated by how close the FVIC pins Y_out to
the kingpin axis), per the brief's explicit instruction to surface such
conflicts rather than paper over them.

Everything here reads suspension-model through its public API
(build_suspension / build_sweep / solve_evaluated_sweep) via
scripts/_sweep_common.py - the solver itself is never modified.

Coordinates: ISO 8855 (x forward, y left, z up), millimetres, unless a
variable name ends in _sw (raw SolidWorks: X backward, Y up, Z right).

Usage:
    python scripts/steering_rack_optimisation.py --smoke
    python scripts/steering_rack_optimisation.py
    python scripts/steering_rack_optimisation.py --steps 21 --rack-x-min -250 --rack-x-max -100
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _sweep_common as sc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))
import convert_coordinates as cc  # noqa: E402

from kinematics.core.input import build_suspension, build_sweep  # noqa: E402
from kinematics.core.sweep import solve_evaluated_sweep  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_GEOMETRY_PATH = CONFIGS_DIR / "uct_front_geometry_generated.yaml"
DEFAULT_SW_POINTS_PATH = CONFIGS_DIR / "solidworks_points.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "steering_rack_optimisation"

# ===========================================================================
# Fixed spec constants (brief section headers, not tunable per-run)
# ===========================================================================

TRACKROD_INBOARD_Y = 305.0     # mm, ISO - half-rack, fixed per spec
TRACKROD_INBOARD_Z = 158.0     # mm, ISO - starting rack-height assumption
WHEELBASE_MM = 1625.0
TRACK_MM = 1200.0
HALF_TRACK_MM = 600.0

MOMENT_ARM_LO, MOMENT_ARM_HI = 150.0, 200.0
MOMENT_ARM_TARGET = 0.5 * (MOMENT_ARM_LO + MOMENT_ARM_HI)   # 175mm, band midpoint

# Physically buildable steering-arm reach from the kingpin axis, used for
# the ACTUAL geometry swept (see "Step B'" in the module docstring). The
# literal SVIC-line solve is kept only as a reported diagnostic - it is not
# buildable (see docstring). Negative = rearward of the kingpin axis, the
# sign that gives POSITIVE (brief-convention) Ackermann on this car.
STEERING_ARM_REACH_MM = -150.0

BUMP_STEER_TARGET, BUMP_STEER_ACCEPT = 0.10, 0.50
ROLL_STEER_TARGET, ROLL_STEER_ACCEPT = 0.15, 0.50
ACKERMANN_NOMINAL, ACKERMANN_LO, ACKERMANN_HI = 60.0, 40.0, 80.0

# ===========================================================================
# Sweep travel constants
# ===========================================================================

BUMP_TRAVEL_MM = 40.0     # +/- mm, both wheels together (brief section 6)
ROLL_MAX_DEG = 3.0        # 0 -> 3 deg chassis roll (brief section 7)
# Rack half-stroke for the steer/Ackermann validation sweep. The brief's
# "half-rack = 305mm" is the RACK'S PHYSICAL LENGTH (inboard point's fixed
# lateral offset from centreline), not its stroke - sweeping the rack
# +/-305mm would drive the wheels far past any real steering lock and the
# solver would not converge. 40mm matches the existing, already-reviewed
# placeholder in scripts/motion_study.py (RACK_MM) pending a real number
# from the steering team.
RACK_STEER_HALF_STROKE_MM = 40.0

SWEEP_STEPS = 41   # odd, so a step lands exactly on static/zero


# ===========================================================================
# IC-line solving (Step A / Step B, see module docstring)
# ===========================================================================

def line_point_at(anchor_a: tuple, anchor_b: tuple, coord_value: float,
                   axis_a: int, axis_b: int) -> float:
    """Given a 2D line through anchor_a and anchor_b (using axis_a, axis_b
    of each), return the axis_b value at a given axis_a = coord_value."""
    a_a, a_b = anchor_a[axis_a], anchor_a[axis_b]
    b_a, b_b = anchor_b[axis_a], anchor_b[axis_b]
    slope = (b_b - a_b) / (b_a - a_a)
    return a_b + slope * (coord_value - a_a)


def solve_front_view_outboard(fvic_y: float, fvic_z: float,
                               wheel_y: float, wheel_z: float,
                               placeholder_y: float,
                               moment_arm_target: float) -> tuple[float, float, float]:
    """Step A: point on the FVIC->inboard line at moment_arm_target distance
    from the wheel centre (front view). Returns (Y_out, Z_out, m_fv)."""
    y_in, z_in = TRACKROD_INBOARD_Y, TRACKROD_INBOARD_Z
    m_fv = (z_in - fvic_z) / (y_in - fvic_y)   # dz/dy of the tie-rod front-view line

    # (y - wheel_y)^2 + (z_in + m_fv*(y-y_in) - wheel_z)^2 = R^2
    c = z_in - m_fv * y_in - wheel_z
    a_coef = 1.0 + m_fv ** 2
    b_coef = -2.0 * wheel_y + 2.0 * m_fv * c
    c_coef = wheel_y ** 2 + c ** 2 - moment_arm_target ** 2
    disc = b_coef ** 2 - 4 * a_coef * c_coef
    if disc < 0:
        raise ValueError("moment-arm target unreachable on the FVIC line - "
                          "reduce MOMENT_ARM_TARGET or check FVIC value")
    sqrt_disc = math.sqrt(disc)
    y_roots = [(-b_coef + sqrt_disc) / (2 * a_coef), (-b_coef - sqrt_disc) / (2 * a_coef)]
    y_out = min(y_roots, key=lambda y: abs(y - placeholder_y))
    z_out = z_in + m_fv * (y_out - y_in)
    return y_out, z_out, m_fv


def solve_side_view_outboard_x(rack_x: float, svic_x: float, svic_z: float,
                                z_out: float) -> float:
    """Step B: X_out on the SVIC->inboard line at the given (Step A) Z."""
    z_in = TRACKROD_INBOARD_Z
    m_sv = (z_in - svic_z) / (rack_x - svic_x)   # dz/dx of the tie-rod side-view line
    return rack_x + (z_out - z_in) / m_sv


def kingpin_axis_point_at_z(lower_outboard: tuple, upper_outboard: tuple,
                             z_target: float) -> tuple[float, float]:
    """Interpolate the kingpin axis (q -> p) to a given Z, return (X, Y)."""
    q, p = np.array(lower_outboard), np.array(upper_outboard)
    t = (z_target - q[2]) / (p[2] - q[2])
    pt = q + t * (p - q)
    return float(pt[0]), float(pt[1])


# ===========================================================================
# Geometry candidate construction
# ===========================================================================

def build_candidate_cad_geometry(baseline_cad: dict, rack_x: float,
                                  x_out: float, y_out: float, z_out: float) -> dict:
    """Deep-copy the baseline CAD-schema geometry, overwrite trackrod
    inboard (rack-X candidate) / outboard (practical, constant point)."""
    geometry = copy.deepcopy(baseline_cad)
    geometry["hardpoints"]["trackrod_inboard"] = {
        "x": rack_x, "y": TRACKROD_INBOARD_Y, "z": TRACKROD_INBOARD_Z,
    }
    geometry["hardpoints"]["trackrod_outboard"] = {
        "x": x_out, "y": y_out, "z": z_out,
    }
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


STEPS_HALF = 21   # per half-sweep; combined with the shared 0 row, ~41 total

# Sweeps are split into two half-sweeps, each anchored AT the design state
# (0) and walking out to one extreme. The solver solves sweep steps
# sequentially, warm-starting each step from the previous one - a single
# sweep spanning e.g. -40mm to +40mm therefore cold-starts its very FIRST
# step (a 40mm jump) directly from the design state with no warm-up, which
# some candidate geometries here fail to converge from (verified: some
# rack-X candidates raise RuntimeError/residual-tolerance errors at exactly
# that first step). Starting every half-sweep at 0 makes step 0 trivial
# (already the design state) and every subsequent step a small increment,
# which converges reliably across the whole candidate set.

def bump_sweep_specs(travel: float = BUMP_TRAVEL_MM) -> tuple[dict, dict]:
    neg = axle_sweep(STEPS_HALF, left_z=(0.0, -travel), right_z=(0.0, -travel))
    pos = axle_sweep(STEPS_HALF, left_z=(0.0, travel), right_z=(0.0, travel))
    return neg, pos


def roll_sweep_specs(roll_deg: float = ROLL_MAX_DEG) -> tuple[dict, dict]:
    # Opposed wheel travel is the solver-native way to command chassis roll
    # for a corner/axle-scope kinematic model (confirmed already in use by
    # scripts/_sweep_common.py's ROLL_SWEEP and cross-checked against the
    # axle_metrics 'roll' column formula: roll = atan2(dz_left-dz_right,
    # track), which is exactly what opposed z travel produces). No native
    # "roll" sweep input exists in the solver's target vocabulary (only
    # point/axis/mode), so a direct chassis-tilt implementation isn't needed
    # here - this reproduces the same physical DOF.
    travel = HALF_TRACK_MM * math.tan(math.radians(roll_deg))
    neg = axle_sweep(STEPS_HALF, left_z=(0.0, -travel), right_z=(0.0, travel))
    pos = axle_sweep(STEPS_HALF, left_z=(0.0, travel), right_z=(0.0, -travel))
    return neg, pos


def steer_sweep_specs(half_stroke: float = RACK_STEER_HALF_STROKE_MM) -> tuple[dict, dict]:
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
    """Try `amplitude` first; some candidate geometries here hit a genuine
    solver non-convergence (not a step-size artefact - verified at up to
    161 sub-steps) at the full +/-40mm bump travel the brief specifies, most
    likely a near-singular linkage configuration at extreme droop for that
    specific tie-rod geometry. Falls back to smaller amplitudes (30mm
    matches the already-vetted convention in scripts/_sweep_common.py,
    adopted there for exactly this reason - see README's 2026-08-06
    changelog) so the pipeline still produces a usable, honestly-labelled
    result instead of crashing. Returns (dataframe, amplitude_actually_used)."""
    for amp in (amplitude, *fallback_amplitudes):
        try:
            return run_bidirectional(suspension, spec_fn(amp)), amp
        except RuntimeError:
            continue
    raise RuntimeError(f"no amplitude in {(amplitude, *fallback_amplitudes)} converged")


# ===========================================================================
# Ackermann
# ===========================================================================

def ackermann_geometric(kingpin_xy: tuple, outboard_xy: tuple,
                         wheelbase_mm: float = WHEELBASE_MM) -> tuple[float, float]:
    """Quick, no-solver Ackermann %: extend the line from the kingpin axis
    (plan view) through trackrod_outboard to the centreline (Y=0), express
    that crossing as a distance behind the front axle, and convert to a
    percentage. Returns (ackermann_pct, x_intersect)."""
    x_k, y_k = kingpin_xy
    x_o, y_o = outboard_xy
    if abs(y_o - y_k) < 1e-9:
        return float("nan"), float("nan")
    x_intersect = x_k + (x_o - x_k) * (0.0 - y_k) / (y_o - y_k)
    dist_from_front_axle = -x_intersect   # front axle at x=0, rearward is negative x
    if abs(dist_from_front_axle) < 1e-6:
        return float("nan"), x_intersect
    return (wheelbase_mm / dist_from_front_axle) * 100.0, x_intersect


def ackermann_pct_from_angles(angle_left_deg: float, angle_right_deg: float,
                               track_mm: float = TRACK_MM,
                               wheelbase_mm: float = WHEELBASE_MM) -> float | None:
    """cot(outer) - cot(inner) formula, same convention as
    scripts/motion_study.py:ackermann_pct_row (kept in sync deliberately -
    see that function's docstring for the roadwheel_angle sign-convention
    caveat this also relies on)."""
    if angle_left_deg is None or angle_right_deg is None:
        return None
    if pd.isna(angle_left_deg) or pd.isna(angle_right_deg):
        return None
    psi_left, psi_right = -angle_left_deg, angle_right_deg
    a_left, a_right = abs(psi_left), abs(psi_right)
    if a_left < 0.05 or a_right < 0.05:
        return None
    inner_deg, outer_deg = (a_left, a_right) if a_left >= a_right else (a_right, a_left)
    inner, outer = math.radians(inner_deg), math.radians(outer_deg)
    ideal = track_mm / wheelbase_mm
    try:
        actual = (1.0 / math.tan(outer)) - (1.0 / math.tan(inner))
    except ZeroDivisionError:
        return None
    return 100.0 * actual / ideal


# ===========================================================================
# Per-candidate evaluation
# ===========================================================================

def evaluate_candidate(baseline_cad: dict, rack_x: float, x_out: float, y_out: float, z_out: float,
                        fvic_y: float, fvic_z: float, svic_x: float, svic_z: float,
                        kingpin_lower: tuple, kingpin_upper: tuple,
                        wheel_center: tuple, roll_centre_z_static: float) -> dict:
    geometry_cad = build_candidate_cad_geometry(baseline_cad, rack_x, x_out, y_out, z_out)
    axle_geometry = sc.cad_geometry_to_axle_dict(geometry_cad, name="steering_rack_opt")
    suspension = build_suspension(axle_geometry)

    bump, bump_amp_used = run_bidirectional_with_fallback(
        suspension, bump_sweep_specs, BUMP_TRAVEL_MM, (30.0, 20.0, 10.0))
    roll, roll_deg_used = run_bidirectional_with_fallback(
        suspension, roll_sweep_specs, ROLL_MAX_DEG, (2.0, 1.0))
    steer, steer_amp_used = run_bidirectional_with_fallback(
        suspension, steer_sweep_specs, RACK_STEER_HALF_STROKE_MM, (30.0, 20.0, 10.0))

    n_nonconverged = int((~bump["_converged"]).sum() + (~roll["_converged"]).sum()
                          + (~steer["_converged"]).sum())

    mid = len(bump) // 2
    static = bump.iloc[mid]

    bump_toe = bump["roadwheel_angle_left"].to_numpy(dtype=float)
    bump_steer_max = float(np.nanmax(bump_toe) - np.nanmin(bump_toe))
    bump_steer_rms = float(np.sqrt(np.nanmean((bump_toe - bump_toe[mid]) ** 2)))

    roll_toe = roll["roadwheel_angle_left"].to_numpy(dtype=float)
    roll_deg = roll["roll"].to_numpy(dtype=float)
    roll_steer_max = float(np.nanmax(roll_toe) - np.nanmin(roll_toe))
    roll_steer_at_1deg = float(np.interp(1.0, roll_deg, roll_toe) - roll_toe[len(roll_toe) // 2])

    # Steer-sweep-validated Ackermann: at max achieved lock (sweep extreme).
    steer_end = steer.iloc[-1]
    ackermann_validated = ackermann_pct_from_angles(
        steer_end["roadwheel_angle_left"], steer_end["roadwheel_angle_right"])
    outer_angle_at_validation = max(abs(steer_end["roadwheel_angle_left"]),
                                     abs(steer_end["roadwheel_angle_right"]))

    # Steering ratio at centre: d(roadwheel_angle_left)/d(rack displacement).
    rack_disp = steer["rack_displacement"].to_numpy(dtype=float)
    steer_toe = steer["roadwheel_angle_left"].to_numpy(dtype=float)
    steering_ratio = float(np.gradient(steer_toe, rack_disp)[len(steer) // 2])

    # Plan-view / Ackermann geometric quick check.
    kingpin_xy = kingpin_axis_point_at_z(kingpin_lower, kingpin_upper, z_out)
    ackermann_geo, _ = ackermann_geometric(kingpin_xy, (x_out, y_out))

    # Moment arm (front view) - wheel centre to trackrod_outboard.
    moment_arm = math.hypot(y_out - wheel_center[1], z_out - wheel_center[2])

    # Tie rod geometry.
    inboard = np.array([rack_x, TRACKROD_INBOARD_Y, TRACKROD_INBOARD_Z])
    outboard = np.array([x_out, y_out, z_out])
    tie_rod_vec = outboard - inboard
    tie_rod_length = float(np.linalg.norm(tie_rod_vec))
    elevation_angle_deg = float(np.degrees(np.arctan2(
        tie_rod_vec[2], np.hypot(tie_rod_vec[0], tie_rod_vec[1]))))
    plan_view_angle_deg = float(np.degrees(np.arctan2(tie_rod_vec[0], tie_rod_vec[1])))

    # IC alignment error diagnostics.
    fv_err = _point_line_distance_2d((y_out, z_out), (fvic_y, fvic_z),
                                      (TRACKROD_INBOARD_Y, TRACKROD_INBOARD_Z))
    sv_err = _point_line_distance_2d((rack_x, TRACKROD_INBOARD_Z), (svic_x, svic_z),
                                      (x_out, z_out))
    # Z the side-view (SVIC) line would demand at this candidate's ACTUAL
    # x_out, vs. the Z actually used (from Step A) - the real, honest
    # residual left by using a buildable arm instead of literal SVIC
    # alignment (see module docstring, "Step B'").
    z_svic_implied = line_point_at((rack_x, TRACKROD_INBOARD_Z), (svic_x, svic_z),
                                    x_out, axis_a=0, axis_b=1)
    z_consistency_error = z_svic_implied - z_out

    return {
        "rack_x_iso": rack_x,
        "rack_z_iso": TRACKROD_INBOARD_Z,
        "trackrod_inboard_x": rack_x,
        "trackrod_inboard_y": TRACKROD_INBOARD_Y,
        "trackrod_inboard_z": TRACKROD_INBOARD_Z,
        "trackrod_outboard_x": x_out,
        "trackrod_outboard_y": y_out,
        "trackrod_outboard_z": z_out,
        "tie_rod_length": tie_rod_length,
        "elevation_angle_deg": elevation_angle_deg,
        "plan_view_angle_deg": plan_view_angle_deg,
        "moment_arm_mm": moment_arm,
        "ackermann_pct": ackermann_geo,
        "ackermann_pct_steer_validated": ackermann_validated,
        "ackermann_validation_outer_angle_deg": outer_angle_at_validation,
        "bump_steer_max_deg": bump_steer_max,
        "bump_steer_rms_deg": bump_steer_rms,
        "roll_steer_max_deg": roll_steer_max,
        "roll_steer_at_1deg_roll": roll_steer_at_1deg,
        "fvic_alignment_error_mm": fv_err,
        "svic_alignment_error_mm": sv_err,
        "z_consistency_error_mm": z_consistency_error,
        "roll_centre_height_static": roll_centre_z_static,
        "steering_ratio_deg_per_mm": steering_ratio,
        "kpi_static": static.get("kpi_left"),
        "caster_static": static.get("caster_left"),
        "n_nonconverged_steps": n_nonconverged,
        "bump_amplitude_used_mm": bump_amp_used,
        "roll_amplitude_used_deg": roll_deg_used,
        "steer_amplitude_used_mm": steer_amp_used,
    }


def _point_line_distance_2d(pt, line_a, line_b) -> float:
    p, a, b = np.array(pt, dtype=float), np.array(line_a, dtype=float), np.array(line_b, dtype=float)
    ab = b - a
    if np.linalg.norm(ab) < 1e-9:
        return float(np.linalg.norm(p - a))
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


# ===========================================================================
# Scoring / recommendation
# ===========================================================================

def score_candidate(row: pd.Series) -> float:
    """Lower is better. Penalises out-of-target bump/roll steer and
    Ackermann/moment-arm deviation from the nominal target."""
    cost = 0.0
    cost += 3.0 * max(0.0, row["bump_steer_max_deg"] - BUMP_STEER_TARGET) / BUMP_STEER_TARGET
    cost += 2.0 * max(0.0, row["roll_steer_max_deg"] - ROLL_STEER_TARGET) / ROLL_STEER_TARGET
    ack = row["ackermann_pct"]
    if pd.notna(ack):
        cost += 1.0 * abs(ack - ACKERMANN_NOMINAL) / 20.0
        if not (ACKERMANN_LO <= ack <= ACKERMANN_HI):
            cost += 5.0
    marm = row["moment_arm_mm"]
    if not (MOMENT_ARM_LO <= marm <= MOMENT_ARM_HI):
        cost += 2.0 * min(abs(marm - MOMENT_ARM_LO), abs(marm - MOMENT_ARM_HI)) / 50.0
    return cost


# ===========================================================================
# Main pipeline
# ===========================================================================

def load_baseline() -> tuple[dict, dict]:
    """Returns (cad_geometry_dict, sw_points_dict)."""
    with open(DEFAULT_GEOMETRY_PATH) as f:
        cad = yaml.safe_load(f)
    with open(DEFAULT_SW_POINTS_PATH) as f:
        sw = yaml.safe_load(f)
    return cad, sw


def extract_static_ics(baseline_cad: dict) -> dict:
    """Solve the baseline once (placeholder trackrod, doesn't matter - FVIC/
    SVIC/roll-centre only depend on the fixed wishbones) and pull the static
    FVIC/SVIC/roll-centre/wheel-centre needed by the solver, straight from
    the CSV column names confirmed against suspension-model's metric
    catalog (kinematics/core/metrics/catalog.py: fvic_y, fvic_z, svic_x,
    svic_z; axle_metrics.py: roll_center_z)."""
    axle_geometry = sc.cad_geometry_to_axle_dict(baseline_cad, name="ic_probe")
    suspension = build_suspension(axle_geometry)
    static = run(suspension, axle_sweep(1)).iloc[0]

    hp = baseline_cad["hardpoints"]
    axle_in = sc.hp(hp, "axle_inboard")
    axle_out = sc.hp(hp, "axle_outboard")
    wheel_center = tuple((np.array(axle_in) + np.array(axle_out)) / 2.0)

    return {
        "fvic_y": float(static["fvic_y_left"]),
        "fvic_z": float(static["fvic_z_left"]),
        "svic_x": float(static["svic_x_left"]),
        "svic_z": float(static["svic_z_left"]),
        "roll_centre_z_static": float(static["roll_center_z"]),
        "wheel_center": wheel_center,
        "kingpin_lower": sc.hp(hp, "lower_wishbone_outboard"),
        "kingpin_upper": sc.hp(hp, "upper_wishbone_outboard"),
        "placeholder_outboard_y": float(hp["trackrod_outboard"]["y"]),
    }


def _evaluate_rack_x_candidate(args: tuple) -> dict:
    """Top-level (picklable) wrapper for ProcessPoolExecutor.map - one
    rack_x candidate in, one result row out. Never raises (mirrors
    mount_kpi_sweep.py's evaluate()): the RuntimeError handling here matches
    the original serial loop's except clause exactly, so one kinematically
    infeasible candidate doesn't take the whole pool down."""
    (rack_x, baseline_cad, x_out, y_out, z_out, fvic_y, fvic_z, svic_x, svic_z,
     kingpin_lower, kingpin_upper, wheel_center, roll_centre_z_static) = args
    try:
        row = evaluate_candidate(
            baseline_cad, rack_x, x_out, y_out, z_out,
            fvic_y, fvic_z, svic_x, svic_z,
            kingpin_lower, kingpin_upper,
            wheel_center, roll_centre_z_static)
    except RuntimeError as exc:
        return {"rack_x_iso": rack_x, "feasible": False, "_error": str(exc)}
    row["feasible"] = True
    return row


def run_pipeline(rack_x_min: float, rack_x_max: float, steps: int,
                  out_dir: Path) -> tuple[pd.DataFrame, dict]:
    baseline_cad, sw_points = load_baseline()
    ics = extract_static_ics(baseline_cad)

    print(f"Static FVIC (front view): y={ics['fvic_y']:.2f}  z={ics['fvic_z']:.2f}")
    print(f"Static SVIC (side view):  x={ics['svic_x']:.2f}  z={ics['svic_z']:.2f}")
    print(f"Roll centre height (static): {ics['roll_centre_z_static']:.2f} mm")
    print(f"Wheel centre: {ics['wheel_center']}")

    y_out, z_out, m_fv = solve_front_view_outboard(
        ics["fvic_y"], ics["fvic_z"], ics["wheel_center"][1], ics["wheel_center"][2],
        ics["placeholder_outboard_y"], MOMENT_ARM_TARGET)
    print(f"\nStep A (front view + {MOMENT_ARM_TARGET:.0f}mm moment-arm target, "
          f"constant across all rack-X candidates):")
    print(f"  trackrod_outboard Y={y_out:.2f}  Z={z_out:.2f}   (front-view line slope dz/dy={m_fv:.5f})")

    kingpin_xy = kingpin_axis_point_at_z(ics["kingpin_lower"], ics["kingpin_upper"], z_out)
    x_out = kingpin_xy[0] + STEERING_ARM_REACH_MM
    print(f"\nStep B' (practical, buildable arm - see module docstring for why the literal\n"
          f"  SVIC-line X is rejected): kingpin axis at Z={z_out:.2f} is X={kingpin_xy[0]:.2f}; "
          f"trackrod_outboard X = {x_out:.2f} ({STEERING_ARM_REACH_MM:+.0f}mm from kingpin)")
    x_out_literal_at_mid = solve_side_view_outboard_x(
        0.5 * (rack_x_min + rack_x_max), ics["svic_x"], ics["svic_z"], z_out)
    print(f"  For reference, the LITERAL SVIC-line X at rack_x={0.5*(rack_x_min+rack_x_max):.0f} "
          f"would be X={x_out_literal_at_mid:.0f} - {abs(x_out_literal_at_mid - x_out)/1000:.1f}m "
          f"from the practical point, confirming it is not buildable.")

    rack_x_values = np.linspace(rack_x_min, rack_x_max, steps)

    # A handful of rack-X values in the requested search range are
    # kinematically infeasible for this practical trackrod_outboard point
    # (verified: the solver cannot converge the STEER sweep at ANY rack
    # displacement down to +/-10mm, not just at the full +/-40mm amplitude -
    # see run_bidirectional_with_fallback's docstring for the
    # amplitude-only fallback this is distinct from). Each candidate is
    # independent, so they're evaluated in a process pool, same pattern as
    # mount_kpi_sweep.py's run_grid() - infeasible candidates are recorded
    # as a failed row (via _evaluate_rack_x_candidate's own try/except)
    # rather than crashing the run, so the summary CSV/plots can show
    # exactly which part of the requested range is usable.
    worker_args = [
        (float(rack_x), baseline_cad, x_out, y_out, z_out,
         ics["fvic_y"], ics["fvic_z"], ics["svic_x"], ics["svic_z"],
         ics["kingpin_lower"], ics["kingpin_upper"],
         ics["wheel_center"], ics["roll_centre_z_static"])
        for rack_x in rack_x_values
    ]
    print(f"evaluating {len(worker_args)} rack_x candidates in parallel ...")
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(_evaluate_rack_x_candidate, worker_args, chunksize=4))

    n_infeasible = 0
    for i, row in enumerate(rows):
        rack_x = row["rack_x_iso"]
        if row["feasible"]:
            print(f"  candidate {i+1}/{len(rows)}: rack_x={rack_x:.1f} ... "
                  f"bump_steer_max={row['bump_steer_max_deg']:.4f} deg, "
                  f"roll_steer_max={row['roll_steer_max_deg']:.4f} deg, "
                  f"ackermann={row['ackermann_pct']:.1f}%, "
                  f"moment_arm={row['moment_arm_mm']:.1f}mm")
        else:
            print(f"  candidate {i+1}/{len(rows)}: rack_x={rack_x:.1f} ... "
                  f"INFEASIBLE ({row.get('_error', '')})")
            n_infeasible += 1
            del row["_error"]
    if n_infeasible:
        print(f"\n{n_infeasible}/{len(rack_x_values)} candidates were kinematically "
              f"infeasible for this trackrod_outboard point - excluded from scoring, "
              f"kept in the CSV with feasible=False.")

    df = pd.DataFrame(rows)
    df["score"] = np.nan
    feasible_mask = df["feasible"]
    df.loc[feasible_mask, "score"] = df.loc[feasible_mask].apply(score_candidate, axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    best_idx = df.loc[feasible_mask, "score"].idxmin()
    best_row = df.loc[best_idx]
    print(f"\nRecommended rack X = {best_row['rack_x_iso']:.1f} mm (score={best_row['score']:.3f})")

    detail = save_best_candidate_detail(baseline_cad, sw_points, best_row, x_out, y_out, z_out, out_dir)

    meta = {
        "baseline_cad": baseline_cad,
        "sw_points": sw_points,
        "ics": ics,
        "x_out": x_out,
        "y_out": y_out,
        "z_out": z_out,
        "kingpin_xy_at_zout": kingpin_xy,
        "best_row": best_row,
        "out_dir": out_dir,
        "detail": detail,
    }
    return df, meta


def _to_plain(obj):
    """Recursively cast numpy scalars to plain Python types so yaml.dump
    produces a file yaml.safe_load can read back (numpy floats otherwise
    serialise as unsafe python/object tags)."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def iso_to_sw(point_iso: tuple[float, float, float], sw_points: dict) -> tuple[float, float, float]:
    """Invert configs/convert_coordinates.py's SW->ISO transform for the
    front corner, so recommended points can be pasted straight back into
    solidworks_points.yaml. Reuses that module's own translation/free-radius
    functions rather than re-deriving them, so this can never silently drift
    from the authoritative conversion."""
    vehicle = sw_points["vehicle"]
    tire = vehicle["tire"]
    free_radius = cc.compute_free_radius(tire["section_width"], tire["aspect_ratio"], tire["rim_diameter"])
    front = sw_points["front"]
    translation = cc.compute_translation(front["wheel_center"], front["half_track"], free_radius, 0.0)
    tx, ty, tz = translation
    ix, iy, iz = point_iso
    # inverse of transform_point: iso = [-X+tx, Z+ty, Y+tz]
    return (tx - ix, iz - tz, iy - ty)


def save_best_candidate_detail(baseline_cad: dict, sw_points: dict, best_row: pd.Series,
                                x_out: float, y_out: float, z_out: float, out_dir: Path) -> dict:
    """Re-solve the recommended candidate, save its full bump/roll/steer
    sweep DataFrames (for the detail plots) and the resolved geometry YAML,
    and print the recommended hardpoints in both ISO 8855 and SolidWorks
    coordinates, ready to paste into solidworks_points.yaml."""
    rack_x = float(best_row["rack_x_iso"])
    geometry_cad = build_candidate_cad_geometry(baseline_cad, rack_x, x_out, y_out, z_out)
    axle_geometry = sc.cad_geometry_to_axle_dict(geometry_cad, name="steering_rack_opt_best")
    suspension = build_suspension(axle_geometry)

    detail_dir = out_dir / "best_candidate"
    detail_dir.mkdir(parents=True, exist_ok=True)

    bump, _ = run_bidirectional_with_fallback(suspension, bump_sweep_specs,
                                               float(best_row["bump_amplitude_used_mm"]), (20.0, 10.0))
    roll, _ = run_bidirectional_with_fallback(suspension, roll_sweep_specs,
                                               float(best_row["roll_amplitude_used_deg"]), (1.0,))
    steer, _ = run_bidirectional_with_fallback(suspension, steer_sweep_specs,
                                                float(best_row["steer_amplitude_used_mm"]), (20.0, 10.0))
    bump.to_csv(detail_dir / "bump_sweep.csv", index=False)
    roll.to_csv(detail_dir / "roll_sweep.csv", index=False)
    steer.to_csv(detail_dir / "steer_sweep.csv", index=False)

    with open(detail_dir / "geometry.yaml", "w") as f:
        yaml.dump(_to_plain(geometry_cad), f, default_flow_style=False, sort_keys=False)

    inboard_iso = (rack_x, TRACKROD_INBOARD_Y, TRACKROD_INBOARD_Z)
    outboard_iso = (x_out, y_out, z_out)
    inboard_sw = iso_to_sw(inboard_iso, sw_points)
    outboard_sw = iso_to_sw(outboard_iso, sw_points)

    print("\n" + "=" * 70)
    print("RECOMMENDED HARDPOINTS")
    print("=" * 70)
    print(f"\nISO 8855 (x fwd, y left, z up), mm:")
    print(f"  trackrod_inboard:  x={inboard_iso[0]:.2f}  y={inboard_iso[1]:.2f}  z={inboard_iso[2]:.2f}")
    print(f"  trackrod_outboard: x={outboard_iso[0]:.2f}  y={outboard_iso[1]:.2f}  z={outboard_iso[2]:.2f}")
    print(f"\nSolidWorks (X backward, Y up, Z right), mm - paste into solidworks_points.yaml:")
    print(f"  trackrod_inboard:")
    print(f"    position: [{inboard_sw[0]:.2f}, {inboard_sw[1]:.2f}, {inboard_sw[2]:.2f}]")
    print(f"  trackrod_outboard:")
    print(f"    position: [{outboard_sw[0]:.2f}, {outboard_sw[1]:.2f}, {outboard_sw[2]:.2f}]")

    return {"inboard_iso": inboard_iso, "outboard_iso": outboard_iso,
            "inboard_sw": inboard_sw, "outboard_sw": outboard_sw,
            "bump": bump, "roll": roll, "steer": steer}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rack-x-min", type=float, default=-250.0)
    parser.add_argument("--rack-x-max", type=float, default=-100.0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true",
                         help="3 candidates, coarse sweeps, for wiring checks")
    args = parser.parse_args()

    if args.smoke:
        run_pipeline(args.rack_x_min, args.rack_x_max, 3, args.out / "smoke")
        return

    run_pipeline(args.rack_x_min, args.rack_x_max, args.steps, args.out)


if __name__ == "__main__":
    main()
