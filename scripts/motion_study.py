"""
Motion study - front axle, scope=axle, steering=rack.

Extends the merged pipeline (mount_kpi_sweep.py -> caster_trail_analysis.py
-> build_optimal_geometry.py) with the load cases that pipeline never
exercises: steering through rack travel, combined roll+steer, single-wheel
bump, and a bump-steer gradient across several fixed steer angles. Built to
diagnose the bump_steer_max / roll_steer_front failures Stage C's
confirmatory solve found - those metrics were only ever measured as a slope
through pure heave/roll, with the rack held dead still the whole time. This
module actually moves it.

Also replaces build_optimal_geometry.py's caster/trail refinement (Stage B,
a purely closed-form geometric calculation with no solver verification) with
a solver-verified alternative: apply_steering_axis() constructs a commanded
kingpin-axis attitude, geometry_for_target_angles() then Newton-corrects that
command until the SOLVER's own kpi_left/caster_left metrics - not our own
hand-rolled formula - actually hit the target. Only achieved values (never
commanded ones) are reported.

IMPORTANT CAVEAT (read before treating any KPI x caster grid as two
independent knobs): in a double wishbone, the same two ball joints set KPI,
caster, scrub radius, mechanical trail, camber gain, AND roll centre height.
You cannot vary KPI and caster in isolation - a KPI x caster grid here is
really a ball-joint-position grid viewed through two of its consequences.
Always look at camber gain and roll_center_z alongside any KPI/caster
result, or a "better KPI" that quietly wrecked the roll centre will slip
through unnoticed.

SECOND CAVEAT: if the hardpoint brief ever constrains the outboard ball
joints to slide along a strictly VERTICAL line on the upright spine, sliding
both joints vertically changes the steering axis LENGTH, not its ANGLE -
under that constraint there is essentially zero KPI/caster freedom to find.
If geometry_for_target_angles() fails to converge across a wide swept range,
that is a finding about the upright's ball-joint freedom to take back to the
upright designer, not a bug in this module.

Does not modify anything in the suspension-model submodule - this is a pure
driver layer on top of the existing public `kinematics` API
(build_suspension / build_sweep / solve_evaluated_sweep), same pattern as
every other script in this repo.

Coordinates: ISO 8855 - X forward, Y left, Z up, millimetres.

Usage:
    python scripts/motion_study.py --smoke
    python scripts/motion_study.py --kpi 4 6 8 --caster 3 5 7
    python scripts/motion_study.py --kpi 6 --caster 5 --motions pure_steer bump_steer_family
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _sweep_common as sc  # noqa: E402

from kinematics.cli.visualization.main import build_render_model  # noqa: E402
from kinematics.core.input import build_suspension, build_sweep  # noqa: E402
from kinematics.core.sweep import solve_evaluated_sweep  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GEOMETRY_PATH = REPO_ROOT / "configs" / "uct_front_geometry_generated.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "motion_study"

# ===========================================================================
# Motion definitions
# ===========================================================================

HEAVE_MM = 30.0   # +/- mm, symmetric bump/droop
ROLL_MM = 25.0    # +/- mm per wheel, opposed (~2.4 deg roll at 1200mm track)
RACK_MM = 40.0    # PLACEHOLDER - confirm the real lock-to-lock half-stroke
                  # with the steering team. Anything near the ends of a
                  # steer sweep is untrustworthy until that number is real.

STEPS_TRAVEL = 61  # heave/roll/single-wheel-bump motions (steps.py must be odd for a static step)
STEPS_STEER = 81   # any motion where the rack value changes across the sweep

BUMP_STEER_FAMILY_RACK_VALUES = (-40.0, -20.0, 0.0, 20.0, 40.0)


def _axle_sweep(steps: int, left_z=(0.0, 0.0), right_z=(0.0, 0.0),
                 rack_y=(0.0, 0.0)) -> dict:
    """Axle-scope motion sweep spec: relative wheel_center Z (left, right)
    and relative trackrod_inboard Y (rack, left side only - shared DOF).
    Every actuator must be pinned at every step, which is why the rack
    target appears even in motions that hold it perfectly still."""
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


def pure_heave() -> dict:
    return _axle_sweep(STEPS_TRAVEL, left_z=(-HEAVE_MM, HEAVE_MM), right_z=(-HEAVE_MM, HEAVE_MM))


def pure_roll() -> dict:
    return _axle_sweep(STEPS_TRAVEL, left_z=(-ROLL_MM, ROLL_MM), right_z=(ROLL_MM, -ROLL_MM))


def pure_steer() -> dict:
    return _axle_sweep(STEPS_STEER, rack_y=(-RACK_MM, RACK_MM))


def roll_steer_ramp() -> dict:
    """Cornering-entry case: roll and steer build together, step 0 = straight
    and flat, last step = full roll at full lock. Written as a LEFT-HAND
    corner: body rolls right, so the RIGHT wheel is the loaded outside wheel
    and compresses (right Z: 0 -> +ROLL_MM), left droops (0 -> -ROLL_MM).
    Rack sign convention needs checking against this geometry - if it runs
    the other way, flip the two wheel targets rather than the rack, so the
    "outside wheel compresses" physical meaning stays correct."""
    return _axle_sweep(STEPS_STEER, left_z=(0.0, -ROLL_MM), right_z=(0.0, ROLL_MM),
                        rack_y=(0.0, RACK_MM))


def steer_at_fixed_roll() -> dict:
    return _axle_sweep(STEPS_STEER, left_z=(-ROLL_MM, -ROLL_MM), right_z=(ROLL_MM, ROLL_MM),
                        rack_y=(-RACK_MM, RACK_MM))


def bump_and_steer() -> dict:
    return _axle_sweep(STEPS_STEER, left_z=(-HEAVE_MM, HEAVE_MM), right_z=(-HEAVE_MM, HEAVE_MM),
                        rack_y=(-RACK_MM, RACK_MM))


def single_wheel_bump() -> dict:
    return _axle_sweep(STEPS_TRAVEL, left_z=(-HEAVE_MM, HEAVE_MM), right_z=(0.0, 0.0))


def bump_steer_family() -> list[tuple[str, dict]]:
    """One (label, sweep) pair per rack value - targets pair by step index,
    not a grid (a 2D bump x steer sweep isn't expressible as one SweepSpec),
    so this is N separate sweeps. Rack is HELD constant per run (not swept),
    so this follows the travel step-count convention (61) even though the
    point of running several is reading a gradient across steer angles."""
    return [
        (f"rack{int(v):+d}",
         _axle_sweep(STEPS_TRAVEL, left_z=(-HEAVE_MM, HEAVE_MM),
                     right_z=(-HEAVE_MM, HEAVE_MM), rack_y=(v, v)))
        for v in BUMP_STEER_FAMILY_RACK_VALUES
    ]


MOTIONS: dict[str, Callable] = {
    "pure_heave": pure_heave,
    "pure_roll": pure_roll,
    "pure_steer": pure_steer,
    "roll_steer_ramp": roll_steer_ramp,
    "steer_at_fixed_roll": steer_at_fixed_roll,
    "bump_and_steer": bump_and_steer,
    "single_wheel_bump": single_wheel_bump,
    "bump_steer_family": bump_steer_family,  # special-cased: returns a list, not a dict
}


def write_motion_yamls(out_dir: Path) -> None:
    """Write every motion out as a reviewable YAML, runnable through the
    `kinematics` CLI directly (`kinematics sweep --geometry=... --sweep=...`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in MOTIONS.items():
        if name == "bump_steer_family":
            for label, spec in fn():
                path = out_dir / f"{name}_{label}.yaml"
                with open(path, "w") as f:
                    yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
        else:
            path = out_dir / f"{name}.yaml"
            with open(path, "w") as f:
                yaml.dump(fn(), f, default_flow_style=False, sort_keys=False)
    print(f"wrote {len(MOTIONS) - 1 + len(BUMP_STEER_FAMILY_RACK_VALUES)} motion YAMLs to {out_dir}")


# ===========================================================================
# KPI / caster: commanded axis attitude -> solver-verified achieved values
# ===========================================================================

def apply_steering_axis(geometry: dict, kpi_deg: float, caster_deg: float,
                         pivot: str = "lower_ball_joint") -> dict:
    """Rebuild upper_wishbone_outboard (p) for a commanded kingpin-axis
    attitude, preserving the ball-joint separation along the axis and
    rotating about `pivot`. Only `hardpoints.left` is edited - the right
    side mirrors automatically at solve time. Deep-copies the input, never
    mutates it.

    Axis unit vector construction: given fixed length L = |p - q| and the
    two target angles (from vertical), ux/uz = -tan(caster) (matches
    caster_trail_analysis.py's XZ-plane sign convention) and
    uy/uz = -tan(kpi) (matches mount_kpi_sweep.py's YZ-plane convention),
    plus |u| = 1 (choosing uz > 0, p above q) fully determines u. This is a
    reasonable starting command, not the final word - geometry_for_target_angles()
    below corrects it against what the solver itself actually measures,
    which is what matters if this module's sign convention and the solver's
    own kpi_left/caster_left definitions don't agree exactly.

    Only pivot="lower_ball_joint" is implemented.
    """
    if pivot != "lower_ball_joint":
        raise NotImplementedError(
            f"pivot={pivot!r} not supported - only 'lower_ball_joint' is implemented"
        )

    variant = copy.deepcopy(geometry)
    hp_left = variant["hardpoints"]["left"]
    q = np.array(sc.hp(hp_left, "lower_wishbone_outboard"))
    p0 = np.array(sc.hp(hp_left, "upper_wishbone_outboard"))
    length = float(np.linalg.norm(p0 - q))

    tx = math.tan(math.radians(caster_deg))
    ty = math.tan(math.radians(kpi_deg))
    uz = 1.0 / math.sqrt(1.0 + tx**2 + ty**2)
    ux, uy = -uz * tx, -uz * ty
    p = q + length * np.array([ux, uy, uz])

    hp_left["upper_wishbone_outboard"] = sc.pt(p)
    return variant


def _measure_static(geometry: dict) -> tuple[float, float]:
    """Solve one static step (design condition, all actuators at 0) and
    return the solver's own (kpi_left, caster_left)."""
    suspension = build_suspension(geometry)
    sweep = build_sweep(_axle_sweep(1), suspension)
    ev = solve_evaluated_sweep(suspension, sweep)
    row = ev.metrics.rows[0].flat_row()
    return float(row["kpi_left"]), float(row["caster_left"])


def geometry_for_target_angles(geometry: dict, kpi_target: float, caster_target: float,
                                pivot: str = "lower_ball_joint", tol_deg: float = 0.01,
                                max_iter: int = 4) -> tuple[dict, float, float, int]:
    """Newton loop: command an axis attitude, measure what the solver
    actually reports, correct via a finite-difference 2x2 Jacobian, repeat.
    Returns (variant_geometry, achieved_kpi, achieved_caster, iterations).
    Only achieved values are ever meaningful downstream - never the
    intermediate commanded ones."""
    cmd_kpi, cmd_caster = kpi_target, caster_target
    eps = 0.5  # deg, finite-difference perturbation for the Jacobian

    variant = achieved_kpi = achieved_caster = None
    for iteration in range(1, max_iter + 1):
        variant = apply_steering_axis(geometry, cmd_kpi, cmd_caster, pivot)
        achieved_kpi, achieved_caster = _measure_static(variant)
        err_kpi = achieved_kpi - kpi_target
        err_caster = achieved_caster - caster_target
        if abs(err_kpi) < tol_deg and abs(err_caster) < tol_deg:
            return variant, achieved_kpi, achieved_caster, iteration

        k_kpi_pert, c_kpi_pert = _measure_static(
            apply_steering_axis(geometry, cmd_kpi + eps, cmd_caster, pivot))
        k_cas_pert, c_cas_pert = _measure_static(
            apply_steering_axis(geometry, cmd_kpi, cmd_caster + eps, pivot))

        jacobian = np.array([
            [(k_kpi_pert - achieved_kpi) / eps, (k_cas_pert - achieved_kpi) / eps],
            [(c_kpi_pert - achieved_caster) / eps, (c_cas_pert - achieved_caster) / eps],
        ])
        err = np.array([err_kpi, err_caster])
        try:
            delta = np.linalg.solve(jacobian, err)
        except np.linalg.LinAlgError:
            break
        cmd_kpi -= float(delta[0])
        cmd_caster -= float(delta[1])

    if abs(achieved_kpi - kpi_target) > tol_deg or abs(achieved_caster - caster_target) > tol_deg:
        print(f"  WARNING: KPI/caster Newton solve did not reach {tol_deg} deg tolerance "
              f"after {max_iter} iterations (target kpi={kpi_target}, caster={caster_target}; "
              f"achieved kpi={achieved_kpi:.4f}, caster={achieved_caster:.4f})")
    return variant, achieved_kpi, achieved_caster, max_iter


# ===========================================================================
# Derived columns (not solver-native)
# ===========================================================================

def _nan_series(n: int) -> pd.Series:
    return pd.Series([None] * n)


def add_track_change_column(df: pd.DataFrame) -> pd.Series:
    """`track_change` is not a real solver metric (verified against source -
    only `track` exists) - defined here as deviation from the sweep's own
    static (mid-step) value."""
    if "track" not in df.columns or len(df) == 0:
        return _nan_series(len(df))
    mid = len(df) // 2
    return df["track"] - df["track"].iloc[mid]


def ackermann_pct_row(roadwheel_angle_left, roadwheel_angle_right, track_mm=1200.0,
                       wheelbase_mm=1625.0) -> float | None:
    """100 x (cot(outer) - cot(inner)) / (track/wheelbase). 0% = parallel
    steer, 100% = pure Ackermann, negative = anti-Ackermann.

    `roadwheel_angle` is defined per-side as "positive = turned toward the
    vehicle centerline" (a mirrored toe-in convention - confirmed against
    source, kinematics/core/metrics/angles.py:calculate_toe) - so left and
    right are on OPPOSITE signs relative to a shared/true yaw angle: a real
    coherent steer event shows left and right with roughly the SAME sign in
    the solver's raw columns (verified empirically: ~+11 deg / -10 deg at
    one rack extreme), not comparable to a cot()-based formula directly.
    Convert to a common true-yaw sign first (negate left, since "toward
    centerline" is the -Y direction for the left wheel but +Y for the
    right), then compare magnitudes - the side with the larger swept angle
    is "inner", independent of which way the car is actually turning.
    Guards the singularity near zero steer.
    """
    if roadwheel_angle_left is None or roadwheel_angle_right is None:
        return None
    if pd.isna(roadwheel_angle_left) or pd.isna(roadwheel_angle_right):
        return None
    psi_left = -roadwheel_angle_left
    psi_right = roadwheel_angle_right
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


def add_ackermann_column(df: pd.DataFrame, track_mm=1200.0, wheelbase_mm=1625.0) -> pd.Series:
    if "roadwheel_angle_left" not in df.columns or "roadwheel_angle_right" not in df.columns:
        return _nan_series(len(df))
    return df.apply(
        lambda r: ackermann_pct_row(r["roadwheel_angle_left"], r["roadwheel_angle_right"],
                                     track_mm, wheelbase_mm),
        axis=1,
    )


def add_camber_wrt_road_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for side in ("left", "right"):
        col = f"camber_{side}"
        if col in df.columns and "roll" in df.columns:
            out[f"camber_wrt_road_{side}"] = df[col] - df["roll"]
        else:
            out[f"camber_wrt_road_{side}"] = _nan_series(len(df))
    return out


def add_motion_ratio_column(df: pd.DataFrame) -> pd.Series:
    """d(damper_length)/d(wheel_travel), left side. Requires damper_length_left
    and wheel_travel_left columns and genuine travel variation - returns an
    all-null column (never raises) if either is missing or degenerate, per
    this module's rule that every derived column must tolerate absence."""
    if "damper_length_left" not in df.columns or "wheel_travel_left" not in df.columns:
        return _nan_series(len(df))
    travel = df["wheel_travel_left"].to_numpy(dtype=float)
    damper = df["damper_length_left"].to_numpy(dtype=float)
    # np.gradient divides by adjacent-step spacing - a motion with no real
    # bump travel (e.g. pure_steer) has near-duplicate travel values step to
    # step, not just a small overall range, which is what actually blows up
    # the finite-difference denominators.
    if len(travel) < 3 or np.min(np.abs(np.diff(travel))) < 1e-6:
        return _nan_series(len(df))
    return pd.Series(np.gradient(damper, travel))


# ===========================================================================
# Point-position export
# ===========================================================================

def _point_position_frame(suspension, states: list) -> pd.DataFrame:
    render_model = build_render_model(suspension)
    rows = []
    for state in states:
        row = {}
        for name, (x, y, z) in render_model.positions(state).items():
            row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"] = x, y, z
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Solve one motion, assemble one tidy DataFrame
# ===========================================================================

def solve_motion(suspension, sweep_spec: dict) -> pd.DataFrame:
    sweep = build_sweep(sweep_spec, suspension)
    ev = solve_evaluated_sweep(suspension, sweep)

    rows = []
    for i, (stats, metric_row) in enumerate(zip(ev.solver_stats, ev.metrics.rows)):
        row = {"step_index": i, "solver_converged": sc.converged(stats)}
        row.update(metric_row.flat_row())
        rows.append(row)
    df = pd.DataFrame(rows)

    pos_df = _point_position_frame(suspension, ev.states)
    df = pd.concat([df.reset_index(drop=True), pos_df.reset_index(drop=True)], axis=1)

    df["track_change"] = add_track_change_column(df)
    df["ackermann_pct"] = add_ackermann_column(df)
    for col, values in add_camber_wrt_road_columns(df).items():
        df[col] = values
    df["motion_ratio_left"] = add_motion_ratio_column(df)
    return df


# ===========================================================================
# sae_to_iso
# ===========================================================================

def sae_to_iso(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert a point authored in SAE / ISO 4130 (Y right, Z down) to this
    solver's ISO 8855 frame (Y left, Z up): x -> x, y -> -y, z -> -z.

    WARNING: our front hardpoint document authors coordinates with positive
    SAE Y on the car's RIGHT side - which is therefore NEGATIVE ISO Y. If
    SAE-authored numbers ever get pasted directly under `hardpoints.left`
    without running them through this conversion, the mirror silently puts
    the whole axle on the wrong side, and every left/right asymmetry this
    module measures (Ackermann, jacking, roll steer) comes out backwards.
    """
    x, y, z = point
    return (x, -y, -z)


# ===========================================================================
# Runner
# ===========================================================================

def run_study(geometry: dict, out_dir: Path, kpi_list: list[float], caster_list: list[float],
              motion_names: list[str] | None = None, save_csvs: bool = False) -> pd.DataFrame:
    motion_names = motion_names or list(MOTIONS.keys())
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_csvs:
        write_motion_yamls(out_dir / "sweeps")

    index_rows = []
    for kpi_target in kpi_list:
        for caster_target in caster_list:
            print(f"variant KPI={kpi_target:.1f} caster={caster_target:.1f} ...")
            variant, achieved_kpi, achieved_caster, n_iter = geometry_for_target_angles(
                geometry, kpi_target, caster_target
            )
            tag = f"kpi{kpi_target:.1f}_caster{caster_target:.1f}"
            variant_dir = out_dir / "runs" / tag
            variant_dir.mkdir(parents=True, exist_ok=True)

            with open(variant_dir / "variant_geometry.yaml", "w") as f:
                yaml.dump(variant, f, default_flow_style=False, sort_keys=False)

            suspension = build_suspension(variant)
            nonconverged_total = 0

            for name in motion_names:
                fn = MOTIONS[name]
                specs = fn() if name != "bump_steer_family" else fn()
                if name == "bump_steer_family":
                    for label, spec in specs:
                        df = solve_motion(suspension, spec)
                        nonconverged_total += int((~df["solver_converged"]).sum())
                        if save_csvs:
                            df.to_csv(variant_dir / f"{name}_{label}.csv", index=False)
                else:
                    df = solve_motion(suspension, specs)
                    nonconverged_total += int((~df["solver_converged"]).sum())
                    if save_csvs:
                        df.to_csv(variant_dir / f"{name}.csv", index=False)

            print(f"  achieved kpi={achieved_kpi:.3f} caster={achieved_caster:.3f} "
                  f"({n_iter} Newton iterations), {nonconverged_total} non-converged steps total")

            index_rows.append({
                "kpi_target": kpi_target,
                "caster_target": caster_target,
                "kpi_achieved": achieved_kpi,
                "caster_achieved": achieved_caster,
                "newton_iterations": n_iter,
                "nonconverged_steps_total": nonconverged_total,
            })

    index_df = pd.DataFrame(index_rows)
    index_out = out_dir / "index.csv"
    index_df.to_csv(index_out, index=False)
    print(f"wrote {index_out}")
    return index_df


# ===========================================================================
# CLI
# ===========================================================================

def smoke() -> None:
    cad_geometry = yaml.safe_load(open(DEFAULT_GEOMETRY_PATH))
    axle_geometry = sc.cad_geometry_to_axle_dict(cad_geometry, name="motion_study_smoke")

    print("SMOKE: one variant (KPI=6.0, caster=5.0), pure_steer motion only")
    variant, achieved_kpi, achieved_caster, n_iter = geometry_for_target_angles(
        axle_geometry, 6.0, 5.0
    )
    print(f"  achieved kpi={achieved_kpi:.4f} caster={achieved_caster:.4f} "
          f"in {n_iter} Newton iteration(s)")

    suspension = build_suspension(variant)
    df = solve_motion(suspension, pure_steer())
    n_bad = int((~df["solver_converged"]).sum())
    print(f"  pure_steer: {len(df)} steps, {n_bad} non-converged")
    cols = ["step_index", "roadwheel_angle_left", "roadwheel_angle_right",
            "camber_left", "camber_right", "ackermann_pct"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].iloc[[0, len(df) // 2, -1]].to_string(index=False))
    print("\nIf these look sane, the wiring is correct - run the full grid next.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kpi", type=float, nargs="+", default=[7.0])
    parser.add_argument("--caster", type=float, nargs="+", default=[5.0])
    parser.add_argument("--motions", nargs="+", default=None, choices=list(MOTIONS.keys()))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-csvs", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return

    cad_geometry = yaml.safe_load(open(args.geometry))
    axle_geometry = sc.cad_geometry_to_axle_dict(cad_geometry, name="motion_study")
    run_study(axle_geometry, args.out, args.kpi, args.caster, args.motions, args.save_csvs)


if __name__ == "__main__":
    main()
