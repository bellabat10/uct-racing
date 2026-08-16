"""
Stage C (rear) - assemble the final recommended rear geometry from rear
Stage A (mount height / KPI sweep) and rear Stage B (caster / mechanical
trail refinement), re-solve it once to confirm nothing drifted, render the
annotated wheel view, and write everything into outputs/rear/optimal/.
Cloned from scripts/build_optimal_geometry.py (front) - read that file
first if this one is confusing.

Differences from the front script:

  Paths: Stage A/B inputs and all outputs point at the outputs/rear/... tree
  (see DEFAULT_STAGE_A/DEFAULT_STAGE_B/STAGE_A_RESULTS_CSV/
  STAGE_A_BEST_TRAVEL_PNG below), and the written files are suffixed _rear
  (optimal_geometry_rear.yaml, optimal_summary_rear.csv,
  RECOMMENDATION_rear.md, linkage_4view_rear.png, wheel_view_rear.png).

  No rack: like scripts/rear_mount_kpi_sweep.py, this can't reuse
  _sweep_common.py's cad_geometry_to_axle_dict() (hardcodes
  steering.type="rack") or evaluate_candidate()/BUMP_SWEEP (hardcoded to pin
  trackrod_inboard, which doesn't exist on a toe-link corner). This file
  defines cad_geometry_to_axle_dict_rear() locally and reuses
  scripts/rear_mount_kpi_sweep.py's evaluate_candidate_rear()/
  BUMP_SWEEP_REAR instead - _sweep_common.py itself is untouched.

  Targets/scoring: uses rear_mount_kpi_sweep.py's TARGETS_REAR/WEIGHTS_REAR
  (rear-specific roll centre height and roll steer bands) as the base,
  extended with caster_static/mechanical_trail_static (same bands as front -
  now meaningful once Stage B has actually set them, exactly like the front
  script reinstates them at this stage). bump_steer_max is excluded from
  TARGETS_FULL_REAR entirely (not just unweighted) - it's a placeholder
  toe-link artifact (see rear_mount_kpi_sweep.py's module docstring), not a
  real geometry problem this optimiser can act on, so it's dropped from the
  pass/fail table rather than shown as a permanent FAIL.

Usage:
    python scripts/rear_build_optimal_geometry.py
    python scripts/rear_build_optimal_geometry.py --stage-a outputs/rear/mount_height_kpi_sweep/best_zkpi_geometry_rear.yaml --stage-b outputs/rear/caster_and_trail/best_caster_offset.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _sweep_common as sc  # noqa: E402
import rear_mount_kpi_sweep as rm  # noqa: E402
import wheel_view  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "rear" / "optimal"

DEFAULT_STAGE_A = REPO_ROOT / "outputs" / "rear" / "mount_height_kpi_sweep" / "best_zkpi_geometry_rear.yaml"
DEFAULT_STAGE_B = REPO_ROOT / "outputs" / "rear" / "caster_and_trail" / "best_caster_offset.yaml"
STAGE_A_RESULTS_CSV = REPO_ROOT / "outputs" / "rear" / "mount_height_kpi_sweep" / "rear_mount_kpi_results.csv"
STAGE_A_BEST_TRAVEL_PNG = REPO_ROOT / "outputs" / "rear" / "mount_height_kpi_sweep" / "rear_mount_kpi_best_vs_travel.png"

# Full (Stage C) target set for the rear: rm.TARGETS_REAR's 6 weighted
# targets (rear-specific roll centre / roll steer bands) plus camber_static
# (Tier 4, inherited from rm.TARGETS_REAR - checked, never weighted), plus
# caster_static/mechanical_trail_static reinstated now that Stage B has
# actually set them (same bands as front - no rear-specific brief for
# these). bump_steer_max is deliberately absent (see module docstring).
TARGETS_FULL_REAR = {
    **rm.TARGETS_REAR,
    "caster_static": sc.TARGETS_FULL["caster_static"],
    "mechanical_trail_static": sc.TARGETS_FULL["mechanical_trail_static"],
}
WEIGHTS_FULL_REAR = {
    **rm.WEIGHTS_REAR,
    "caster_static": sc.WEIGHTS_FULL["caster_static"],
    "mechanical_trail_static": sc.WEIGHTS_FULL["mechanical_trail_static"],
}

# The metrics rear Stage A actually scored/predicted (rm.WEIGHTS_REAR's
# keys, plus camber_static) - compared against what this stage's
# confirmatory re-solve produces on the FINAL (caster/offset-shifted)
# geometry. No bump_steer_max (not scored anywhere for the rear) and no
# caster_static/mechanical_trail_static (Stage A never touches those).
DRIFT_CHECK_METRICS = [
    "roll_centre_z_static", "camber_gain_bump_per25mm", "camber_gain_roll_pct",
    "kpi_static", "scrub_radius_static", "roll_steer_front", "camber_static",
]
# Reasonable per-metric tolerances before flagging a drift warning (same
# values as the front script for the metrics they share).
DRIFT_TOLERANCE = {
    "roll_centre_z_static": 5.0,
    "camber_gain_bump_per25mm": 0.1,
    "camber_gain_roll_pct": 0.05,
    "kpi_static": 0.5,
    "scrub_radius_static": 2.0,
    "roll_steer_front": 0.05,
    "camber_static": 0.5,
}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cad_geometry_to_axle_dict_rear(cad_geometry: dict, name: str = "axle") -> dict:
    """Same as _sweep_common.py's cad_geometry_to_axle_dict(), except
    axle_config.steering is "none" (toe link, not rack) - that shared
    function hardcodes "rack" unconditionally, which is wrong for the rear
    and can't be changed without touching _sweep_common.py. Hardpoint names
    (including toe_link_inboard/outboard, axle_inboard/outboard) pass
    through unchanged, same as the shared function."""
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
            "axle_position": "rear",
            "steering": {"type": "none"},
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


def build_final_geometry(stage_a_geometry: dict, stage_b: dict) -> dict:
    final = copy.deepcopy(stage_a_geometry)
    hp = final["hardpoints"]
    hp["upper_wishbone_outboard"] = sc.pt(stage_b["p_shifted"])
    hp["lower_wishbone_outboard"] = sc.pt(stage_b["q_shifted"])
    final["name"] = "UCT Racing FSA rear left - optimal (Stage A + B combined)"
    return final


def run_drift_check(final_metrics: dict, stage_a_best_row: pd.Series) -> pd.DataFrame:
    rows = []
    for metric in DRIFT_CHECK_METRICS:
        predicted = stage_a_best_row.get(metric)
        actual = final_metrics.get(metric)
        delta = (actual - predicted) if (predicted is not None and actual is not None) else None
        tol = DRIFT_TOLERANCE[metric]
        flagged = delta is not None and abs(delta) > tol
        rows.append({"metric": metric, "stage_a_predicted": predicted,
                      "stage_c_actual": actual, "delta": delta,
                      "tolerance": tol, "flagged": flagged})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--stage-b", type=Path, default=DEFAULT_STAGE_B)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage_a_geometry = load_yaml(args.stage_a)
    stage_b = load_yaml(args.stage_b)
    final_geometry = build_final_geometry(stage_a_geometry, stage_b)

    print(f"Loaded Stage A geometry: {args.stage_a}")
    print(f"Loaded Stage B refinement: {args.stage_b}")
    print(
        "\nNOTE: the final geometry relocates BOTH wishbones' outboard ball "
        "joints from CAD, not just the upper one - Stage B's rotate_axis() "
        "repositions both p and q along the newly-oriented kingpin axis.\n"
    )

    # ---- confirmatory re-solve on the combined final geometry -------------
    axle_dict = cad_geometry_to_axle_dict_rear(final_geometry, name="rear_optimal")
    metrics = rm.evaluate_candidate_rear(axle_dict)
    metrics["weighted_cost_full9"] = sc.weighted_cost(metrics, TARGETS_FULL_REAR, WEIGHTS_FULL_REAR)

    print("=" * 60)
    print("STAGE C (rear) - Confirmatory solve, full target scoring")
    print("=" * 60)
    checks = {}
    for name, (lo, hi, opt_lo, opt_hi) in TARGETS_FULL_REAR.items():
        value = metrics.get(name)
        checks[name] = sc.in_band(value, lo, hi)
        status = "PASS" if checks[name] else "FAIL"
        opt_str = f"{opt_lo}" if opt_lo == opt_hi else f"{opt_lo}-{opt_hi}"
        print(f"  {name:28s} {value:10.4f}  accept[{lo},{hi}] optimal[{opt_str}]  {status}")
    n_pass = sum(checks.values())
    print(f"\n{n_pass}/{len(checks)} targets pass. Weighted cost: "
          f"{metrics['weighted_cost_full9']:.3f}\n")

    # ---- drift check vs Stage A's predicted values -------------------------
    stage_a_df = pd.read_csv(STAGE_A_RESULTS_CSV)
    stage_a_best = stage_a_df[stage_a_df.feasible].sort_values("weighted_cost").iloc[0]
    drift_df = run_drift_check(metrics, stage_a_best)

    print("=" * 60)
    print("Drift check - rear Stage A prediction vs Stage C confirmatory re-solve")
    print("(Stage A scored these on the CAD lower_wishbone_outboard position;")
    print(" Stage B moved that point, so this checks nothing broke)")
    print("=" * 60)
    print(drift_df.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    n_flagged = int(drift_df["flagged"].sum())
    if n_flagged:
        print(f"\n{n_flagged} metric(s) drifted beyond tolerance - review before locking in.")
    else:
        print("\nAll metrics within tolerance - no significant drift from Stage A's prediction.")
    print()

    # ---- build MetricContext for the annotated wheel view -----------------
    from kinematics.core.input import build_suspension, build_sweep
    from kinematics.core.metrics.context import MetricContext
    from kinematics.core.primitives.point_ref import Side
    from kinematics.core.sweep import solve_evaluated_sweep

    suspension = build_suspension(axle_dict)
    sweep = build_sweep(rm.BUMP_SWEEP_REAR, suspension)
    ev = solve_evaluated_sweep(suspension, sweep)
    static_state = ev.states[len(ev.states) // 2]

    corner = suspension.corners[Side.LEFT]
    corner_state = suspension.corner_state(static_state, Side.LEFT)
    corner_config = corner.config if corner.config is not None else suspension.config
    ctx = MetricContext(state=corner_state, suspension=corner, config=corner_config)

    wheel_view_summary = wheel_view.render(
        ctx, OUTPUT_DIR / "wheel_view_rear.png",
        roll_centre_z=metrics.get("roll_centre_z_static"),
        weighted_cost=metrics.get("weighted_cost_full9"),
        corner_label="rear-left")

    # ---- 4-view linkage render, for continuity -----------------------------
    from kinematics.cli.visualization.api import visualize_geometry

    linkage_png = OUTPUT_DIR / "linkage_4view_rear.png"
    result = visualize_geometry(suspension, linkage_png)
    print(f"wrote {result.output_path}  (contact patch on ground: {result.contact_patch_on_ground})")

    # Same headline metrics as wheel_view_rear.png's text box, plus P/Q
    # coordinates (upper/lower wishbone outboard ball joints) - overlaid
    # onto the already-rendered PNG so kinematics.cli.visualization.api
    # stays untouched.
    p_xyz, q_xyz = wheel_view_summary["p_xyz"], wheel_view_summary["q_xyz"]
    linkage_lines = wheel_view.build_summary_lines(
        wheel_view_summary["kpi"], wheel_view_summary["caster"], wheel_view_summary["scrub"],
        wheel_view_summary["trail"], wheel_view_summary["camber"], wheel_view_summary["toe"],
        wheel_view_summary["roll_centre_z"], wheel_view_summary["weighted_cost"],
    ) + [
        f"P: ({p_xyz[0]:.1f}, {p_xyz[1]:.1f}, {p_xyz[2]:.1f}) mm",
        f"Q: ({q_xyz[0]:.1f}, {q_xyz[1]:.1f}, {q_xyz[2]:.1f}) mm",
    ]
    wheel_view.annotate_png(linkage_png, linkage_lines)

    # ---- outputs ------------------------------------------------------------
    geom_out = OUTPUT_DIR / "optimal_geometry_rear.yaml"
    with open(geom_out, "w") as f:
        yaml.dump(final_geometry, f, default_flow_style=False, sort_keys=False)
    print(f"wrote {geom_out}")

    summary_row = {
        "caster_deg": stage_b["caster_deg"],
        "offset_mm": stage_b["offset_mm"],
        "trail_mm_target": stage_b["trail_mm"],
        "p_shift_mm": stage_b["p_shift_mm"],
        "q_shift_mm": stage_b["q_shift_mm"],
        "z_up": float(stage_a_best["z_up"]),
        "kpi_deg": float(stage_a_best["kpi_deg"]),
        **metrics,
        "targets_passed": n_pass,
        "targets_total": len(checks),
        "drift_flags": n_flagged,
    }
    summary_df = pd.DataFrame([summary_row])
    summary_out = OUTPUT_DIR / "optimal_summary_rear.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"wrote {summary_out}")

    if STAGE_A_BEST_TRAVEL_PNG.exists():
        import shutil
        shutil.copy(STAGE_A_BEST_TRAVEL_PNG, OUTPUT_DIR / STAGE_A_BEST_TRAVEL_PNG.name)
        print(f"copied {STAGE_A_BEST_TRAVEL_PNG.name} into {OUTPUT_DIR}")

    caster = stage_b["caster_deg"]
    offset = stage_b["offset_mm"]
    trail = stage_b["trail_mm"]
    rec_lines = [
        "# RECOMMENDATION (rear)\n",
        f"- Mount height Z: {stage_a_best['z_up']:.0f} mm\n",
        f"- KPI angle: {stage_a_best['kpi_deg']:.1f} degrees\n",
        f"- Caster: {caster:.1f} degrees\n",
        f"- Axis offset: {abs(offset):.1f} mm (rearward)\n",
        f"- Mechanical trail: {trail:.1f} mm\n",
        f"- Upper ball joint (p) shifts {stage_b['p_shift_mm']:.1f} mm from current CAD position\n",
        f"- Lower ball joint (q) shifts {stage_b['q_shift_mm']:.1f} mm from current CAD position\n",
        "\n",
        "**Both wishbones' outboard ball joints move** - not just the upper "
        "one. The caster/trail refinement (Stage B) repositions both the "
        "upper and lower ball joints along the newly-oriented kingpin axis.\n",
        "\n",
        f"## Target check (against confirmatory re-solve, bump steer excluded - see module docstring)\n\n",
        f"{n_pass}/{len(checks)} targets pass. Weighted cost: "
        f"{metrics['weighted_cost_full9']:.3f}.\n",
        "\n",
        "## Drift check (rear Stage A prediction vs Stage C confirmatory re-solve)\n\n",
        drift_df.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n\n",
        (f"{n_flagged} metric(s) drifted beyond tolerance - review before locking in.\n"
         if n_flagged else
         "All metrics within tolerance - no significant drift from Stage A's prediction.\n"),
        "\n",
        "CHECK: does this shift fit inside the upright and clear the wheel rim?\n",
        "Take these numbers to the CAD model before locking anything in. "
        "Roll centre and motion ratio have been re-checked by this stage's "
        "confirmatory solve. Bump steer is not scored for the rear (it's a "
        "placeholder-toe-link artifact, not real geometry - see "
        "scripts/rear_mount_kpi_sweep.py's module docstring), and axle_inboard/"
        "axle_outboard/toe_link_inboard/toe_link_outboard are all still "
        "PLACEHOLDERS until the real upright/toe-link is measured, per the "
        "note in scripts/rear_caster_trail_analysis.py.\n",
    ]
    rec_out = OUTPUT_DIR / "RECOMMENDATION_rear.md"
    with open(rec_out, "w") as f:
        f.writelines(rec_lines)
    print(f"wrote {rec_out}")

    print("\nRECOMMENDATION (rear)")
    print(f"Caster:           {caster:.1f} degrees")
    print(f"Axis offset:      {abs(offset):.1f} mm (rearward)")
    print(f"Mechanical trail: {trail:.1f} mm")
    print(f"Upper ball joint (p) shifts {stage_b['p_shift_mm']:.1f} mm from current CAD position")
    print(f"Lower ball joint (q) shifts {stage_b['q_shift_mm']:.1f} mm from current CAD position")


if __name__ == "__main__":
    main()
