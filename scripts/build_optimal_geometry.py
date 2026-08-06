"""
Stage C - assemble the final recommended geometry from Stage A (mount
height / KPI sweep) and Stage B (caster / mechanical trail refinement),
re-solve it once to confirm nothing drifted, render the annotated wheel
view, and write everything into outputs/optimal/.

Both wishbones' outboard ball joints move in the final geometry - Stage B's
existing rotate_axis() repositions BOTH p (upper) and q (lower) along the
newly-oriented kingpin axis, not just the upper one. Stage A's roll-centre
and camber numbers were computed against the CAD lower_wishbone_outboard
position, and Stage B silently moves that same point - this is exactly why
this stage's confirmatory re-solve matters.

Usage:
    python scripts/build_optimal_geometry.py
    python scripts/build_optimal_geometry.py --stage-a outputs/mount_height_kpi_sweep/best_zkpi_geometry.yaml --stage-b outputs/optimal/caster_and_trail/best_caster_offset.yaml
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
import wheel_view  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "optimal"

DEFAULT_STAGE_A = REPO_ROOT / "outputs" / "mount_height_kpi_sweep" / "best_zkpi_geometry.yaml"
DEFAULT_STAGE_B = OUTPUT_DIR / "caster_and_trail" / "best_caster_offset.yaml"
STAGE_A_RESULTS_CSV = REPO_ROOT / "outputs" / "mount_height_kpi_sweep" / "mount_kpi_results.csv"
STAGE_A_BEST_TRAVEL_PNG = REPO_ROOT / "outputs" / "mount_height_kpi_sweep" / "mount_kpi_best_vs_travel.png"

# The 7 metrics Stage A actually scored/searched over - the drift check
# compares Stage A's predicted value for these against what this stage's
# confirmatory re-solve produces on the FINAL (caster/offset-shifted) geometry.
DRIFT_CHECK_METRICS = [
    "bump_steer_max", "camber_gain_roll_pct", "roll_centre_z_static",
    "kpi_static", "scrub_radius_static", "roll_steer_front",
    "camber_gain_bump_per25mm",
]
# Reasonable per-metric tolerances before flagging a drift warning.
DRIFT_TOLERANCE = {
    "bump_steer_max": 0.5,
    "camber_gain_roll_pct": 0.05,
    "roll_centre_z_static": 5.0,
    "kpi_static": 0.5,
    "scrub_radius_static": 2.0,
    "roll_steer_front": 0.05,
    "camber_gain_bump_per25mm": 0.1,
}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_final_geometry(stage_a_geometry: dict, stage_b: dict) -> dict:
    final = copy.deepcopy(stage_a_geometry)
    hp = final["hardpoints"]
    hp["upper_wishbone_outboard"] = sc.pt(stage_b["p_shifted"])
    hp["lower_wishbone_outboard"] = sc.pt(stage_b["q_shifted"])
    final["name"] = "UCT Racing FSA front left - optimal (Stage A + B combined)"
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
    axle_dict = sc.cad_geometry_to_axle_dict(final_geometry, name="optimal")
    metrics = sc.evaluate_candidate(axle_dict)
    metrics["weighted_cost_full9"] = sc.weighted_cost(metrics, sc.TARGETS_FULL, sc.WEIGHTS_FULL)

    print("=" * 60)
    print("STAGE C - Confirmatory solve, full 9-target scoring")
    print("=" * 60)
    checks = {}
    for name, (lo, hi, opt_lo, opt_hi) in sc.TARGETS_FULL.items():
        value = metrics.get(name)
        checks[name] = sc.in_band(value, lo, hi)
        status = "PASS" if checks[name] else "FAIL"
        opt_str = f"{opt_lo}" if opt_lo == opt_hi else f"{opt_lo}-{opt_hi}"
        print(f"  {name:28s} {value:10.4f}  accept[{lo},{hi}] optimal[{opt_str}]  {status}")
    n_pass = sum(checks.values())
    print(f"\n{n_pass}/{len(checks)} targets pass. Weighted cost (full 9): "
          f"{metrics['weighted_cost_full9']:.3f}\n")

    # ---- drift check vs Stage A's predicted values -------------------------
    stage_a_df = pd.read_csv(STAGE_A_RESULTS_CSV)
    stage_a_best = stage_a_df[stage_a_df.feasible].sort_values("weighted_cost").iloc[0]
    drift_df = run_drift_check(metrics, stage_a_best)

    print("=" * 60)
    print("Drift check - Stage A prediction vs Stage C confirmatory re-solve")
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
    sweep = build_sweep(sc.BUMP_SWEEP, suspension)
    ev = solve_evaluated_sweep(suspension, sweep)
    static_state = ev.states[len(ev.states) // 2]

    corner = suspension.corners[Side.LEFT]
    corner_state = suspension.corner_state(static_state, Side.LEFT)
    corner_config = corner.config if corner.config is not None else suspension.config
    ctx = MetricContext(state=corner_state, suspension=corner, config=corner_config)

    wheel_view.render(ctx, OUTPUT_DIR / "wheel_view.png")

    # ---- 4-view linkage render, for continuity -----------------------------
    from kinematics.cli.visualization.api import visualize_geometry

    result = visualize_geometry(suspension, OUTPUT_DIR / "linkage_4view.png")
    print(f"wrote {result.output_path}  (contact patch on ground: {result.contact_patch_on_ground})")

    # ---- outputs ------------------------------------------------------------
    geom_out = OUTPUT_DIR / "optimal_geometry.yaml"
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
    summary_out = OUTPUT_DIR / "optimal_summary.csv"
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
        "# RECOMMENDATION\n",
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
        f"## Target check (full 9, against confirmatory re-solve)\n\n",
        f"{n_pass}/{len(checks)} targets pass. Weighted cost: "
        f"{metrics['weighted_cost_full9']:.3f}.\n",
        "\n",
        "## Drift check (Stage A prediction vs Stage C confirmatory re-solve)\n\n",
        drift_df.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n\n",
        (f"{n_flagged} metric(s) drifted beyond tolerance - review before locking in.\n"
         if n_flagged else
         "All metrics within tolerance - no significant drift from Stage A's prediction.\n"),
        "\n",
        "CHECK: does this shift fit inside the upright and clear the wheel rim?\n",
        "Take these numbers to the CAD model before locking anything in. "
        "Roll centre, bump steer and motion ratio have been re-checked by "
        "this stage's confirmatory solve, but the full optimiser (all three "
        "stages together) should be re-run once hardpoints actually move in "
        "CAD, per the note in scripts/caster_trail_analysis.py.\n",
    ]
    rec_out = OUTPUT_DIR / "RECOMMENDATION.md"
    with open(rec_out, "w") as f:
        f.writelines(rec_lines)
    print(f"wrote {rec_out}")

    print("\nRECOMMENDATION")
    print(f"Caster:           {caster:.1f} degrees")
    print(f"Axis offset:      {abs(offset):.1f} mm (rearward)")
    print(f"Mechanical trail: {trail:.1f} mm")
    print(f"Upper ball joint (p) shifts {stage_b['p_shift_mm']:.1f} mm from current CAD position")
    print(f"Lower ball joint (q) shifts {stage_b['q_shift_mm']:.1f} mm from current CAD position")


if __name__ == "__main__":
    main()
