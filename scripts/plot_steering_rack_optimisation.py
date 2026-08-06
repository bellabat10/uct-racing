"""
Plots for scripts/steering_rack_optimisation.py.

Reads outputs/steering_rack_optimisation/summary.csv (+ the best_candidate/
sweep CSVs and geometry.yaml the main script also writes) and produces:

  outputs/steering_rack_optimisation_summary.png    - 6-panel candidate sweep
  outputs/best_candidate_bump_steer.png
  outputs/best_candidate_roll_steer.png
  outputs/best_candidate_ackermann.png
  outputs/best_candidate_front_view.png

Run scripts/steering_rack_optimisation.py first.

Usage:
    python scripts/plot_steering_rack_optimisation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import steering_rack_optimisation as m  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "outputs" / "steering_rack_optimisation"
OUT_DIR = REPO_ROOT / "outputs"

# Palette (colour-blind-safe categorical set + green/amber/red status bands)
C_PRIMARY = "#3B82C4"
C_SECONDARY = "#D9782D"
C_TERTIARY = "#5FA05C"
C_GOOD = "#2E7D46"
C_WARN = "#B8860B"
GOOD_BAND = "#2E7D4622"
WARN_BAND = "#B8860B22"
C_RECOMMENDED = "#7A4FA3"
C_INFEASIBLE = "#B0B0B0"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(RUN_DIR / "summary.csv")
    return df


def recommended_rack_x(df: pd.DataFrame) -> float:
    feasible = df[df["feasible"]]
    return float(feasible.loc[feasible["score"].idxmin(), "rack_x_iso"])


# ===========================================================================
# 6-panel summary figure
# ===========================================================================

def plot_summary(df: pd.DataFrame, out_path: Path) -> None:
    rec_x = recommended_rack_x(df)
    feasible = df[df["feasible"]].sort_values("rack_x_iso")
    infeasible = df[~df["feasible"]]

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle("Steering rack / tie-rod optimisation - rack-X candidate sweep",
                 fontsize=13, fontweight="bold")
    x = feasible["rack_x_iso"]

    def mark_infeasible(ax):
        for xv in infeasible["rack_x_iso"]:
            ax.axvspan(xv - 2.5, xv + 2.5, color=C_INFEASIBLE, alpha=0.35, zorder=0)

    def mark_recommended(ax):
        ax.axvline(rec_x, color=C_RECOMMENDED, linestyle="--", linewidth=1.6, zorder=5,
                   label=f"recommended ({rec_x:.0f}mm)")

    # Panel 1: bump steer
    ax = axes[0, 0]
    ax.axhspan(0, m.BUMP_STEER_TARGET, color=C_GOOD, alpha=0.15)
    ax.axhspan(m.BUMP_STEER_TARGET, m.BUMP_STEER_ACCEPT, color=C_WARN, alpha=0.12)
    ax.plot(x, feasible["bump_steer_max_deg"], "o-", color=C_PRIMARY, label="max")
    ax.plot(x, feasible["bump_steer_rms_deg"], "s--", color=C_SECONDARY, label="rms", markersize=4)
    mark_infeasible(ax)
    mark_recommended(ax)
    ax.set_title("Bump steer vs rack X")
    ax.set_ylabel("toe change, deg")
    ax.legend(fontsize=8, loc="best")
    _style_axes(ax)

    # Panel 2: roll steer
    ax = axes[0, 1]
    ax.axhspan(0, m.ROLL_STEER_TARGET, color=C_GOOD, alpha=0.15)
    ax.axhspan(m.ROLL_STEER_TARGET, m.ROLL_STEER_ACCEPT, color=C_WARN, alpha=0.12)
    ax.plot(x, feasible["roll_steer_max_deg"], "o-", color=C_PRIMARY, label="max")
    ax.plot(x, feasible["roll_steer_at_1deg_roll"].abs(), "s--", color=C_SECONDARY,
            label="|at 1deg roll|", markersize=4)
    mark_infeasible(ax)
    mark_recommended(ax)
    ax.set_title("Roll steer vs rack X")
    ax.set_ylabel("toe change, deg")
    ax.legend(fontsize=8, loc="best")
    _style_axes(ax)

    # Panel 3: Ackermann
    ax = axes[1, 0]
    ax.axhspan(m.ACKERMANN_LO, m.ACKERMANN_HI, color=C_GOOD, alpha=0.15)
    ax.axhline(m.ACKERMANN_NOMINAL, color=C_GOOD, linestyle=":", linewidth=1.2)
    ax.plot(x, feasible["ackermann_pct"], "o-", color=C_PRIMARY, label="geometric (quick)")
    ax.plot(x, feasible["ackermann_pct_steer_validated"], "s--", color=C_SECONDARY,
            label="steer-sweep-validated", markersize=4)
    mark_infeasible(ax)
    mark_recommended(ax)
    ax.set_title("Ackermann % vs rack X")
    ax.set_ylabel("Ackermann, %")
    ax.legend(fontsize=8, loc="best")
    _style_axes(ax)

    # Panel 4: moment arm
    ax = axes[1, 1]
    ax.axhspan(m.MOMENT_ARM_LO, m.MOMENT_ARM_HI, color=C_GOOD, alpha=0.15)
    ax.plot(x, feasible["moment_arm_mm"], "o-", color=C_PRIMARY)
    mark_infeasible(ax)
    mark_recommended(ax)
    ax.set_title("Steering moment arm vs rack X")
    ax.set_ylabel("moment arm, mm")
    _style_axes(ax)

    # Panel 5: tie rod geometry (twin axes)
    ax = axes[2, 0]
    l1, = ax.plot(x, feasible["tie_rod_length"], "o-", color=C_PRIMARY, label="length")
    ax.set_ylabel("tie rod length, mm", color=C_PRIMARY)
    ax.tick_params(axis="y", labelcolor=C_PRIMARY)
    ax2 = ax.twinx()
    l2, = ax2.plot(x, feasible["elevation_angle_deg"], "s--", color=C_SECONDARY,
                    label="elevation angle", markersize=4)
    ax2.set_ylabel("elevation angle, deg", color=C_SECONDARY)
    ax2.tick_params(axis="y", labelcolor=C_SECONDARY)
    mark_infeasible(ax)
    mark_recommended(ax)
    ax.set_title("Tie rod geometry vs rack X")
    ax.legend([l1, l2], ["length", "elevation angle"], fontsize=8, loc="best")
    _style_axes(ax)
    ax2.grid(False)

    # Panel 6: IC alignment quality
    ax = axes[2, 1]
    ax.plot(x, feasible["fvic_alignment_error_mm"], "o-", color=C_PRIMARY, label="FVIC error", markersize=4)
    ax.plot(x, feasible["svic_alignment_error_mm"], "s-", color=C_SECONDARY, label="SVIC error", markersize=4)
    ax.plot(x, feasible["z_consistency_error_mm"].abs(), "^-", color=C_TERTIARY,
            label="|z consistency error|", markersize=4)
    mark_infeasible(ax)
    mark_recommended(ax)
    ax.set_title("IC alignment quality vs rack X")
    ax.set_ylabel("error, mm")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.legend(fontsize=8, loc="best")
    _style_axes(ax)

    for ax in axes[-1, :]:
        ax.set_xlabel("rack X (ISO), mm")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ===========================================================================
# Best-candidate detail plots
# ===========================================================================

def plot_bump_steer(df_bump: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    travel = df_bump["wheel_travel_left"]
    toe = df_bump["roadwheel_angle_left"]
    mid_toe = toe.iloc[len(toe) // 2]
    toe_rel = toe - mid_toe
    ax.axhspan(-m.BUMP_STEER_TARGET, m.BUMP_STEER_TARGET, color=C_GOOD, alpha=0.15, label="target")
    ax.axhspan(-m.BUMP_STEER_ACCEPT, -m.BUMP_STEER_TARGET, color=C_WARN, alpha=0.12)
    ax.axhspan(m.BUMP_STEER_TARGET, m.BUMP_STEER_ACCEPT, color=C_WARN, alpha=0.12, label="acceptable")
    ax.plot(travel, toe_rel, "o-", color=C_PRIMARY, markersize=3)
    ax.axvline(0, color="grey", linewidth=0.8)
    _style_axes(ax)
    ax.set_xlabel("wheel travel, mm (+ bump / - droop)")
    ax.set_ylabel("toe change from static, deg")
    ax.set_title("Best candidate - bump steer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_roll_steer(df_roll: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    roll = df_roll["roll"]
    toe = df_roll["roadwheel_angle_left"]
    mid_toe = toe.iloc[len(toe) // 2]
    toe_rel = toe - mid_toe
    ax.axhspan(-m.ROLL_STEER_TARGET, m.ROLL_STEER_TARGET, color=C_GOOD, alpha=0.15, label="target")
    ax.axhspan(-m.ROLL_STEER_ACCEPT, -m.ROLL_STEER_TARGET, color=C_WARN, alpha=0.12)
    ax.axhspan(m.ROLL_STEER_TARGET, m.ROLL_STEER_ACCEPT, color=C_WARN, alpha=0.12, label="acceptable")
    ax.plot(roll, toe_rel, "o-", color=C_PRIMARY, markersize=3)
    ax.axvline(0, color="grey", linewidth=0.8)
    _style_axes(ax)
    ax.set_xlabel("chassis roll, deg")
    ax.set_ylabel("toe change from static, deg")
    ax.set_title("Best candidate - roll steer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_ackermann(df_steer: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    outer = df_steer["roadwheel_angle_right"]   # right = outer for a left-hand-lock convention
    inner = -df_steer["roadwheel_angle_left"]
    order = np.argsort(outer.to_numpy())
    outer_s, inner_s = outer.to_numpy()[order], inner.to_numpy()[order]

    ax.plot(outer_s, inner_s, "o-", color=C_PRIMARY, markersize=3, label="actual (solver)")
    ax.plot(outer_s, outer_s, "--", color="grey", linewidth=1.2, label="parallel steer (0%)")

    track_mm, wheelbase_mm = m.TRACK_MM, m.WHEELBASE_MM
    outer_rad = np.radians(np.abs(outer_s))
    with np.errstate(divide="ignore", invalid="ignore"):
        ideal_inner = np.degrees(np.arctan(
            1.0 / (1.0 / np.tan(np.where(outer_rad == 0, np.nan, outer_rad)) - track_mm / wheelbase_mm)))
    ax.plot(outer_s, np.sign(outer_s) * np.abs(ideal_inner), ":", color=C_GOOD, linewidth=1.4,
            label="100% Ackermann")

    _style_axes(ax)
    ax.set_xlabel("outer wheel steer angle, deg")
    ax.set_ylabel("inner wheel steer angle, deg")
    ax.set_title("Best candidate - Ackermann (steer sweep)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_front_view(geometry_cad: dict, detail: dict, out_path: Path) -> None:
    hp = geometry_cad["hardpoints"]

    def yz(name):
        p = hp[name]
        return p["y"], p["z"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Wishbones (front view: Y-Z plane)
    lwo = yz("lower_wishbone_outboard")
    uwo = yz("upper_wishbone_outboard")
    for inb_name, outb, color, label in [
        ("lower_wishbone_inboard_front", lwo, C_PRIMARY, "lower wishbone"),
        ("lower_wishbone_inboard_rear", lwo, C_PRIMARY, None),
        ("upper_wishbone_inboard_front", uwo, C_SECONDARY, "upper wishbone"),
        ("upper_wishbone_inboard_rear", uwo, C_SECONDARY, None),
    ]:
        inb = yz(inb_name)
        ax.plot([inb[0], outb[0]], [inb[1], outb[1]], "-", color=color, linewidth=2, label=label)

    # FVIC and dashed extensions of wishbone lines through it
    ics = detail["ics"]
    fvic = (ics["fvic_y"], ics["fvic_z"])
    for inb_name, outb, color in [
        ("lower_wishbone_inboard_front", lwo, C_PRIMARY),
        ("upper_wishbone_inboard_front", uwo, C_SECONDARY),
    ]:
        inb = yz(inb_name)
        ax.plot([fvic[0], inb[0]], [fvic[1], inb[1]], "--", color=color, linewidth=1, alpha=0.6)
    ax.plot(*fvic, "*", color="black", markersize=14, label="FVIC", zorder=6)

    # Tie rod + FVIC line overlay
    trackrod_in = yz("trackrod_inboard")
    trackrod_out = yz("trackrod_outboard")
    ax.plot([trackrod_in[0], trackrod_out[0]], [trackrod_in[1], trackrod_out[1]],
            "-", color=C_TERTIARY, linewidth=2.5, label="tie rod", zorder=5)
    # FVIC line overlaid on tie rod (extend from FVIC through inboard, well past outboard)
    dv = np.array(trackrod_in) - np.array(fvic)
    far = np.array(fvic) + dv * 1.15
    ax.plot([fvic[0], far[0]], [fvic[1], far[1]], ":", color="black", linewidth=1.2,
            label="FVIC line (should coincide with tie rod)", zorder=4)

    # Wheel outline (circle at wheel centre, front view)
    axle_in = yz("axle_inboard")
    axle_out = yz("axle_outboard")
    wheel_y = 0.5 * (axle_in[0] + axle_out[0])
    wheel_z = 0.5 * (axle_in[1] + axle_out[1])
    theta = np.linspace(0, 2 * np.pi, 100)
    tire = geometry_cad["config"]["wheel"]["tire"]
    free_radius = (tire["rim_diameter"] * 25.4 / 2.0) + tire["section_width"] * tire["aspect_ratio"]
    ax.plot(wheel_y + free_radius * np.cos(theta), wheel_z + free_radius * np.sin(theta),
            "-", color="grey", linewidth=1, label="wheel")

    # Ground line
    ax.axhline(0, color="black", linewidth=1)

    ax.plot(*trackrod_in, "o", color=C_TERTIARY, markersize=6)
    ax.plot(*trackrod_out, "o", color=C_TERTIARY, markersize=6)

    ax.set_xlabel("y (ISO, left), mm")
    ax.set_ylabel("z (ISO, up), mm")
    ax.set_title("Best candidate - front view (FVIC / tie-rod alignment)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, loc="upper left")
    _style_axes(ax)
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    df = load_summary()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_summary(df, OUT_DIR / "steering_rack_optimisation_summary.png")

    detail_dir = RUN_DIR / "best_candidate"
    bump = pd.read_csv(detail_dir / "bump_sweep.csv")
    roll = pd.read_csv(detail_dir / "roll_sweep.csv")
    steer = pd.read_csv(detail_dir / "steer_sweep.csv")
    with open(detail_dir / "geometry.yaml") as f:
        geometry_cad = yaml.safe_load(f)

    plot_bump_steer(bump, OUT_DIR / "best_candidate_bump_steer.png")
    plot_roll_steer(roll, OUT_DIR / "best_candidate_roll_steer.png")
    plot_ackermann(steer, OUT_DIR / "best_candidate_ackermann.png")

    baseline_cad, _ = m.load_baseline()
    ics = m.extract_static_ics(baseline_cad)
    plot_front_view(geometry_cad, {"ics": ics}, OUT_DIR / "best_candidate_front_view.png")


if __name__ == "__main__":
    main()
