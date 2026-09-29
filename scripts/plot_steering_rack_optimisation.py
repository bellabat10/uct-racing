"""
Plots for scripts/steering_rack_optimisation.py.

Reads a run folder (default outputs/front/steering_rack_optimisation/):
summary.csv plus best_candidate/bump_sweep.csv and roll_sweep.csv, and writes
into the same folder:

  summary.png                      - bump steer, roll steer slope, peak toe
                                     change in roll, max steer angle and
                                     steering ratio vs steering arm inset,
                                     target bands shaded
  best_candidate/bump_steer.png    - recommended candidate's toe vs wheel travel
  best_candidate/roll_steer.png    - recommended candidate's toe vs chassis roll

Target values come from scripts/steering_rack_optimisation.py (which reads
them from _sweep_common.TARGETS_FULL), so the shaded bands always match the
PASS/FAIL rule in summary.csv.

Run scripts/steering_rack_optimisation.py first.

Usage:
    python scripts/plot_steering_rack_optimisation.py
    python scripts/plot_steering_rack_optimisation.py --run-dir outputs/front/steering_rack_optimisation/smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import steering_rack_optimisation as m  # noqa: E402

C_SERIES = "#2a78d6"        # the one data series per panel
C_TARGET_BAND = "#1baf7a"   # target / acceptable region
C_REJECTED = "#9a9892"      # infeasible or hard-rejected candidates
C_RECOMMENDED = "#0b0b0b"
C_TEXT_MUTED = "#52514e"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def load_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary.csv"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found - run scripts/steering_rack_optimisation.py first.")
    df = pd.read_csv(path)
    df["feasible"] = df["feasible"].astype(bool)
    return df.sort_values("arm_inset_mm").reset_index(drop=True)


def recommended_row(df: pd.DataFrame) -> pd.Series | None:
    """Same rule as the optimiser: lowest cost among scored candidates
    (cost is NaN for infeasible and hard-rejected ones)."""
    scored = df[df["cost"].notna()]
    return None if scored.empty else scored.loc[scored["cost"].idxmin()]


def _travel_label(feasible: pd.DataFrame, col: str) -> str:
    """The travel actually solved, e.g. '25', or '10-25' if some candidates
    fell back to less travel."""
    values = feasible[col].dropna().unique()
    if len(values) == 0:
        return "?"
    if len(values) == 1:
        return f"{values[0]:g}"
    return f"{min(values):g}-{max(values):g}"


# ===========================================================================
# Summary figure
# ===========================================================================

def plot_summary(df: pd.DataFrame, out_path: Path) -> None:
    feasible = df[df["feasible"]]
    rejected_x = df.loc[~df["feasible"] | (df["feasible"] & (df["max_steer_angle_deg"] < m.MIN_STEER_ANGLE_DEG)),
                        "arm_inset_mm"]
    best = recommended_row(df)
    x = feasible["arm_inset_mm"]

    # Grey band per rejected candidate: half the candidate spacing (so a 5mm
    # sweep tiles exactly), capped so a sparse --smoke run doesn't grey out
    # whole stretches of the axis that were never evaluated.
    steps = np.diff(df["arm_inset_mm"].to_numpy())
    half_width = min(0.5 * float(np.min(steps)), 5.0) if len(steps) else 2.5

    # Top axis: true steering arm length at the same candidates (monotonic in
    # inset). Linear extrapolation past the ends so the axis doesn't clamp.
    inset_all = df["arm_inset_mm"].to_numpy(dtype=float)
    length_all = df["steering_arm_length_mm"].to_numpy(dtype=float)

    def _interp_extrap(v, xp, fp):
        v = np.asarray(v, dtype=float)
        out = np.interp(v, xp, fp)
        if len(xp) >= 2:
            lo_slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
            hi_slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            out = np.where(v < xp[0], fp[0] + (v - xp[0]) * lo_slope, out)
            out = np.where(v > xp[-1], fp[-1] + (v - xp[-1]) * hi_slope, out)
        return out

    def inset_to_length(v):
        return _interp_extrap(v, inset_all, length_all)

    def length_to_inset(v):
        return _interp_extrap(v, length_all, inset_all)

    bump_label = _travel_label(feasible, "bump_travel_used_mm")
    droop_label = _travel_label(feasible, "droop_travel_used_mm")
    roll_label = f"{feasible['roll_max_deg'].max():.1f}" if not feasible.empty else "?"

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), sharex=True)
    offset = float(df["steering_arm_rearward_offset_mm"].iloc[0])
    placeholder = " (PLACEHOLDER)" if m.RACK_HALF_STROKE_IS_PLACEHOLDER else ""
    fig.suptitle(
        "Steering rack optimisation - candidates vs steering arm inset\n"
        f"rack z = {df['trackrod_inboard_z'].dropna().iloc[0]:.1f} mm, "
        f"rearward offset = {offset:.0f} mm, rack half-stroke = {m.RACK_HALF_STROKE_MM:.0f} mm{placeholder}",
        fontsize=12)

    def decorate(ax, title, ylabel):
        for xv in rejected_x:
            ax.axvspan(xv - half_width, xv + half_width, color=C_REJECTED, alpha=0.18, lw=0, zorder=0)
        if best is not None:
            ax.axvline(best["arm_inset_mm"], color=C_RECOMMENDED, linestyle="--", linewidth=1.2, zorder=5)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_ylabel(ylabel)
        _style_axes(ax)

    def plot_series(ax, col):
        ax.plot(x, feasible[col], "o-", color=C_SERIES, linewidth=2, markersize=4, zorder=4)

    # Bump steer
    ax = axes[0, 0]
    ax.axhspan(0, m.BUMP_STEER_TARGET, color=C_TARGET_BAND, alpha=0.18, lw=0,
               label=f"target <= {m.BUMP_STEER_TARGET:g} deg")
    plot_series(ax, "bump_steer_deg")
    decorate(ax, f"Bump steer (toe range, -{droop_label} / +{bump_label} mm travel)", "deg")
    ax.set_ylim(bottom=0)

    # Roll steer slope (scored)
    ax = axes[0, 1]
    ax.axhspan(-m.ROLL_STEER_TARGET, m.ROLL_STEER_TARGET, color=C_TARGET_BAND, alpha=0.18, lw=0,
               label=f"target +/-{m.ROLL_STEER_TARGET:g} deg/deg")
    ax.axhline(0, color=C_TEXT_MUTED, linewidth=0.8)
    plot_series(ax, "roll_steer_slope_deg_per_deg")
    decorate(ax, "Roll steer slope (scored)", "deg toe / deg roll")

    # Peak toe change in roll (reported only)
    ax = axes[0, 2]
    ax.axhline(0, color=C_TEXT_MUTED, linewidth=0.8)
    plot_series(ax, "roll_toe_change_peak_deg")
    decorate(ax, f"Peak toe change in roll, +/-{roll_label} deg (reported only)", "deg")

    # Max steer angle
    ax = axes[1, 0]
    plot_series(ax, "max_steer_angle_deg")
    top = max(float(feasible["max_steer_angle_deg"].max()) * 1.08, m.MIN_STEER_ANGLE_DEG * 1.2)
    ax.axhspan(m.MIN_STEER_ANGLE_DEG, top, color=C_TARGET_BAND, alpha=0.18, lw=0,
               label=f">= {m.MIN_STEER_ANGLE_DEG:g} deg required")
    ax.set_ylim(0, top)
    decorate(ax, f"Max steer angle = ratio x {m.RACK_HALF_STROKE_MM:g} mm half-stroke{placeholder.lower()}",
             "deg")

    # Steering ratio - reported only, no target
    ax = axes[1, 1]
    plot_series(ax, "steering_ratio_deg_per_mm")
    decorate(ax, "Steering ratio (reported only)", "deg / mm of rack")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=8, loc="best", frameon=False)

    for ax in axes[1, :2]:
        ax.set_xlabel("steering arm inset, mm (lateral, inboard of kingpin axis)")
    axes[0, 2].tick_params(labelbottom=True)
    axes[0, 2].set_xlabel("steering arm inset, mm")
    for ax in axes[0, :]:
        sec = ax.secondary_xaxis("top", functions=(inset_to_length, length_to_inset))
        sec.set_xlabel("true steering arm length, mm", fontsize=9, color=C_TEXT_MUTED)
        sec.tick_params(labelsize=8, colors=C_TEXT_MUTED)

    # Sixth panel: shared legend + the recommended candidate's numbers.
    ax = axes[1, 2]
    ax.axis("off")
    handles = [Patch(color=C_TARGET_BAND, alpha=0.35, label="target / required region"),
               Patch(color=C_REJECTED, alpha=0.35,
                     label=f"infeasible or < {m.MIN_STEER_ANGLE_DEG:g} deg (not scored)")]
    if best is not None:
        handles.append(Line2D([], [], color=C_RECOMMENDED, linestyle="--", label="recommended candidate"))
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False)
    if best is not None:
        lines = [
            f"Recommended: inset {best['arm_inset_mm']:.0f} mm  ({best['pass_fail']})",
            f"true steering arm length   {best['steering_arm_length_mm']:.1f} mm",
            f"bump steer                 {best['bump_steer_deg']:.3f} deg",
            f"roll steer slope           {best['roll_steer_slope_deg_per_deg']:+.4f} deg/deg",
            f"peak toe change in roll    {best['roll_toe_change_peak_deg']:+.3f} deg",
            f"max steer angle            {best['max_steer_angle_deg']:.1f} deg",
            f"steering ratio             {best['steering_ratio_deg_per_mm']:.3f} deg/mm",
        ]
    else:
        lines = ["No scored candidate - every one was infeasible or under",
                 f"{m.MIN_STEER_ANGLE_DEG:g} deg max steer angle."]
    ax.text(0.0, 0.62, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, family="monospace", color=C_RECOMMENDED)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ===========================================================================
# Best-candidate detail plots
# ===========================================================================

def plot_bump_steer(df_bump: pd.DataFrame, best: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    travel = df_bump["wheel_travel_left"].to_numpy(dtype=float)
    toe = df_bump["roadwheel_angle_left"].to_numpy(dtype=float)
    toe_rel = toe - toe[int(np.argmin(np.abs(travel)))]
    ax.plot(travel, toe_rel, "o-", color=C_SERIES, linewidth=2, markersize=3)
    ax.axvline(0, color=C_TEXT_MUTED, linewidth=0.8)
    ax.axhline(0, color=C_TEXT_MUTED, linewidth=0.8)
    _style_axes(ax)
    ax.set_xlabel("wheel travel, mm (+ bump / - droop)")
    ax.set_ylabel("toe change from static, deg")
    ax.set_title(f"Recommended candidate (inset {best['arm_inset_mm']:.0f} mm) - bump steer\n"
                 f"toe range {best['bump_steer_deg']:.3f} deg, target <= {m.BUMP_STEER_TARGET:g} deg",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_roll_steer(df_roll: pd.DataFrame, best: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    roll = df_roll["roll"].to_numpy(dtype=float)
    toe = df_roll["roadwheel_angle_left"].to_numpy(dtype=float)
    toe_rel = toe - toe[int(np.argmin(np.abs(roll)))]
    order = np.argsort(roll)
    r = roll[order]
    ax.fill_between(r, -m.ROLL_STEER_TARGET * np.abs(r), m.ROLL_STEER_TARGET * np.abs(r),
                    color=C_TARGET_BAND, alpha=0.18, lw=0,
                    label=f"slope within +/-{m.ROLL_STEER_TARGET:g} deg/deg")
    ax.plot(roll, toe_rel, "o-", color=C_SERIES, linewidth=2, markersize=3)
    ax.axvline(0, color=C_TEXT_MUTED, linewidth=0.8)
    ax.axhline(0, color=C_TEXT_MUTED, linewidth=0.8)
    _style_axes(ax)
    ax.set_xlabel("chassis roll, deg")
    ax.set_ylabel("toe change from static, deg")
    ax.set_title(f"Recommended candidate (inset {best['arm_inset_mm']:.0f} mm) - roll steer\n"
                 f"slope {best['roll_steer_slope_deg_per_deg']:+.4f} deg/deg, "
                 f"peak toe change {best['roll_toe_change_peak_deg']:+.3f} deg", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=m.DEFAULT_OUTPUT_DIR,
                        help="folder holding summary.csv and best_candidate/ "
                             f"(default {m.DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    df = load_summary(args.run_dir)
    plot_summary(df, args.run_dir / "summary.png")

    best = recommended_row(df)
    detail_dir = args.run_dir / "best_candidate"
    if best is None or not (detail_dir / "bump_sweep.csv").exists():
        print("no recommended candidate / best_candidate sweeps - skipping detail plots")
        return
    plot_bump_steer(pd.read_csv(detail_dir / "bump_sweep.csv"), best, detail_dir / "bump_steer.png")
    plot_roll_steer(pd.read_csv(detail_dir / "roll_sweep.csv"), best, detail_dir / "roll_steer.png")


if __name__ == "__main__":
    main()
