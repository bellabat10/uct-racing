"""
Caster / mechanical trail analysis - UCT Racing rear-left corner. Cloned
from scripts/caster_trail_analysis.py (front) - the geometry/scoring logic
is IDENTICAL, only the default geometry source and output directory differ
(see below). Read the front script first if this one is confusing.

The rear corner doesn't steer, but it still has a kingpin axis (p, q) with a
caster angle and a ground-intersection point ahead of or behind the contact
patch - i.e. it still has mechanical trail in the same geometric sense the
front does. That trail (and the caster angle that sets it) affects anti-dive/
anti-squat and how the rear tyre reacts to longitudinal force (traction,
braking), even with no steering rack involved - this is exactly why Stage B
runs for the rear too, not just the front.

Three sequential stages:
  1. Baseline caster sweep (0-7 deg, axis rotated through the wheel centre,
     zero x-offset) -> mechanical trail and acceptance status.
  2. For every caster angle, the x-offset needed to hit specific trail
     targets (5/10/15/20/30 mm), plus the resulting ball-joint displacement
     from the current CAD position.
  3. Rank the geometrically sensible (offset <= 0) candidates and recommend
     one.

Coordinates: ISO 8855 (x forward, y left, z up), millimetres. All fixed
hardpoints are read from the geometry file given via --geometry (defaults to
configs/uct_rear_geometry_generated.yaml, but the pipeline points this at
Stage A's winner, outputs/rear/mount_height_kpi_sweep/best_zkpi_geometry_rear.yaml,
which - unlike the raw CAD file - has real axle_inboard/axle_outboard values,
even if only the rear Stage A placeholder ones).

This is a purely geometric (line/ground-plane intersection) calculation -
it does not run the kinematics solver. Roll centre, bump steer and motion
ratio for any candidate geometry must be re-checked separately once hardpoints
are actually moved in CAD.

Usage:
    python scripts/rear_caster_trail_analysis.py
    python scripts/rear_caster_trail_analysis.py --geometry outputs/rear/mount_height_kpi_sweep/best_zkpi_geometry_rear.yaml --out-dir outputs/rear/caster_and_trail
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ===========================================================================
# Configuration
# ===========================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GEOMETRY_PATH = REPO_ROOT / "configs" / "uct_rear_geometry_generated.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "rear" / "caster_and_trail"

# Dunlop 155/70R13, free (unloaded) rolling radius - same tyre spec as the
# front corner (confirmed identical in both uct_*_geometry_generated.yaml
# config blocks), so this constant is shared/unchanged from the front script.
FREE_RADIUS_MM = 273.6

CASTER_MIN_DEG = 0.0
CASTER_MAX_DEG = 7.0
CASTER_STEP_DEG = 0.25

ACCEPT_LOW_MM, ACCEPT_HIGH_MM = 5.0, 30.0
OPTIMAL_LOW_MM, OPTIMAL_HIGH_MM = 10.0, 20.0
TARGET_TRAIL_MM = 15.0  # midpoint of the optimal band -> the "ideal" offset

FIXED_OFFSETS_MM = [0, -5, -10, -15, -20, -25]
SCORE_OFFSET_WEIGHT = 0.1  # 1 deg of caster is worth tolerating 10mm of offset


# ===========================================================================
# Geometry loading
# ===========================================================================

def load_geometry(geometry_path: Path) -> dict:
    with open(geometry_path) as f:
        return yaml.safe_load(f)


def hp(hardpoints: dict, name: str) -> np.ndarray:
    p = hardpoints[name]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def build_context(geometry_path: Path) -> SimpleNamespace:
    """Load `geometry_path` (any file in the CAD file's schema - the raw
    CAD file itself, or a sweep winner written in the same schema) and
    derive every constant the rest of this module needs."""
    geometry = load_geometry(geometry_path)
    hardpoints = geometry["hardpoints"]

    p0 = hp(hardpoints, "upper_wishbone_outboard")  # upper ball joint (top of kingpin axis)
    q0 = hp(hardpoints, "lower_wishbone_outboard")  # lower ball joint (bottom of kingpin axis)
    axle_in = hp(hardpoints, "axle_inboard")
    axle_out = hp(hardpoints, "axle_outboard")

    wc_xy = (axle_in + axle_out) / 2.0
    x_wc, y_wc = wc_xy[0], wc_xy[1]
    z_wc = FREE_RADIUS_MM  # wheel centre sits at the tyre's free radius above ground

    upright_length = float(np.linalg.norm(p0 - q0))
    vy0 = p0[1] - q0[1]  # y-separation of the ball joints, unchanged by caster rotation
    lxz = math.sqrt(upright_length**2 - vy0**2)  # length of the axis's XZ projection

    # Fractional height at which the wheel centre sits along the *original*
    # axis (by z). Kept fixed across the sweep so the rotated axis is
    # anchored consistently relative to the upright, rather than arbitrarily.
    t_wc = (z_wc - q0[2]) / (p0[2] - q0[2])

    return SimpleNamespace(
        geometry_path=geometry_path, geometry=geometry,
        p0=p0, q0=q0, x_wc=x_wc, y_wc=y_wc, z_wc=z_wc,
        upright_length=upright_length, lxz=lxz, t_wc=t_wc,
    )


# ===========================================================================
# Core geometry helpers
# ===========================================================================

def rotate_axis(ctx: SimpleNamespace, caster_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotated (p, q) at the given caster angle: axis passes through the wheel
    centre, upright length preserved, each point's y unchanged, rotation
    confined to the XZ plane. Positive caster tilts the top (p) rearward
    (negative x)."""
    theta = math.radians(caster_deg)
    vx = -ctx.lxz * math.sin(theta)
    vz = ctx.lxz * math.cos(theta)

    qx = ctx.x_wc - ctx.t_wc * vx
    qz = ctx.z_wc - ctx.t_wc * vz
    px, pz = qx + vx, qz + vz

    p = np.array([px, ctx.p0[1], pz])
    q = np.array([qx, ctx.q0[1], qz])
    return p, q


def ground_intersection_x(p: np.ndarray, q: np.ndarray) -> float:
    """Side-view (XZ) x-position where the line through p and q meets z = 0."""
    t = (0.0 - q[2]) / (p[2] - q[2])
    return q[0] + t * (p[0] - q[0])


def mechanical_trail(ctx: SimpleNamespace, p: np.ndarray, q: np.ndarray) -> float:
    return ground_intersection_x(p, q) - ctx.x_wc


def front_view_ground_y(p: np.ndarray, q: np.ndarray) -> float:
    """Front-view (YZ) y-position where the line through p and q meets z = 0."""
    t = (0.0 - q[2]) / (p[2] - q[2])
    return q[1] + t * (p[1] - q[1])


def classify(trail_mm: float) -> str:
    if OPTIMAL_LOW_MM <= trail_mm <= OPTIMAL_HIGH_MM:
        return "OPTIMAL"
    if ACCEPT_LOW_MM <= trail_mm <= ACCEPT_HIGH_MM:
        return "ACCEPTABLE"
    return "FAIL"


# ===========================================================================
# Stage 1 - baseline caster sweep (zero offset)
# ===========================================================================

def run_stage1(ctx: SimpleNamespace) -> pd.DataFrame:
    casters = np.round(
        np.arange(CASTER_MIN_DEG, CASTER_MAX_DEG + 1e-9, CASTER_STEP_DEG), 2
    )

    rows = []
    for caster in casters:
        p, q = rotate_axis(ctx, caster)
        trail = mechanical_trail(ctx, p, q)
        rows.append(
            {
                "caster_deg": caster,
                "trail_mm": trail,
                "status": classify(trail),
                "p_rot": p,
                "q_rot": q,
            }
        )

    df = pd.DataFrame(rows)

    print("=" * 60)
    print("STAGE 1 - Baseline caster sweep (zero offset)")
    print("=" * 60)
    print(
        df[["caster_deg", "trail_mm", "status"]].to_string(
            index=False, formatters={"caster_deg": "{:.2f}".format, "trail_mm": "{:.2f}".format}
        )
    )
    print()
    return df


# ===========================================================================
# Stage 2 - required offset at each caster
# ===========================================================================

def offset_for_target(caster_deg: float, target_trail_mm: float) -> float:
    return target_trail_mm - FREE_RADIUS_MM * math.tan(math.radians(caster_deg))


def run_stage2(ctx: SimpleNamespace, stage1_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in stage1_df.iterrows():
        caster = r["caster_deg"]
        p_rot, q_rot = r["p_rot"], r["q_rot"]

        offset_5 = offset_for_target(caster, 5.0)
        offset_10 = offset_for_target(caster, 10.0)
        offset_15 = offset_for_target(caster, TARGET_TRAIL_MM)
        offset_20 = offset_for_target(caster, 20.0)
        offset_30 = offset_for_target(caster, 30.0)

        p_shifted = p_rot + np.array([offset_15, 0.0, 0.0])
        q_shifted = q_rot + np.array([offset_15, 0.0, 0.0])
        p_shift_mm = float(np.linalg.norm(p_shifted - ctx.p0))
        q_shift_mm = float(np.linalg.norm(q_shifted - ctx.q0))

        score = caster - SCORE_OFFSET_WEIGHT * abs(offset_15)

        rows.append(
            {
                "caster_deg": caster,
                "trail_at_zero_offset_mm": r["trail_mm"],
                "offset_for_5mm_trail": offset_5,
                "offset_for_10mm_trail": offset_10,
                "offset_for_15mm_trail": offset_15,
                "offset_for_20mm_trail": offset_20,
                "offset_for_30mm_trail": offset_30,
                "p_shift_mm": p_shift_mm,
                "q_shift_mm": q_shift_mm,
                "score": score,
                "p_rot": p_rot,
                "q_rot": q_rot,
                "p_shifted": p_shifted,
                "q_shifted": q_shifted,
            }
        )

    df = pd.DataFrame(rows)

    print("=" * 60)
    print("STAGE 2 - Required offset at each caster (target: 15mm trail)")
    print("=" * 60)
    print_cols = [
        "caster_deg",
        "trail_at_zero_offset_mm",
        "offset_for_15mm_trail",
        "p_shift_mm",
        "q_shift_mm",
    ]
    print(
        df[print_cols].to_string(
            index=False,
            formatters={c: "{:.2f}".format for c in print_cols if c != "caster_deg"}
            | {"caster_deg": "{:.2f}".format},
        )
    )
    print()
    return df


# ===========================================================================
# Stage 3 - rank and recommend
# ===========================================================================

def run_stage3(ctx: SimpleNamespace, stage2_df: pd.DataFrame) -> pd.Series:
    candidates = stage2_df[stage2_df["offset_for_15mm_trail"] <= 1e-9].copy()
    candidates = candidates.sort_values("score", ascending=False)
    top5 = candidates.head(5)

    print("=" * 60)
    print("STAGE 3 - Ranked candidates (offset <= 0, higher score better)")
    print("=" * 60)
    print_cols = ["caster_deg", "offset_for_15mm_trail", "p_shift_mm", "q_shift_mm", "score"]
    print(
        top5[print_cols].to_string(
            index=False,
            formatters={c: "{:.3f}".format for c in print_cols if c != "caster_deg"}
            | {"caster_deg": "{:.2f}".format},
        )
    )
    print()

    best = top5.iloc[0]

    caster = best["caster_deg"]
    offset = best["offset_for_15mm_trail"]
    trail = offset + FREE_RADIUS_MM * math.tan(math.radians(caster))

    print("RECOMMENDATION")
    print(f"Caster:           {caster:.1f} degrees")
    print(f"Axis offset:      {abs(offset):.1f} mm (rearward)")
    print(f"Mechanical trail: {trail:.1f} mm")
    print(f"Upper ball joint (p) shifts {best['p_shift_mm']:.1f} mm from current CAD position")
    print(f"Lower ball joint (q) shifts {best['q_shift_mm']:.1f} mm from current CAD position")
    print()
    print("CHECK: does this shift fit inside the upright and clear the wheel rim?")
    print("Take these numbers to the CAD model before locking anything in.")
    print()

    y_ground_before = front_view_ground_y(best["p_rot"], best["q_rot"])
    y_ground_after = front_view_ground_y(best["p_shifted"], best["q_shifted"])
    print("Front-view (KPI/scrub) check - ground intersection in YZ plane:")
    print(f"  before x-offset: y = {y_ground_before:.3f} mm")
    print(f"  after  x-offset: y = {y_ground_after:.3f} mm")
    print(f"  match: {'YES' if math.isclose(y_ground_before, y_ground_after, abs_tol=1e-6) else 'NO'}")
    print()

    return best


# ===========================================================================
# Outputs
# ===========================================================================

def write_csv(out_dir: Path, stage2_df: pd.DataFrame) -> None:
    cols = [
        "caster_deg",
        "trail_at_zero_offset_mm",
        "offset_for_5mm_trail",
        "offset_for_10mm_trail",
        "offset_for_15mm_trail",
        "offset_for_20mm_trail",
        "offset_for_30mm_trail",
        "p_shift_mm",
        "q_shift_mm",
        "score",
    ]
    out_path = out_dir / "caster_trail_analysis.csv"
    stage2_df[cols].to_csv(out_path, index=False, float_format="%.4f")
    print(f"Wrote {out_path}")


def write_handoff(out_dir: Path, best: pd.Series) -> None:
    """Stage B -> Stage C handoff: the winning caster/offset and the
    resulting shifted ball-joint positions, for
    scripts/rear_build_optimal_geometry.py to apply on top of the Z/KPI
    winner."""
    caster = float(best["caster_deg"])
    offset = float(best["offset_for_15mm_trail"])
    trail = offset + FREE_RADIUS_MM * math.tan(math.radians(caster))
    payload = {
        "caster_deg": caster,
        "offset_mm": offset,
        "trail_mm": float(trail),
        "p_shifted": [float(v) for v in best["p_shifted"]],
        "q_shifted": [float(v) for v in best["q_shifted"]],
        "p_shift_mm": float(best["p_shift_mm"]),
        "q_shift_mm": float(best["q_shift_mm"]),
    }
    out_path = out_dir / "best_caster_offset.yaml"
    with open(out_path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path}")


def plot_caster_vs_trail(out_dir: Path, stage1_df: pd.DataFrame, best: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)

    casters = stage1_df["caster_deg"].to_numpy()

    # Acceptance bands (behind everything else)
    ax.axhspan(ACCEPT_LOW_MM, ACCEPT_HIGH_MM, color="#c8f0c8", zorder=0, label="Acceptable (5-30 mm)")
    ax.axhspan(OPTIMAL_LOW_MM, OPTIMAL_HIGH_MM, color="#c8dcf5", zorder=1, label="Optimal (10-20 mm)")

    # Target trail line
    ax.axhline(TARGET_TRAIL_MM, color="gray", linestyle="--", linewidth=1, zorder=2, label="Target trail (15 mm)")

    # Baseline (Stage 1, zero offset) curve
    ax.plot(
        casters,
        stage1_df["trail_mm"].to_numpy(),
        color="#ff6f4f",
        linewidth=2.2,
        zorder=4,
        label="Baseline (zero offset)",
    )

    # Fixed-offset curves, progressively darker teal
    teal_shades = ["#a3e4d7", "#76d7c4", "#48c9b0", "#1abc9c", "#149174", "#0e6655"]
    for offset, color in zip(FIXED_OFFSETS_MM, teal_shades):
        trail = offset + FREE_RADIUS_MM * np.tan(np.radians(casters))
        ax.plot(casters, trail, color=color, linewidth=1.6, zorder=3)
        ax.annotate(
            f"{offset:+d} mm" if offset != 0 else "0 mm",
            xy=(casters[-1], trail[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=color,
            clip_on=False,
        )

    # Best candidate
    ax.plot(
        best["caster_deg"],
        TARGET_TRAIL_MM,
        marker="*",
        markersize=22,
        color="gold",
        markeredgecolor="black",
        markeredgewidth=0.8,
        linestyle="none",
        zorder=6,
        clip_on=False,
        label="Recommended candidate",
    )

    ax.set_xlim(0, 7)
    ax.set_ylim(-10, 60)
    ax.set_xlabel("Caster angle (degrees)")
    ax.set_ylabel("Mechanical trail (mm)")
    ax.set_title("Caster vs. mechanical trail - baseline and fixed-offset curves (rear)")
    ax.margins(x=0)
    fig.subplots_adjust(right=0.86)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    out_path = out_dir / "caster_vs_trail.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_offset_and_shift(out_dir: Path, stage2_df: pd.DataFrame, best: pd.Series) -> None:
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 8), dpi=150, sharex=True, gridspec_kw={"height_ratios": [1, 1.2]}
    )

    casters = stage2_df["caster_deg"].to_numpy()
    offset_15 = stage2_df["offset_for_15mm_trail"].to_numpy()

    y_min = min(offset_15.min(), -30) - 2
    y_max = max(offset_15.max(), 2) + 2

    ax_top.axhspan(-25, 0, color="#c8f0c8", zorder=0, label="Likely feasible (0 to -25 mm)")
    ax_top.axhspan(y_min, -25, color="#f8d7b0", zorder=0, label="Check carefully (< -25 mm)")
    ax_top.plot(casters, offset_15, color="#1abc9c", linewidth=2, zorder=3, label="Offset for 15 mm trail")
    ax_top.plot(
        best["caster_deg"],
        best["offset_for_15mm_trail"],
        marker="*",
        markersize=20,
        color="gold",
        markeredgecolor="black",
        markeredgewidth=0.8,
        linestyle="none",
        zorder=5,
        clip_on=False,
        label="Recommended candidate",
    )
    ax_top.set_ylim(y_min, y_max)
    ax_top.set_ylabel("Required offset (mm)")
    ax_top.set_title("Offset required for 15 mm trail, and resulting ball-joint displacement (rear)")
    ax_top.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax_top.grid(True, alpha=0.25)

    p_shift = stage2_df["p_shift_mm"].to_numpy()
    q_shift = stage2_df["q_shift_mm"].to_numpy()
    ax_bot.plot(casters, p_shift, color="#2e86c1", linewidth=2, label="Upper ball joint (p) shift")
    ax_bot.plot(casters, q_shift, color="#cb4335", linewidth=2, label="Lower ball joint (q) shift")
    ax_bot.plot(
        best["caster_deg"],
        best["p_shift_mm"],
        marker="*",
        markersize=18,
        color="gold",
        markeredgecolor="black",
        markeredgewidth=0.8,
        linestyle="none",
        zorder=5,
        clip_on=False,
    )
    ax_bot.plot(
        best["caster_deg"],
        best["q_shift_mm"],
        marker="*",
        markersize=18,
        color="gold",
        markeredgecolor="black",
        markeredgewidth=0.8,
        linestyle="none",
        zorder=5,
        clip_on=False,
    )
    ax_bot.axvline(best["caster_deg"], color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax_top.axvline(best["caster_deg"], color="gray", linestyle=":", linewidth=1, alpha=0.7)

    ax_bot.set_xlabel("Caster angle (degrees)")
    ax_bot.set_ylabel("Ball joint displacement (mm)")
    ax_bot.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_bot.grid(True, alpha=0.25)
    ax_bot.set_xlim(0, 7)

    out_path = out_dir / "offset_and_shift_vs_caster.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY_PATH,
                         help="geometry YAML to analyse (CAD file schema)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help="directory to write CSV/PNG/YAML outputs to")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = build_context(args.geometry)
    print(f"Analysing geometry: {args.geometry}\n")

    stage1_df = run_stage1(ctx)
    stage2_df = run_stage2(ctx, stage1_df)
    best = run_stage3(ctx, stage2_df)

    write_csv(out_dir, stage2_df)
    write_handoff(out_dir, best)
    plot_caster_vs_trail(out_dir, stage1_df, best)
    plot_offset_and_shift(out_dir, stage2_df, best)


if __name__ == "__main__":
    main()
