"""
Scoped plotting for motion_study.py's CSV outputs.

Only the panels most relevant to diagnosing the bump-steer/roll-steer
failures Stage C's confirmatory solve found: the pure_steer /
steer_at_fixed_roll 8-panel set (the primary KPI/caster test - the only
motion with zero travel contamination) and the bump_steer_family panel set
(the spread between curves is the actual bump-steer number). Deferred:
roll+steer cornering, pure heave, pure roll, and cross-variant overlay panel
sets - build those once this first pass proves useful.

Two rules throughout: plot against the physical variable (roadwheel angle,
not rack mm), and every panel tolerates a missing or all-null column - skip
that series, don't crash the figure (metric availability varies with the
solver's mechanism config, e.g. damper_length is None on our current
spring-less geometry).

Usage:
    python scripts/plot_motion_study.py --kpi 6 --caster 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STUDY_DIR = REPO_ROOT / "outputs" / "motion_study"

COLOR_LEFT = "#2e86c1"
COLOR_RIGHT = "#cb4335"
COLOR_REF = "#888888"

STEER_X_COL = "roadwheel_angle_left"
STEER_XLABEL = "steer angle, left wheel (deg, roadwheel_angle_left)"


def _true_steer(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Convert roadwheel_angle_left/right (a per-side, mirrored
    toward-centerline convention - see motion_study.ackermann_pct_row) to a
    common true-yaw sign, so left and right are directly comparable."""
    return -df["roadwheel_angle_left"], df["roadwheel_angle_right"]


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns and df[c].notna().any() for c in cols)


def _no_data(ax) -> None:
    ax.text(0.5, 0.5, "data unavailable", ha="center", va="center",
            transform=ax.transAxes, color=COLOR_REF, fontsize=8)


def _line_panel(ax, df: pd.DataFrame, xcol: str, series: list[tuple[str, str, str]],
                 title: str, xlabel: str, ylabel: str) -> None:
    """series: list of (column, label, color). Skips any column that's
    missing or entirely null; shows a placeholder if nothing plotted."""
    plotted = False
    for ycol, label, color in series:
        if ycol in df.columns and df[ycol].notna().any() and xcol in df.columns:
            ax.plot(df[xcol], df[ycol], color=color, linewidth=1.6, label=label)
            plotted = True
    if not plotted:
        _no_data(ax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(alpha=0.2)
    if plotted:
        ax.legend(fontsize=7)


# ===========================================================================
# Pure steer / steer at fixed roll - 8-panel set + overlay
# ===========================================================================

def static_track_mm(df: pd.DataFrame) -> float | None:
    """Track at the sweep's mid step (static, rack centred for pure_steer),
    from the solver's own `track` metric - None if unavailable."""
    if not _has(df, "track"):
        return None
    value = df["track"].iloc[len(df) // 2]
    return None if pd.isna(value) else float(value)


def variant_wheelbase_mm(study_dir: Path, tag: str) -> float | None:
    """Wheelbase from the variant geometry motion_study.py saved alongside
    the CSVs - None if the file or the field is missing."""
    path = study_dir / "runs" / tag / "variant_geometry.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        geometry = yaml.safe_load(f) or {}
    value = (geometry.get("vehicle_config") or {}).get("wheelbase")
    return float(value) if value is not None else None


def _ackermann_panel(ax, df: pd.DataFrame, track_mm: float | None, wheelbase_mm: float | None) -> None:
    if not _has(df, "roadwheel_angle_left", "roadwheel_angle_right"):
        _no_data(ax)
        ax.set_title("Ackermann: left vs right steer", fontsize=9)
        return

    psi_l, psi_r = _true_steer(df)
    ax.plot(psi_l, psi_r, color=COLOR_LEFT, linewidth=1.8, label="achieved")

    lo, hi = min(psi_l.min(), psi_r.min()), max(psi_l.max(), psi_r.max())
    ax.plot([lo, hi], [lo, hi], "--", color=COLOR_REF, linewidth=1, label="parallel steer (0%)")

    # 100% Ackermann reference: cot(outer) - cot(inner) = track/wheelbase.
    span = max(abs(lo), abs(hi))
    if track_mm is None or wheelbase_mm is None:
        ax.text(0.02, 0.98, "100% Ackermann reference unavailable\n(no track or wheelbase)",
                transform=ax.transAxes, va="top", fontsize=7, color=COLOR_REF)
    elif span > 0.5:
        ideal = track_mm / wheelbase_mm
        inner_mag = np.linspace(0.5, span, 60)
        with np.errstate(divide="ignore", invalid="ignore"):
            outer_mag = np.degrees(np.arctan(1.0 / (1.0 / np.tan(np.radians(inner_mag)) - ideal)))
        sign = 1.0 if (psi_l.iloc[-1] - psi_l.iloc[0]) >= 0 else -1.0
        ax.plot(sign * inner_mag, sign * outer_mag, ":", color="#1abc9c", linewidth=1.4,
                label="100% Ackermann")

    ax.set_title("Ackermann: left vs right steer", fontsize=9)
    ax.set_xlabel("true steer, left (deg)", fontsize=8)
    ax.set_ylabel("true steer, right (deg)", fontsize=8)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)


def _contact_patch_panel(ax, df: pd.DataFrame) -> None:
    plotted = False
    mid = len(df) // 2
    for side, color in (("left", COLOR_LEFT), ("right", COLOR_RIGHT)):
        xcol, ycol = f"{side}_contact_patch_center_x", f"{side}_contact_patch_center_y"
        if xcol in df.columns and ycol in df.columns and df[xcol].notna().any():
            ax.plot(df[xcol] - df[xcol].iloc[mid], df[ycol] - df[ycol].iloc[mid],
                     color=color, linewidth=1.6, label=side)
            plotted = True
    if not plotted:
        _no_data(ax)
    ax.set_title("Contact patch migration (ref. to static)", fontsize=9)
    ax.set_xlabel("delta X, forward (mm)", fontsize=8)
    ax.set_ylabel("delta Y, left (mm)", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    if plotted:
        ax.legend(fontsize=7)


def plot_pure_steer(df_steer: pd.DataFrame, variant_label: str, out_path: Path,
                     wheelbase_mm: float | None) -> None:
    track_mm = static_track_mm(df_steer)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)

    _line_panel(axes[0, 0], df_steer, STEER_X_COL,
                [("camber_left", "left", COLOR_LEFT), ("camber_right", "right", COLOR_RIGHT)],
                "Camber vs steer (common mode ~ KPI, differential ~ caster)", STEER_XLABEL, "camber (deg)")

    _ackermann_panel(axes[0, 1], df_steer, track_mm, wheelbase_mm)

    _line_panel(axes[0, 2], df_steer, STEER_X_COL,
                [("scrub_radius_left", "left", COLOR_LEFT), ("scrub_radius_right", "right", COLOR_RIGHT)],
                "Scrub radius vs steer", STEER_XLABEL, "scrub radius (mm)")

    _line_panel(axes[0, 3], df_steer, STEER_X_COL,
                [("mechanical_trail_left", "left", COLOR_LEFT), ("mechanical_trail_right", "right", COLOR_RIGHT)],
                "Mechanical trail vs steer (where it flattens/reverses, feel goes numb)",
                STEER_XLABEL, "trail (mm)")

    _contact_patch_panel(axes[1, 0], df_steer)

    _line_panel(axes[1, 1], df_steer, STEER_X_COL,
                [("ride_height_change", "ride height", COLOR_REF),
                 ("wheel_travel_left", "left travel", COLOR_LEFT),
                 ("wheel_travel_right", "right travel", COLOR_RIGHT)],
                "Chassis jacking vs steer", STEER_XLABEL, "mm")

    _line_panel(axes[1, 2], df_steer, STEER_X_COL,
                [("kpi_left", "KPI", COLOR_LEFT), ("caster_left", "caster", COLOR_RIGHT)],
                "KPI and caster vs steer (not constant through lock)", STEER_XLABEL, "deg")

    _line_panel(axes[1, 3], df_steer, STEER_X_COL,
                [("roadwheel_angle_left", "toe, left", COLOR_LEFT),
                 ("roadwheel_angle_right", "toe, right", COLOR_RIGHT),
                 ("track_change", "track change (mm)", COLOR_REF)],
                "Toe angle and track change vs steer", STEER_XLABEL, "deg / mm")

    fig.suptitle(f"Pure steer at ride height - {variant_label}", fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_steer_vs_fixed_roll_overlay(df_steer: pd.DataFrame, df_fixed_roll: pd.DataFrame,
                                      variant_label: str, out_path: Path) -> None:
    """Solid = pure_steer, dashed = steer_at_fixed_roll. The difference
    between them is how much roll corrupts the steering geometry."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    def overlay(ax, ycol, title, ylabel):
        plotted = False
        for df, ls, tag in ((df_steer, "-", "pure steer"), (df_fixed_roll, "--", "fixed roll")):
            for side, color in (("left", COLOR_LEFT), ("right", COLOR_RIGHT)):
                col = ycol.format(side=side)
                if col in df.columns and df[col].notna().any() and STEER_X_COL in df.columns:
                    ax.plot(df[STEER_X_COL], df[col], linestyle=ls, color=color, linewidth=1.5,
                             label=f"{side} ({tag})")
                    plotted = True
        if not plotted:
            _no_data(ax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(STEER_XLABEL, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.2)
        if plotted:
            ax.legend(fontsize=6.5)

    overlay(axes[0], "camber_{side}", "Camber vs steer", "camber (deg)")
    overlay(axes[1], "mechanical_trail_{side}", "Mechanical trail vs steer", "trail (mm)")

    ax = axes[2]
    plotted = False
    for df, ls, tag in ((df_steer, "-", "pure steer"), (df_fixed_roll, "--", "fixed roll")):
        if _has(df, "roadwheel_angle_left", "roadwheel_angle_right"):
            psi_l, psi_r = _true_steer(df)
            ax.plot(psi_l, psi_r, linestyle=ls, color=COLOR_LEFT, linewidth=1.6, label=tag)
            plotted = True
    if not plotted:
        _no_data(ax)
    ax.set_title("Ackermann overlay", fontsize=9)
    ax.set_xlabel("true steer, left (deg)", fontsize=8)
    ax.set_ylabel("true steer, right (deg)", fontsize=8)
    ax.grid(alpha=0.2)
    if plotted:
        ax.legend(fontsize=7)

    fig.suptitle(f"Pure steer (solid) vs steer at fixed roll (dashed) - {variant_label}", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ===========================================================================
# Bump steer family
# ===========================================================================

def plot_bump_steer_family(dfs_by_rack: dict[str, pd.DataFrame], variant_label: str,
                            out_path: Path) -> None:
    """One curve per steer angle. The spread between curves is the number
    that matters - flat at zero steer, fanning out at lock = unsolved bump
    steer."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("coolwarm")
    try:
        rack_values = sorted(dfs_by_rack.keys(), key=float)
    except ValueError:
        rack_values = sorted(dfs_by_rack.keys())
    n = max(len(rack_values) - 1, 1)
    colors = {v: cmap(i / n) for i, v in enumerate(rack_values)}

    panels = [
        (axes[0], "roadwheel_angle_left", "Toe vs travel (bump steer)", "toe, left (deg)"),
        (axes[1], "camber_left", "Camber vs travel", "camber, left (deg)"),
        (axes[2], "scrub_radius_left", "Scrub radius vs travel", "scrub radius, left (mm)"),
    ]
    for ax, ycol, title, ylabel in panels:
        plotted = False
        for v in rack_values:
            df = dfs_by_rack[v]
            if "wheel_travel_left" in df.columns and ycol in df.columns and df[ycol].notna().any():
                ax.plot(df["wheel_travel_left"], df[ycol], color=colors[v], linewidth=1.6,
                         label=f"rack {v}mm")
                plotted = True
        if not plotted:
            _no_data(ax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("wheel travel, left (mm)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.2)
        if plotted:
            ax.legend(fontsize=7)

    fig.suptitle(f"Bump steer family - {variant_label}\n"
                 "(spread between curves is the number that matters)", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ===========================================================================
# Driver
# ===========================================================================

def load_variant_csvs(study_dir: Path, tag: str) -> dict[str, pd.DataFrame]:
    variant_dir = study_dir / "runs" / tag
    return {csv.stem: pd.read_csv(csv) for csv in sorted(variant_dir.glob("*.csv"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--kpi", type=float, required=True)
    parser.add_argument("--caster", type=float, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    if not args.plots:
        print("--plots not set, skipping plot generation")
        return

    tag = f"kpi{args.kpi:.1f}_caster{args.caster:.1f}"
    dfs = load_variant_csvs(args.study_dir, tag)
    if not dfs:
        print(f"no CSVs found for {tag} under {args.study_dir / 'runs'}")
        return

    out_dir = args.study_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_steer = dfs.get("pure_steer")
    df_fixed_roll = dfs.get("steer_at_fixed_roll")
    if df_steer is not None:
        plot_pure_steer(df_steer, tag, out_dir / f"{tag}_pure_steer.png",
                        variant_wheelbase_mm(args.study_dir, tag))
        if df_fixed_roll is not None:
            plot_steer_vs_fixed_roll_overlay(
                df_steer, df_fixed_roll, tag, out_dir / f"{tag}_pure_steer_vs_fixed_roll.png"
            )
    else:
        print(f"no pure_steer.csv for {tag}, skipping pure_steer panels")

    prefix = "bump_steer_family_rack"
    family_dfs = {k[len(prefix):]: v for k, v in dfs.items() if k.startswith(prefix)}
    if family_dfs:
        plot_bump_steer_family(family_dfs, tag, out_dir / f"{tag}_bump_steer_family.png")
    else:
        print(f"no bump_steer_family CSVs for {tag}, skipping")


if __name__ == "__main__":
    main()
