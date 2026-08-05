"""
Iteration 1 (AXLE SCOPE) - optimal position of the UPPER wishbone chassis mount.

Outer loop  : sweep the upper wishbone chassis mount position (y, z)
Inner loops : for each candidate geometry, run TWO motion sweeps
                - heave  (both wheels same direction)  -> bump behaviour
                - roll   (wheels opposite directions)  -> cornering behaviour
Then reduce each candidate to scalars, score, rank, and plot.

Axle scope is used deliberately: roll_center_y, roll_center_z, roll, track and
track_change are AXLE-ONLY metrics in this solver, and a true roll sweep needs
both wheels. That gives roll centre and camber compensation natively instead of
computing them by hand.

Simplifications for this iteration:
  - lower wishbone chassis mounts   FIXED (loaded from the CAD-derived geometry)
  - lower ball joint                FIXED (loaded from the CAD-derived geometry)
  - upper ball joint                FIXED (loaded from the CAD-derived geometry)
  - upper chassis mount             THE FREE VARIABLE (Z only - see below)

Coordinates: ISO 8855 - X forward, Y left, Z up, millimetres.
The right corner is mirrored automatically (the `right` block is omitted).

Usage:
    python iteration1_axle_sweep.py --smoke        # one geometry, prove it runs
    python iteration1_axle_sweep.py --detail 220 340   # full curves, one geometry
    python iteration1_axle_sweep.py --plots        # full grid sweep + plots
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ===========================================================================
# 1. CONFIGURATION  -  the only block you should need to edit
# ===========================================================================

AXLE_POSITION = "front"                        # "front" or "rear"

# ---- FIXED hardpoints, LEFT corner (right is mirrored) --------------------
# Loaded from configs/uct_<axle_position>_geometry_generated.yaml - the
# CAD-derived geometry - rather than hand-typed. A previous hand-typed
# version of this block had the lower wishbone's chassis-side mount ~35mm
# ABOVE the upright mount; the real CAD has it ~10mm BELOW, which is the
# opposite relationship and was inflating the computed roll centre height.

def _load_generated_geometry(axle_position: str) -> dict:
    path = (Path(__file__).resolve().parent / "configs"
            / f"uct_{axle_position}_geometry_generated.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


_GEOMETRY = _load_generated_geometry(AXLE_POSITION)
_HARDPOINTS = _GEOMETRY["hardpoints"]


def _hp(name: str) -> tuple[float, float, float]:
    p = _HARDPOINTS[name]
    return (float(p["x"]), float(p["y"]), float(p["z"]))


LOWER_INBOARD_FRONT = _hp("lower_wishbone_inboard_front")
LOWER_INBOARD_REAR = _hp("lower_wishbone_inboard_rear")
LOWER_OUTBOARD = _hp("lower_wishbone_outboard")           # LOWER BALL JOINT
UPPER_OUTBOARD = _hp("upper_wishbone_outboard")            # UPPER BALL JOINT

# ---- upper chassis legs: x fixed, y and z are the sweep variables --------
# X stays at each leg's real chassis-tube fore/aft position.
UPPER_INBOARD_X_FRONT = _hp("upper_wishbone_inboard_front")[0]
UPPER_INBOARD_X_REAR = _hp("upper_wishbone_inboard_rear")[0]

# ---- wheel / axle --------------------------------------------------------
_AXLE_INBOARD = _hp("axle_inboard")
_AXLE_OUTBOARD = _hp("axle_outboard")
WHEEL_CENTRE = tuple((a + b) / 2.0 for a, b in zip(_AXLE_INBOARD, _AXLE_OUTBOARD))
AXLE_HALF_LENGTH = abs(_AXLE_OUTBOARD[1] - _AXLE_INBOARD[1]) / 2.0

# ---- track rod (keeps the upright constrained; does not affect camber,
#      roll centre or the front-view IC) ----------------------------------
# The tie rod (rack to steering arm) must run as one straight line, level
# and with no fore-aft kink - a raked or bent tie rod is a bump-steer/binding
# problem and the rack can't push the steering arm cleanly through one. The
# CAD source flags trackrod_inboard's X and height as placeholders (both
# marked MISSING in solidworks_points.yaml), so instead of trusting them
# as-is, the rack attachment is pulled onto the same X and Z as the outboard
# (steering arm) mount - only Y (lateral, rack travel) differs between them.
# Both sides move together (mirrored), so the rack stays level and straight.
TRACKROD_OUTBOARD = _hp("trackrod_outboard")
_TRACKROD_INBOARD_RAW = _hp("trackrod_inboard")
TRACKROD_INBOARD = (TRACKROD_OUTBOARD[0], _TRACKROD_INBOARD_RAW[1], TRACKROD_OUTBOARD[2])

# ---- vehicle context -----------------------------------------------------
_CG = _GEOMETRY["config"]["cg_position"]
CG_POSITION = (float(_CG["x"]), float(_CG["y"]), float(_CG["z"]))
WHEELBASE = float(_GEOMETRY["config"]["wheelbase"])
_TIRE_CFG = _GEOMETRY["config"]["wheel"]["tire"]
TIRE = {"aspect_ratio": float(_TIRE_CFG["aspect_ratio"]),
        "section_width": float(_TIRE_CFG["section_width"]),
        "rim_diameter": float(_TIRE_CFG["rim_diameter"])}
WHEEL_OFFSET = float(_GEOMETRY["config"]["wheel"]["offset"] or 0)

# ---- outer sweep: upper chassis mount ------------------------------------
# The upper wishbone bolts to the same chassis-side structure as the lower
# wishbone. In the real chassis the front and rear legs are NOT at the same
# Y (209.64 vs 254.34 for the lower wishbone) - there's no single "vertical
# plate" - so each leg's Y is pinned to its OWN lower-wishbone counterpart.
# Z is shared front/rear (the real lower wishbone's front/rear Z differ by
# well under 1mm, so this is an accurate simplification) and is the only
# free variable, capped at the upper ball joint's height and sweeping
# downward from there.
CHASSIS_MOUNT_Y_FRONT = LOWER_INBOARD_FRONT[1]
CHASSIS_MOUNT_Y_REAR = LOWER_INBOARD_REAR[1]
Y_RANGE = np.array([CHASSIS_MOUNT_Y_FRONT])    # reporting/plot axis; see build_geometry
Z_RANGE = np.arange(240.0, UPPER_OUTBOARD[2] + 1e-9, 10.0)

# ---- inner sweeps: motion ------------------------------------------------
HEAVE_TRAVEL = 30.0        # +/- mm, both wheels together
ROLL_TRAVEL = 25.0         # +/- mm, wheels opposed  (~2.4 deg roll on 1200 track)
MOTION_STEPS = 21          # must be odd so a step lands exactly on static

# ---- targets: (target, accept half-width, reject half-width) -------------
TARGETS = {
    "camber_compensation": (0.60, 0.15, 0.35),   # fraction of body roll cancelled
    "roll_centre_z":       (35.0, 15.0, 45.0),   # mm above ground, static
    "rc_migration_heave":  (0.0,  0.5,  1.0),    # |d(RC_z)/d(heave)|, mm/mm
    "rc_migration_roll_y": (0.0,  40.0, 120.0),  # lateral RC excursion in roll, mm
    "camber_gain_heave":   (-0.048, 0.015, 0.040),  # deg/mm, static (sign matters)
}
WEIGHTS = {
    "camber_compensation": 0.30,
    "roll_centre_z":       0.30,
    "rc_migration_heave":  0.13,
    "rc_migration_roll_y": 0.17,
    "camber_gain_heave":   0.10,
}


# ===========================================================================
# 2. GEOMETRY BUILDER  (axle scope)
# ===========================================================================

def _pt(t):
    return {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])}


BASE_AXLE = {
    "type": "double_wishbone",
    "scope": "axle",
    "name": "iteration1_axle",
    "version": "0.0.1",
    "units": "millimeters",
    "vehicle_config": {"cg_position": _pt(CG_POSITION), "wheelbase": WHEELBASE},
    "axle_config": {
        "axle_position": AXLE_POSITION,
        "steering": {"type": "rack"},
        "actuation": {"type": "direct", "mount": "lower_wishbone"},
        "spring": {"type": "none"},
        "anti_roll": {"type": "none"},
        "heave_link": {"type": "none"},
        "wheel": {"offset": WHEEL_OFFSET, "tire": TIRE},
    },
    # omitting `right` mirrors the whole left corner through Y = 0
    "hardpoints": {
        "left": {
            "lower_wishbone_inboard_front": _pt(LOWER_INBOARD_FRONT),
            "lower_wishbone_inboard_rear": _pt(LOWER_INBOARD_REAR),
            "lower_wishbone_outboard": _pt(LOWER_OUTBOARD),
            "upper_wishbone_inboard_front": _pt((UPPER_INBOARD_X_FRONT, 0, 0)),
            "upper_wishbone_inboard_rear": _pt((UPPER_INBOARD_X_REAR, 0, 0)),
            "upper_wishbone_outboard": _pt(UPPER_OUTBOARD),
            "trackrod_inboard": _pt(TRACKROD_INBOARD),
            "trackrod_outboard": _pt(TRACKROD_OUTBOARD),
            "axle_inboard": _pt((WHEEL_CENTRE[0],
                                 WHEEL_CENTRE[1] - AXLE_HALF_LENGTH,
                                 WHEEL_CENTRE[2])),
            "axle_outboard": _pt((WHEEL_CENTRE[0],
                                  WHEEL_CENTRE[1] + AXLE_HALF_LENGTH,
                                  WHEEL_CENTRE[2])),
        }
    },
}


def build_geometry(y_up: float, z_up: float) -> dict:
    """Place the upper chassis mount at height z_up on the left corner.

    y_up is accepted for interface stability (it's what's swept/reported)
    but not used directly: each leg's Y is pinned to its own real
    lower-wishbone counterpart (CHASSIS_MOUNT_Y_FRONT/REAR), since the front
    and rear legs sit at different Y in the real chassis.
    """
    g = copy.deepcopy(BASE_AXLE)
    hp = g["hardpoints"]["left"]
    hp["upper_wishbone_inboard_front"] = _pt((UPPER_INBOARD_X_FRONT, CHASSIS_MOUNT_Y_FRONT, z_up))
    hp["upper_wishbone_inboard_rear"] = _pt((UPPER_INBOARD_X_REAR, CHASSIS_MOUNT_Y_REAR, z_up))
    return g


# ---- the two motion sweeps ------------------------------------------------
# Axle targets carry a `side`. The rack is one shared lateral DOF, so it is
# targeted on ONE side only. Every actuator DOF must be pinned at every step,
# which is why the rack appears even when it is held at zero.

def _motion(left_start, left_stop, right_start, right_stop) -> dict:
    return {
        "version": 1,
        "steps": MOTION_STEPS,
        "targets": [
            {"point": "wheel_center", "side": "left", "direction": {"axis": "z"},
             "mode": "relative", "start": left_start, "stop": left_stop},
            {"point": "wheel_center", "side": "right", "direction": {"axis": "z"},
             "mode": "relative", "start": right_start, "stop": right_stop},
            {"point": "trackrod_inboard", "side": "left", "direction": {"axis": "y"},
             "mode": "relative", "start": 0.0, "stop": 0.0},
        ],
    }


HEAVE_SWEEP = _motion(-HEAVE_TRAVEL, HEAVE_TRAVEL, -HEAVE_TRAVEL, HEAVE_TRAVEL)
ROLL_SWEEP = _motion(-ROLL_TRAVEL, ROLL_TRAVEL, ROLL_TRAVEL, -ROLL_TRAVEL)


# ===========================================================================
# 3. SOLVER PLUMBING
# ===========================================================================

def _converged(info) -> bool:
    for attr in ("converged", "success", "solver_converged"):
        if hasattr(info, attr):
            return bool(getattr(info, attr))
    return True


def _run(geometry: dict, sweep_spec: dict):
    """Solve one motion sweep, return flat metric rows (or raise)."""
    from kinematics.core.input import build_suspension, build_sweep
    from kinematics.core.sweep import solve_evaluated_sweep

    suspension = build_suspension(geometry)
    sweep = build_sweep(sweep_spec, suspension)
    ev = solve_evaluated_sweep(suspension, sweep)
    if not all(_converged(s) for s in ev.solver_stats):
        raise RuntimeError("solver did not converge at every step")
    # At axle scope each row is an AxleMetricRows. flat_row() gives one dict per
    # step: corner metrics carry a _left / _right suffix, axle metrics do not.
    return [r.flat_row() for r in ev.metrics.rows]


def _slope(x, y):
    """Least-squares slope, ignoring None entries. Returns None if too few points."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    if max(xs) - min(xs) < 1e-9:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def _span(values):
    vals = [v for v in values if v is not None]
    return (max(vals) - min(vals)) if len(vals) >= 2 else None


# ===========================================================================
# 4. EVALUATE ONE CANDIDATE GEOMETRY
# ===========================================================================

def evaluate(args) -> dict:
    y_up, z_up = args
    out = {"y_up": y_up, "z_up": z_up, "feasible": False, "error": ""}
    try:
        geometry = build_geometry(y_up, z_up)
        heave = _run(geometry, HEAVE_SWEEP)
        roll = _run(geometry, ROLL_SWEEP)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    mid = len(heave) // 2                       # static ride height

    # ---- from the HEAVE sweep -------------------------------------------
    rc_z_h = [r.get("roll_center_z") for r in heave]
    heave_mm = [r.get("heave") for r in heave]
    cgain_h = [r.get("deriv_camber_wrt_hub_z_left") for r in heave]

    # ---- from the ROLL sweep --------------------------------------------
    roll_deg = [r.get("roll") for r in roll]
    camber_l = [r.get("camber_left") for r in roll]
    rc_y_r = [r.get("roll_center_y") for r in roll]
    rc_z_r = [r.get("roll_center_z") for r in roll]

    # Camber compensation: with no geometric help the wheel simply follows the
    # chassis, so camber relative to the road changes one-for-one with roll and
    # d(camber)/d(roll) = 1. Perfect compensation holds camber constant, giving 0.
    dcam_droll = _slope(roll_deg, camber_l)
    compensation = (1.0 - abs(dcam_droll)) if dcam_droll is not None else None

    if rc_z_h[mid] is None or cgain_h[mid] is None:
        out["error"] = "roll centre or camber gain undefined at static"
        return out

    out.update({
        "feasible": True,
        "roll_centre_z": rc_z_h[mid],
        "roll_centre_y": heave[mid].get("roll_center_y"),
        "camber_gain_heave": cgain_h[mid],
        "camber_static": heave[mid].get("camber_left"),
        "track_static": heave[mid].get("track"),
        "fvsa_length": heave[mid].get("fvsa_length_left"),
        "anti_dive": heave[mid].get("anti_dive_left"),
        "rc_migration_heave": abs(_slope(heave_mm, rc_z_h) or np.nan),
        "rc_span_heave": _span(rc_z_h),
        "camber_linearity": _span(cgain_h),
        "camber_compensation": compensation,
        "dcamber_droll": dcam_droll,
        "roll_range_deg": _span(roll_deg),
        "rc_migration_roll_y": _span(rc_y_r),
        "rc_migration_roll_z": _span(rc_z_r),
        "track_change": _span([r.get("track") for r in heave]),
        "upper_arm_length": math.dist(UPPER_OUTBOARD[1:], (y_up, z_up)),
    })
    return out


# ===========================================================================
# 5. SCORING
# ===========================================================================

def trapezoid(value, target, accept, reject) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    d = abs(value - target)
    if d <= accept:
        return 1.0
    if d >= reject:
        return 0.0
    return (reject - d) / (reject - accept)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    for key, (t, a, r) in TARGETS.items():
        col = df[key] if key in df.columns else pd.Series([None] * len(df))
        df[f"s_{key}"] = [trapezoid(v, t, a, r) for v in col]
    df["total"] = sum(df[f"s_{k}"] * WEIGHTS[k] for k in TARGETS)
    df["worst"] = df[[f"s_{k}" for k in TARGETS]].min(axis=1)
    df.loc[~df["feasible"], ["total", "worst"]] = 0.0
    return df


# ===========================================================================
# 6. DRIVERS
# ===========================================================================

def smoke() -> None:
    """One geometry, both sweeps. Run this first - it proves the API wiring."""
    y, z = float(np.median(Y_RANGE)), float(np.median(Z_RANGE))
    print(f"smoke test at upper mount ({y:.0f}, {z:.0f})")
    r = evaluate((y, z))
    if not r["feasible"]:
        print(f"  FAILED: {r['error']}")
        return
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:24s} {v:12.4f}")
    print("\nIf these look sane, the wiring is correct - run --plots next.")


def detail(y: float, z: float, prefix: str = str(OUTPUT_DIR / "detail")) -> None:
    """Full curve set through travel for ONE geometry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometry = build_geometry(y, z)
    heave = _run(geometry, HEAVE_SWEEP)
    roll = _run(geometry, ROLL_SWEEP)
    pd.DataFrame(heave).to_csv(f"{prefix}_heave.csv", index=False)
    pd.DataFrame(roll).to_csv(f"{prefix}_roll.csv", index=False)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    hm = [r.get("heave") for r in heave]
    rd = [r.get("roll") for r in roll]
    panels = [
        (ax[0, 0], hm, [r.get("camber_left") for r in heave], "heave (mm)", "camber (deg)"),
        (ax[0, 1], hm, [r.get("roll_center_z") for r in heave], "heave (mm)", "roll centre z (mm)"),
        (ax[0, 2], hm, [r.get("toe_angle_left") for r in heave], "heave (mm)", "toe (deg)"),
        (ax[1, 0], rd, [r.get("camber_left") for r in roll], "roll (deg)", "camber, left (deg)"),
        (ax[1, 1], rd, [r.get("roll_center_z") for r in roll], "roll (deg)", "roll centre z (mm)"),
        (ax[1, 2], rd, [r.get("roll_center_y") for r in roll], "roll (deg)", "roll centre y (mm)"),
    ]
    for a, xs, ys, xl, yl in panels:
        a.plot(xs, ys, "-o", ms=3)
        a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.3)
    fig.suptitle(f"Upper chassis mount at ({y:.0f}, {z:.0f})   "
                 f"top row: heave    bottom row: roll")
    fig.savefig(f"{prefix}_curves.png", dpi=150)
    print(f"wrote {prefix}_curves.png, {prefix}_heave.csv, {prefix}_roll.csv")


def run_grid(workers=None) -> pd.DataFrame:
    grid = list(itertools.product(Y_RANGE, Z_RANGE))
    print(f"sweeping {len(grid)} mount positions x 2 motion sweeps "
          f"x {MOTION_STEPS} steps = {len(grid)*2*MOTION_STEPS} solves")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(evaluate, grid, chunksize=4))
    df = add_scores(pd.DataFrame(results))
    print(f"  {int(df.feasible.sum())}/{len(df)} solved cleanly")
    return df


def _ranked_candidates(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.feasible & (df.worst > 0)].sort_values("total", ascending=False)
    if ok.empty:
        print("\nNo candidate satisfies every parameter. That usually means the "
              "targets are unreachable from this LOWER ball joint height, not that "
              "the search failed - check the trade curve plot before widening bands.")
        ok = df[df.feasible].sort_values("total", ascending=False)
    return ok


def report(df: pd.DataFrame, n: int = 10) -> None:
    ok = _ranked_candidates(df)
    cols = ["y_up", "z_up", "camber_compensation", "roll_centre_z",
            "rc_migration_heave", "rc_migration_roll_y", "camber_gain_heave",
            "upper_arm_length", "total", "worst"]
    print(f"\nTop {n} upper chassis mount positions:\n")
    print(ok[[c for c in cols if c in ok.columns]].head(n)
          .to_string(index=False, float_format=lambda v: f"{v:9.3f}"))


def make_plots(df: pd.DataFrame, prefix: str = str(OUTPUT_DIR / "iter1_axle")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Y = np.array(sorted(df.y_up.unique()))
    Z = np.array(sorted(df.z_up.unique()))

    def grid_of(col):
        return (df.pivot(index="z_up", columns="y_up", values=col)
                  .reindex(index=Z, columns=Y).to_numpy(dtype=float))

    panels = [
        ("camber_compensation", "Camber compensation in roll", "RdYlGn", "camber_compensation"),
        ("roll_centre_z", "Roll centre height (mm)", "viridis", "roll_centre_z"),
        ("rc_migration_roll_y", "Lateral RC excursion in roll (mm)", "magma_r", None),
        ("rc_migration_heave", "RC migration in heave (mm/mm)", "magma_r", None),
        ("camber_gain_heave", "Camber gain in heave (deg/mm)", "RdBu_r", "camber_gain_heave"),
        ("total", "Weighted total score", "YlGn", None),
    ]
    best = df[df.feasible & (df.worst > 0)].sort_values("total", ascending=False)
    best_row = best.iloc[0] if not best.empty else None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    if len(Y) == 1:
        # Y is pinned to the chassis mounting plane, so only Z varies -
        # a 2D contour is degenerate here; plot each metric against Z.
        d = df.sort_values("z_up")
        for ax, (col, title, cmap, tkey) in zip(axes.ravel(), panels):
            ax.plot(d.z_up, d[col], "-o", ms=4, color="tab:blue")
            if tkey:
                t, a, _ = TARGETS[tkey]
                ax.axhspan(t - a, t + a, color="tab:orange", alpha=0.15)
                ax.axhline(t, color="tab:red", linewidth=1)
            if best_row is not None:
                ax.axvline(best_row.z_up, color="k", linestyle="--", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("upper mount z (mm)"); ax.set_ylabel(col)
            ax.grid(alpha=0.3)
    else:
        for ax, (col, title, cmap, tkey) in zip(axes.ravel(), panels):
            G = grid_of(col)
            m = ax.contourf(Y, Z, G, levels=24, cmap=cmap)
            fig.colorbar(m, ax=ax)
            cs = ax.contour(Y, Z, G, levels=12, colors="k", linewidths=0.4, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=6, fmt="%.3g")
            if tkey:
                t, a, _ = TARGETS[tkey]
                b = ax.contour(Y, Z, G, levels=[t - a, t, t + a],
                               colors=["w", "r", "w"], linewidths=[1.2, 2.0, 1.2])
                ax.clabel(b, inline=True, fontsize=7, fmt="%.3g")
            ax.set_title(title)
            ax.set_xlabel("upper mount y (mm)"); ax.set_ylabel("upper mount z (mm)")
        if best_row is not None:
            for ax in axes.ravel():
                ax.plot(best_row.y_up, best_row.z_up, "o", ms=9, mfc="none", mec="k", mew=2)
    fig.suptitle("Iteration 1, axle scope - upper wishbone chassis mount", fontsize=14)
    fig.savefig(f"{prefix}_contours.png", dpi=150)

    fig2, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ok = df[df.feasible]
    s = ax.scatter(ok.camber_compensation, ok.roll_centre_z, s=6, c=ok.total,
                   cmap="YlGn", edgecolors="none")
    fig2.colorbar(s, ax=ax, label="total score")
    t, a, _ = TARGETS["camber_compensation"]; ax.axvspan(t - a, t + a, color="tab:blue", alpha=0.12)
    t, a, _ = TARGETS["roll_centre_z"]; ax.axhspan(t - a, t + a, color="tab:orange", alpha=0.12)
    ax.set_xlabel("camber compensation in roll"); ax.set_ylabel("roll centre height (mm)")
    ax.set_title("Trade curve\n(shaded bands are the accept ranges - do they overlap?)")
    ax.grid(alpha=0.3)
    fig2.savefig(f"{prefix}_trade_curve.png", dpi=150)
    print(f"wrote {prefix}_contours.png and {prefix}_trade_curve.png")


def render_best(csv_path: str, out_path: str = str(OUTPUT_DIR / "iter1_axle_best_geometry.png")) -> None:
    """Draw the actual linkage (4-view) at the top-scoring mount position."""
    from kinematics.core.input import build_suspension
    from kinematics.cli.visualization.api import visualize_geometry

    df = pd.read_csv(csv_path)
    ok = _ranked_candidates(df)
    if ok.empty:
        print("no feasible candidates to render")
        return
    best = ok.iloc[0]
    y, z = float(best.y_up), float(best.z_up)
    print(f"rendering best mount position ({y:.0f}, {z:.0f}), total score {best.total:.3f}")

    geometry = build_geometry(y, z)
    suspension = build_suspension(geometry)
    result = visualize_geometry(suspension, Path(out_path))
    print(f"wrote {result.output_path}  (contact patch on ground: {result.contact_patch_on_ground})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--detail", nargs=2, type=float, metavar=("Y", "Z"))
    p.add_argument("--plots", action="store_true")
    p.add_argument("--render-best", action="store_true",
                    help="draw the linkage at the top-scoring mount position from --out")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default=str(OUTPUT_DIR / "iter1_axle_results.csv"))
    a = p.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if a.smoke:
        smoke(); return
    if a.detail:
        detail(a.detail[0], a.detail[1]); return
    if a.render_best:
        render_best(a.out); return

    df = run_grid(a.workers)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
    report(df)
    if a.plots:
        make_plots(df)
        render_best(a.out)


if __name__ == "__main__":
    main()
