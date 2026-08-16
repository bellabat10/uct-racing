"""
Stage A - merged mount-height (Z) x kingpin-inclination (KPI) sweep, axle
scope, front-left corner.

Replaces the three earlier, mutually-inconsistent sweep scripts
(iteration1_axle_sweep.py, iteration2_upright_sweep.py,
iteration2_kpi_sweep.py) with one grid search:

Outer loop 1 : upper wishbone chassis mount height Z, 240mm to the upright's
               height at KPI=0 (the tallest it ever is), 10mm steps. Y stays
               pinned per-leg to the lower wishbone's own Y (front/rear legs
               sit at different Y on the real chassis - not a shared plate).
Outer loop 2 : KPI angle, 0-10 degrees in 1 degree steps, front-view (Y-Z)
               rigid-body rotation of the upright (upper ball joint p, track
               rod outboard, axle inboard/outboard) about the lower ball
               joint q. Every point keeps its real CAD distance to q - only
               orientation changes.
Inner loops  : for each (Z, KPI) candidate, run TWO travel sweeps
                 - BUMP (+/-30mm, both wheels together, rack static)
                 - ROLL (+/-25mm, wheels opposed)
               (no RACK sweep here - none of the scored/reported metrics
               below need it; it's reproduced once, for the final geometry
               only, in build_optimal_geometry.py)

trackrod_inboard is NOT re-derived from the rotated trackrod_outboard - it
stays exactly at its raw CAD value from solidworks_points.yaml, for every
(Z, KPI) candidate. (Reversing the earlier "straight tie rod" convention:
that forced the rack colinear with an unmeasured placeholder point, which
iteration2_kpi_sweep.py's docstring already flagged as a mistake.)

Scoring is iteration2_kpi_sweep.py's 9-target weighted-cost formula, trimmed
to 7: caster_static and mechanical_trail_static are dropped here because a
Z x KPI-only sweep can't meaningfully move them (KPI rotates in the Y-Z
plane; caster/trail live in the X-Z plane) - both are constant across this
whole grid, so scoring against them here would just add a fixed offset to
every candidate's cost, not discriminate between them. They come back into
the full 9-target score in build_optimal_geometry.py, once caster/trail are
actually set by scripts/caster_trail_analysis.py.

Coordinates: ISO 8855 - X forward, Y left, Z up, millimetres.
The right corner is mirrored automatically (the `right` block is omitted).

Usage:
    python scripts/mount_kpi_sweep.py --smoke              # one geometry, prove it runs
    python scripts/mount_kpi_sweep.py --detail Z KPI        # full curves, one geometry
    python scripts/mount_kpi_sweep.py --plots               # full grid sweep + all plots
    python scripts/mount_kpi_sweep.py --render-best         # linkage drawing at the winner
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _sweep_common as sc  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "front" / "mount_height_kpi_sweep"

AXLE_POSITION = "front"

# Target scrub radius (mm, positive = kingpin axis meets the ground inboard
# of the contact patch), hit via a lateral (Y) shift of the kingpin axis
# endpoints (p, q) applied after the KPI rotation - see build_upright().
TARGET_SCRUB_RADIUS_MM = 10.0

# Target static camber (deg, solver convention: negative = top of wheel
# tilts inward), hit via a front-view rotation of axle_inboard/axle_outboard
# about the wheel centre, applied after the KPI rotation and scrub-radius
# shift - see build_upright().
TARGET_CAMBER_DEG = -3.5

_GEOMETRY = sc.load_generated_geometry(AXLE_POSITION)
_HARDPOINTS = _GEOMETRY["hardpoints"]

LOWER_INBOARD_FRONT = sc.hp(_HARDPOINTS, "lower_wishbone_inboard_front")
LOWER_INBOARD_REAR = sc.hp(_HARDPOINTS, "lower_wishbone_inboard_rear")
LOWER_OUTBOARD = sc.hp(_HARDPOINTS, "lower_wishbone_outboard")            # q - the KPI pivot

UPPER_INBOARD_X_FRONT = sc.hp(_HARDPOINTS, "upper_wishbone_inboard_front")[0]
UPPER_INBOARD_X_REAR = sc.hp(_HARDPOINTS, "upper_wishbone_inboard_rear")[0]
CHASSIS_MOUNT_Y_FRONT = LOWER_INBOARD_FRONT[1]
CHASSIS_MOUNT_Y_REAR = LOWER_INBOARD_REAR[1]

_UPPER_OUTBOARD_CAD = sc.hp(_HARDPOINTS, "upper_wishbone_outboard")        # p
_TRACKROD_OUTBOARD_CAD = sc.hp(_HARDPOINTS, "trackrod_outboard")
_AXLE_INBOARD_CAD = sc.hp(_HARDPOINTS, "axle_inboard")
_AXLE_OUTBOARD_CAD = sc.hp(_HARDPOINTS, "axle_outboard")
TRACKROD_INBOARD_RAW = sc.hp(_HARDPOINTS, "trackrod_inboard")             # constant, raw CAD

_CG = _GEOMETRY["config"]["cg_position"]
CG_POSITION = (float(_CG["x"]), float(_CG["y"]), float(_CG["z"]))
WHEELBASE = float(_GEOMETRY["config"]["wheelbase"])
_TIRE_CFG = _GEOMETRY["config"]["wheel"]["tire"]
TIRE = {"aspect_ratio": float(_TIRE_CFG["aspect_ratio"]),
        "section_width": float(_TIRE_CFG["section_width"]),
        "rim_diameter": float(_TIRE_CFG["rim_diameter"])}
WHEEL_OFFSET = float(_GEOMETRY["config"]["wheel"]["offset"] or 0)

from kinematics.core.schema.config import TireConfig  # noqa: E402

TIRE_RADIUS_MM = TireConfig(**TIRE).nominal_radius

# ===========================================================================
# Rigid-body KPI rotation (front-view Y-Z plane, about q)
# ===========================================================================
# Angle convention: from vertical, positive = top leans inboard (toward
# vehicle centreline) - matching "0 to 10 degrees".


def _angle_from_vertical(point, pivot) -> float:
    dy = point[1] - pivot[1]
    dz = point[2] - pivot[2]
    return math.degrees(math.atan2(-dy, dz))


def _rotate_about_pivot(point, pivot, delta_deg: float):
    """Rotate `point` about `pivot` in the Y-Z plane by delta_deg. X and the
    distance to pivot are preserved exactly."""
    dy = point[1] - pivot[1]
    dz = point[2] - pivot[2]
    r = math.hypot(dy, dz)
    theta = math.atan2(-dy, dz) + math.radians(delta_deg)
    new_dy = -r * math.sin(theta)
    new_dz = r * math.cos(theta)
    return (point[0], pivot[1] + new_dy, pivot[2] + new_dz)


BASELINE_KPI_DEG = _angle_from_vertical(_UPPER_OUTBOARD_CAD, LOWER_OUTBOARD)
FV_UPRIGHT_LENGTH = math.hypot(_UPPER_OUTBOARD_CAD[1] - LOWER_OUTBOARD[1],
                                _UPPER_OUTBOARD_CAD[2] - LOWER_OUTBOARD[2])


def _wheel_center_from_axle(axle_inboard, axle_outboard) -> tuple[float, float, float]:
    """Same convention as the solver's own get_wheel_center(): start at the
    hub face (axle_outboard) and move along the axle axis by WHEEL_OFFSET
    toward axle_inboard."""
    ai, ao = np.array(axle_inboard, dtype=float), np.array(axle_outboard, dtype=float)
    v = ao - ai
    v = v / np.linalg.norm(v)
    return tuple(ao - v * WHEEL_OFFSET)


def _wheel_plane_down_from_axle(axle_inboard, axle_outboard) -> np.ndarray:
    """Exact reimplementation of the solver's get_wheel_plane_down_vector()
    (kinematics.core.points.derived.definitions): the component of global
    'down' (-Z) orthogonal to the axle (spin axis), via Gram-Schmidt - i.e.
    'down' as seen within the wheel's own plane, which tilts with camber."""
    axle_in, axle_out = np.array(axle_inboard, dtype=float), np.array(axle_outboard, dtype=float)
    axle_dir = axle_out - axle_in
    axle_dir = axle_dir / np.linalg.norm(axle_dir)
    global_down = np.array([0.0, 0.0, -1.0])
    down_parallel = np.dot(global_down, axle_dir) * axle_dir
    wheel_down = global_down - down_parallel
    return wheel_down / np.linalg.norm(wheel_down)


def _contact_patch_from_axle(axle_inboard, axle_outboard) -> tuple[float, float, float]:
    """True contact patch centre - matches the solver's own
    get_contact_patch_center() exactly: wheel centre + tire radius along
    the wheel-plane 'down' direction. Unlike the wheel centre alone, this
    accounts for the camber-driven lateral offset (tire_radius *
    sin(camber)) between the wheel centre and where the tyre actually
    touches the ground."""
    wheel_centre = np.array(_wheel_center_from_axle(axle_inboard, axle_outboard))
    wheel_down = _wheel_plane_down_from_axle(axle_inboard, axle_outboard)
    return tuple(wheel_centre + wheel_down * TIRE_RADIUS_MM)


def _kingpin_ground_intersection_y(p, q) -> float:
    """Y where the line through q (lower) and p (upper) crosses the world
    ground plane (Z = 0, ISO) - a simple planar approximation (unlike the
    solver's own scrub_radius_static, this ignores steer and the contact
    patch's actual Z height), good enough to pre-position a candidate
    before the solver's confirmatory re-solve reports the real number."""
    t = (0.0 - q[2]) / (p[2] - q[2])
    return q[1] + t * (p[1] - q[1])


def _scrub_shift_y(p, q, axle_inboard, axle_outboard, target_scrub_mm: float) -> float:
    """Lateral (Y) shift to apply to BOTH p and q so the kingpin axis's
    ground intersection lands target_scrub_mm inboard of the TRUE contact
    patch centre Y (positive = inboard, matching the solver's scrub-radius
    sign convention) - not the wheel centre, which ignores the camber-driven
    offset between the two. A pure Y shift of both endpoints moves the
    ground intersection Y by that same amount (their Z's - and so the
    line's ground-crossing parameter t - don't change), so this is a direct
    closed-form solve, not an iteration."""
    contact_patch = _contact_patch_from_axle(axle_inboard, axle_outboard)
    intersection_y = _kingpin_ground_intersection_y(p, q)
    current_scrub = contact_patch[1] - intersection_y
    return current_scrub - target_scrub_mm


def _camber_from_axle(axle_inboard, axle_outboard, side: float = 1.0) -> float:
    """Static camber (deg) computed directly from the axle hardpoints - an
    exact reimplementation of the solver's own calculate_camber()
    (kinematics.core.metrics.angles), which depends only on wheel_axis
    (axle_inboard -> axle_outboard) and never references p or q. side=+1.0
    matches ctx.side_sign for the left corner this file builds. Verified
    numerically against the real solver's calculate_camber() output on the
    same geometry."""
    axle_in, axle_out = np.array(axle_inboard, dtype=float), np.array(axle_outboard, dtype=float)
    axle = axle_out - axle_in
    axle = axle / np.linalg.norm(axle)
    wheel_up = np.cross(axle, np.array([1.0, 0.0, 0.0])) * -side
    angle = math.atan2(wheel_up[1], wheel_up[2])
    camber_rad = angle if side > 0 else -angle
    return math.degrees(camber_rad)


def _camber_correction_delta(axle_inboard, axle_outboard, target_camber_deg: float) -> float:
    """Front-view rotation angle to apply to axle_inboard/axle_outboard
    (about the wheel centre) so static camber lands on target_camber_deg.
    Verified numerically: rotating both axle points by +delta via
    _rotate_about_pivot shifts camber by -delta (that helper's angle sign
    runs opposite to calculate_camber's), so the needed delta is
    current - target, not target - current."""
    current_camber = _camber_from_axle(axle_inboard, axle_outboard)
    return current_camber - target_camber_deg


def build_upright(kpi_deg: float):
    """Rotate all upright-mounted points as a rigid body around q so the
    kingpin axis (p-q) sits at kpi_deg from vertical, then rotate
    axle_inboard/axle_outboard about the wheel centre so static camber hits
    TARGET_CAMBER_DEG, then shift p and q together laterally (Y) so the
    kingpin axis hits TARGET_SCRUB_RADIUS_MM of scrub radius - measured
    against the TRUE contact patch centre, which needs the axle points at
    their FINAL (post-camber-correction) orientation, since the contact
    patch moves with camber. Computing the camber correction first (even
    though it reads as the "second" correction) is what makes the scrub
    target land on the geometry the solver will actually confirm, instead
    of one that's about to shift out from under it.

    trackrod_outboard is untouched by either correction.
    upper/lower_outboard (p, q) move only in the scrub-radius shift - never
    in the camber correction. axle_inboard/axle_outboard move only in the
    camber correction - never in the scrub-radius shift.

    KPI is a p-q angle and caster lives in the X-Z plane, so neither
    correction affects them: the scrub shift moves p and q by an identical
    Y offset (preserving the p-q angle exactly, and never touching X or Z);
    the camber correction never moves p or q at all. Static camber, in
    turn, depends only on axle_inboard/axle_outboard (see
    _camber_from_axle()) - completely independent of p and q - so rotating
    just the axle points about the wheel centre (itself invariant under
    that rotation, since it lies on the line being rotated) changes camber
    without moving the wheel or perturbing KPI/caster.

    Returns (upper_outboard, lower_outboard, trackrod_outboard,
    axle_inboard, axle_outboard)."""
    delta = kpi_deg - BASELINE_KPI_DEG
    upper_outboard = _rotate_about_pivot(_UPPER_OUTBOARD_CAD, LOWER_OUTBOARD, delta)
    trackrod_outboard = _rotate_about_pivot(_TRACKROD_OUTBOARD_CAD, LOWER_OUTBOARD, delta)
    axle_inboard = _rotate_about_pivot(_AXLE_INBOARD_CAD, LOWER_OUTBOARD, delta)
    axle_outboard = _rotate_about_pivot(_AXLE_OUTBOARD_CAD, LOWER_OUTBOARD, delta)

    wheel_centre = _wheel_center_from_axle(axle_inboard, axle_outboard)
    camber_delta = _camber_correction_delta(axle_inboard, axle_outboard, TARGET_CAMBER_DEG)
    axle_inboard = _rotate_about_pivot(axle_inboard, wheel_centre, camber_delta)
    axle_outboard = _rotate_about_pivot(axle_outboard, wheel_centre, camber_delta)

    shift_y = _scrub_shift_y(upper_outboard, LOWER_OUTBOARD, axle_inboard, axle_outboard,
                              TARGET_SCRUB_RADIUS_MM)
    lower_outboard = (LOWER_OUTBOARD[0], LOWER_OUTBOARD[1] + shift_y, LOWER_OUTBOARD[2])
    upper_outboard = (upper_outboard[0], upper_outboard[1] + shift_y, upper_outboard[2])

    return upper_outboard, lower_outboard, trackrod_outboard, axle_inboard, axle_outboard


# Z capped at the upright's height at KPI=0 (vertical - the tallest it ever
# is across the swept KPI range); infeasible (Z, KPI) pairs are rejected
# per-candidate in build_geometry().
_Z_CEILING = LOWER_OUTBOARD[2] + FV_UPRIGHT_LENGTH
Z_RANGE = np.arange(240.0, _Z_CEILING + 1e-9, 10.0)
KPI_RANGE = np.arange(5.0, 10.0 + 1e-9, 1.0)

TARGETS_STAGE_A = {k: v for k, v in sc.TARGETS_FULL.items()
                    if k not in ("caster_static", "mechanical_trail_static")}
WEIGHTS_STAGE_A = {k: v for k, v in sc.WEIGHTS_FULL.items()
                    if k not in ("caster_static", "mechanical_trail_static")}


# ===========================================================================
# Geometry builder (axle scope, solver schema)
# ===========================================================================

BASE_AXLE = {
    "type": "double_wishbone",
    "scope": "axle",
    "name": "mount_kpi_sweep",
    "version": "0.0.1",
    "units": "millimeters",
    "vehicle_config": {"cg_position": sc.pt(CG_POSITION), "wheelbase": WHEELBASE},
    "axle_config": {
        "axle_position": AXLE_POSITION,
        "steering": {"type": "rack"},
        "actuation": {"type": "direct", "mount": "lower_wishbone"},
        "spring": {"type": "none"},
        "anti_roll": {"type": "none"},
        "heave_link": {"type": "none"},
        "wheel": {"offset": WHEEL_OFFSET, "tire": TIRE},
    },
    "hardpoints": {
        "left": {
            "lower_wishbone_inboard_front": sc.pt(LOWER_INBOARD_FRONT),
            "lower_wishbone_inboard_rear": sc.pt(LOWER_INBOARD_REAR),
            "lower_wishbone_outboard": sc.pt(LOWER_OUTBOARD),
            "upper_wishbone_inboard_front": sc.pt((UPPER_INBOARD_X_FRONT, 0, 0)),
            "upper_wishbone_inboard_rear": sc.pt((UPPER_INBOARD_X_REAR, 0, 0)),
            "upper_wishbone_outboard": sc.pt((0, 0, 0)),
            "trackrod_inboard": sc.pt(TRACKROD_INBOARD_RAW),
            "trackrod_outboard": sc.pt((0, 0, 0)),
            "axle_inboard": sc.pt((0, 0, 0)),
            "axle_outboard": sc.pt((0, 0, 0)),
        }
    },
}


def build_geometry(z_up: float, kpi_deg: float) -> dict:
    """Place the upper chassis mount at height z_up and rotate the upright
    rigid body to kpi_deg (degrees from vertical, front view)."""
    upper_outboard, lower_outboard, trackrod_outboard, axle_inboard, axle_outboard = build_upright(kpi_deg)

    if z_up > upper_outboard[2] + 1e-6:
        raise ValueError(
            f"upper chassis mount z={z_up:.1f} is above the upright ball "
            f"joint (z={upper_outboard[2]:.1f}) at KPI={kpi_deg:.1f} deg"
        )

    g = copy.deepcopy(BASE_AXLE)
    hp = g["hardpoints"]["left"]
    hp["upper_wishbone_inboard_front"] = sc.pt((UPPER_INBOARD_X_FRONT, CHASSIS_MOUNT_Y_FRONT, z_up))
    hp["upper_wishbone_inboard_rear"] = sc.pt((UPPER_INBOARD_X_REAR, CHASSIS_MOUNT_Y_REAR, z_up))
    hp["upper_wishbone_outboard"] = sc.pt(upper_outboard)
    hp["lower_wishbone_outboard"] = sc.pt(lower_outboard)
    hp["trackrod_outboard"] = sc.pt(trackrod_outboard)
    hp["axle_inboard"] = sc.pt(axle_inboard)
    hp["axle_outboard"] = sc.pt(axle_outboard)
    return g


def write_cad_schema_geometry(path: Path, z_up: float, kpi_deg: float) -> None:
    """Write the winning (z_up, kpi_deg) geometry in the SAME schema as
    configs/uct_front_geometry_generated.yaml (flat hardpoints, config
    block) - so scripts/caster_trail_analysis.py's existing point-loading
    code works against it with zero changes beyond the file path."""
    upper_outboard, lower_outboard, trackrod_outboard, axle_inboard, axle_outboard = build_upright(kpi_deg)

    out = copy.deepcopy(_GEOMETRY)
    hp = out["hardpoints"]
    hp["upper_wishbone_inboard_front"] = sc.pt((UPPER_INBOARD_X_FRONT, CHASSIS_MOUNT_Y_FRONT, z_up))
    hp["upper_wishbone_inboard_rear"] = sc.pt((UPPER_INBOARD_X_REAR, CHASSIS_MOUNT_Y_REAR, z_up))
    hp["upper_wishbone_outboard"] = sc.pt(upper_outboard)
    hp["lower_wishbone_outboard"] = sc.pt(lower_outboard)
    hp["trackrod_outboard"] = sc.pt(trackrod_outboard)
    hp["axle_inboard"] = sc.pt(axle_inboard)
    hp["axle_outboard"] = sc.pt(axle_outboard)
    out["name"] = "UCT Racing FSA front left - mount/KPI sweep winner"

    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)


# ===========================================================================
# Evaluate one candidate
# ===========================================================================

def evaluate(args) -> dict:
    z_up, kpi_deg = args
    out = {"z_up": z_up, "kpi_deg": kpi_deg, "feasible": False, "error": ""}
    try:
        geometry = build_geometry(z_up, kpi_deg)
        metrics = sc.evaluate_candidate(geometry)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    out["feasible"] = True
    out.update(metrics)
    out["weighted_cost"] = sc.weighted_cost(out, TARGETS_STAGE_A, WEIGHTS_STAGE_A)
    return out


# ===========================================================================
# Drivers
# ===========================================================================

def smoke() -> None:
    z, kpi = float(np.median(Z_RANGE)), float(np.median(KPI_RANGE))
    print(f"smoke test at upper mount z={z:.0f}, KPI={kpi:.1f} deg")
    r = evaluate((z, kpi))
    if not r["feasible"]:
        print(f"  FAILED: {r['error']}")
        return
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:26s} {v:12.4f}")
    print("\nIf these look sane, the wiring is correct - run --plots next.")


def detail(z: float, kpi: float, prefix: str = str(OUTPUT_DIR / "detail"),
           curves_name: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometry = build_geometry(z, kpi)
    bump = sc.run_sweep(geometry, sc.BUMP_SWEEP)
    roll = sc.run_sweep(geometry, sc.ROLL_SWEEP)
    pd.DataFrame(bump).to_csv(f"{prefix}_bump.csv", index=False)
    pd.DataFrame(roll).to_csv(f"{prefix}_roll.csv", index=False)

    hm = [r.get("heave") for r in bump]
    rd = [r.get("roll") for r in roll]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = [
        (ax[0, 0], hm, [r.get("camber_left") for r in bump], "bump (mm)", "camber (deg)"),
        (ax[0, 1], hm, [r.get("roll_center_z") for r in bump], "bump (mm)", "roll centre z (mm)"),
        (ax[0, 2], hm, [r.get("roadwheel_angle_left") for r in bump], "bump (mm)", "roadwheel angle / toe (deg)"),
        (ax[1, 0], rd, [r.get("camber_left") for r in roll], "roll (deg)", "camber, left (deg)"),
        (ax[1, 1], rd, [r.get("roll_center_z") for r in roll], "roll (deg)", "roll centre z (mm)"),
        (ax[1, 2], rd, [r.get("roadwheel_angle_left") for r in roll], "roll (deg)", "roadwheel angle, left (deg)"),
    ]
    for a, xs, ys, xl, yl in panels:
        a.plot(xs, ys, "-o", ms=3)
        a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.3)
    fig.suptitle(f"Upper mount z={z:.0f}, KPI={kpi:.1f} deg   top row: bump    bottom row: roll")
    curves_path = curves_name or f"{prefix}_curves.png"
    fig.savefig(curves_path, dpi=150)
    print(f"wrote {curves_path}, {prefix}_bump.csv, {prefix}_roll.csv")


def run_grid(workers=None) -> pd.DataFrame:
    grid = list(itertools.product(Z_RANGE, KPI_RANGE))
    print(f"sweeping {len(grid)} (Z, KPI) candidates x 2 motion sweeps "
          f"x {sc.MOTION_STEPS} steps = {len(grid) * 2 * sc.MOTION_STEPS} solves")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(evaluate, grid, chunksize=4))
    df = pd.DataFrame(results)
    print(f"  {int(df.feasible.sum())}/{len(df)} solved cleanly")
    return df


def _ranked_candidates(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.feasible].sort_values("weighted_cost", ascending=True)


def report(df: pd.DataFrame, n: int = 15) -> None:
    ok = _ranked_candidates(df)
    cols = ["z_up", "kpi_deg", "kpi_static", "scrub_radius_static", "roll_centre_z_static",
            "bump_steer_max", "camber_gain_roll_pct", "camber_gain_bump_per25mm",
            "roll_steer_front", "weighted_cost"]
    print(f"\nTop {n} of {len(ok)} feasible (Z, KPI) candidates by weighted cost "
          f"(against {len(TARGETS_STAGE_A)} weighted targets):\n")
    print(ok[[c for c in cols if c in ok.columns]].head(n)
          .to_string(index=False, float_format=lambda v: f"{v:9.3f}"))


def make_plots(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z = np.array(sorted(df.z_up.unique()))
    K = np.array(sorted(df.kpi_deg.unique()))
    best = _ranked_candidates(df).iloc[0]

    def grid_of(col):
        return (df.pivot(index="kpi_deg", columns="z_up", values=col)
                  .reindex(index=K, columns=Z).to_numpy(dtype=float))

    def heatmap_fig(panels, suptitle, fname, figsize):
        n = len(panels)
        ncols = min(n, 3)
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False)
        for ax, (col, title, cmap) in zip(axes.ravel(), panels):
            G = grid_of(col)
            m = ax.contourf(Z, K, G, levels=20, cmap=cmap)
            fig.colorbar(m, ax=ax)
            cs = ax.contour(Z, K, G, levels=10, colors="k", linewidths=0.4, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=6, fmt="%.3g")
            ax.plot(best.z_up, best.kpi_deg, "o", ms=10, mfc="none", mec="k", mew=2)
            ax.set_title(title)
            ax.set_xlabel("upper mount z (mm)"); ax.set_ylabel("KPI angle (deg)")
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        fig.suptitle(suptitle, fontsize=13)
        fig.savefig(OUTPUT_DIR / fname, dpi=150)
        plt.close(fig)
        print(f"wrote {OUTPUT_DIR / fname}")

    heatmap_fig(
        [("roll_centre_z_static", "Roll centre height (mm)", "viridis"),
         ("camber_gain_roll_pct", "Camber gain in roll (compensation frac.)", "RdYlGn"),
         ("roll_steer_front", "Front roll steer (deg/deg)", "RdBu_r"),
         ("rc_migration_roll", "RC migration in roll (mm/deg)", "magma_r"),
         ("rc_migration_roll_y", "Lateral RC excursion in roll (mm)", "magma_r"),
         ("weighted_cost", "Weighted cost (lower better)", "YlOrRd_r")],
        "Roll behaviour - Z x KPI sweep", "mount_kpi_heatmaps_roll.png", (16, 9),
    )
    heatmap_fig(
        [("kpi_static", "KPI (deg) - sanity check vs input", "viridis"),
         ("scrub_radius_static", "Scrub radius (mm)", "RdBu_r")],
        "KPI and scrub radius - Z x KPI sweep", "mount_kpi_heatmaps_kpi_scrub.png", (10, 4.5),
    )
    heatmap_fig(
        [("bump_steer_max", "Bump steer, max toe change (deg)", "RdYlGn_r"),
         ("camber_gain_bump_per25mm", "Camber gain in bump (deg/25mm)", "RdBu_r"),
         ("rc_migration_heave", "RC migration in heave (mm/mm)", "magma_r")],
        "Bump behaviour - Z x KPI sweep", "mount_kpi_heatmaps_bump.png", (15, 4.5),
    )
    heatmap_fig(
        [("upper_arm_length", "Upper wishbone front-leg length (mm)", "viridis"),
         ("fvsa_length_left", "Front-view swing arm length (mm)", "viridis")],
        "How arm lengths change across the sweep - Z x KPI", "arm_lengths_vs_zkpi.png", (10, 4.5),
    )


def render_best(csv_path: str) -> None:
    from kinematics.core.input import build_suspension
    from kinematics.cli.visualization.api import visualize_geometry

    df = pd.read_csv(csv_path)
    ok = _ranked_candidates(df)
    if ok.empty:
        print("no feasible candidates to render")
        return
    best = ok.iloc[0]
    z, kpi = float(best.z_up), float(best.kpi_deg)
    print(f"rendering best (Z, KPI) = ({z:.0f}, {kpi:.1f}), weighted cost {best.weighted_cost:.3f}")

    geometry = build_geometry(z, kpi)
    suspension = build_suspension(geometry)
    out_path = OUTPUT_DIR / "mount_kpi_best_geometry.png"
    result = visualize_geometry(suspension, out_path)
    print(f"wrote {result.output_path}  (contact patch on ground: {result.contact_patch_on_ground})")

    detail(z, kpi, prefix=str(OUTPUT_DIR / "mount_kpi_best"),
           curves_name=str(OUTPUT_DIR / "mount_kpi_best_vs_travel.png"))

    geom_out = OUTPUT_DIR / "best_zkpi_geometry.yaml"
    write_cad_schema_geometry(geom_out, z, kpi)
    print(f"wrote {geom_out}")


def main() -> None:
    global Z_RANGE, KPI_RANGE

    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--detail", nargs=2, type=float, metavar=("Z", "KPI"))
    p.add_argument("--plots", action="store_true")
    p.add_argument("--render-best", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default=str(OUTPUT_DIR / "mount_kpi_results.csv"))
    a = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if a.smoke:
        smoke(); return
    if a.detail:
        detail(a.detail[0], a.detail[1]); return
    if a.render_best:
        render_best(a.out); return

    # Two-pass grid search: a coarse pass to find the promising (Z, KPI)
    # region, then a fine pass confined to that region - much cheaper than
    # running the fine step size over the whole original range.
    orig_z_range, orig_kpi_range = Z_RANGE, KPI_RANGE
    z_lo, z_hi = float(orig_z_range[0]), float(orig_z_range[-1])
    kpi_lo, kpi_hi = float(orig_kpi_range[0]), float(orig_kpi_range[-1])
    COARSE_Z_STEP, COARSE_KPI_STEP = 20.0, 2.0
    FINE_Z_STEP, FINE_KPI_STEP = 5.0, 0.5
    COST_FILTER_MULTIPLIER = 1.5

    print("=== pass 1/2: coarse grid ===")
    Z_RANGE = np.arange(z_lo, z_hi + 1e-9, COARSE_Z_STEP)
    KPI_RANGE = np.arange(kpi_lo, kpi_hi + 1e-9, COARSE_KPI_STEP)
    coarse_df = run_grid(a.workers)
    ok = coarse_df[coarse_df.feasible]
    survivors = (ok[ok.weighted_cost <= ok.weighted_cost.min() * COST_FILTER_MULTIPLIER]
                 if not ok.empty else ok)

    if survivors.empty:
        print(f"WARNING: 0/{len(coarse_df)} coarse candidates survived the cost filter - "
              f"falling back to a single-pass grid search at the original step sizes.")
        Z_RANGE, KPI_RANGE = orig_z_range, orig_kpi_range
        df = run_grid(a.workers)
    else:
        z_min = max(survivors.z_up.min() - COARSE_Z_STEP, z_lo)
        z_max = min(survivors.z_up.max() + COARSE_Z_STEP, z_hi)
        kpi_min = max(survivors.kpi_deg.min() - COARSE_KPI_STEP, kpi_lo)
        kpi_max = min(survivors.kpi_deg.max() + COARSE_KPI_STEP, kpi_hi)
        print(f"coarse pass: {len(coarse_df)} candidates evaluated, {len(survivors)} survived "
              f"(cost <= {COST_FILTER_MULTIPLIER:.1f}x best)")
        print(f"refined ranges: Z [{z_min:.0f}, {z_max:.0f}] mm, KPI [{kpi_min:.1f}, {kpi_max:.1f}] deg")

        print("=== pass 2/2: fine grid ===")
        Z_RANGE = np.arange(z_min, z_max + 1e-9, FINE_Z_STEP)
        KPI_RANGE = np.arange(kpi_min, kpi_max + 1e-9, FINE_KPI_STEP)
        df = run_grid(a.workers)

    Z_RANGE, KPI_RANGE = orig_z_range, orig_kpi_range

    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
    report(df)
    if a.plots:
        make_plots(df)
        render_best(a.out)


if __name__ == "__main__":
    main()
