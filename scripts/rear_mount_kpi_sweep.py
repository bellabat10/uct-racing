"""
Stage A (rear) - mount-height (Z) x kingpin-inclination (KPI) sweep, axle
scope, rear-left corner. Cloned from scripts/mount_kpi_sweep.py (front) and
adapted for the rear corner's differences - read that file first if this one
is confusing, most of the structure and all the geometric derivations
(scrub-radius lateral shift, camber-target rotation, coarse-to-fine grid,
early termination) are identical.

Differences from the front script:

  Geometry source: configs/uct_rear_geometry_generated.yaml (steering type
  "none", scope "corner" in that file - this script still builds its own
  axle-scope geometry directly, same as the front script does, ignoring the
  file's own declared scope/steering).

  5th constraint: the rear has no rack - it uses a fixed TOE LINK
  (toe_link_inboard/outboard), not a trackrod. configs/solidworks_points.yaml
  has BOTH toe_link_inboard/outboard AND axle_inboard/axle_outboard marked
  null ("MISSING - measure from SolidWorks"), and uct_rear_geometry_generated.yaml
  doesn't carry those 4 keys at all. Placeholders are generated for whichever
  are missing (see _toe_link_points()/_axle_points()), with a warning printed
  at every entry point. Once the real upright/toe-link is measured, drop the
  values into solidworks_points.yaml and this script picks them up
  automatically - no code change needed.

  IMPORTANT technical note: scripts/_sweep_common.py's BUMP_SWEEP/ROLL_SWEEP
  (and therefore evaluate_candidate()) are hardcoded to pin
  "trackrod_inboard" at every sweep step - that point doesn't exist on a
  toe-link corner (verified empirically: the solver rejects the sweep with
  "Sweep target point 'LEFT_TRACKROD_INBOARD' is not present"). A fixed toe
  link has zero actuated degrees of freedom (only its OUTBOARD point moves,
  rigidly anchored to the upright - see kinematics' ToeLink), so it needs no
  actuator target at all: a 2-target sweep (both wheel_center Z, no 3rd
  target) solves cleanly (also verified empirically). This file therefore
  defines its own BUMP_SWEEP_REAR/ROLL_SWEEP_REAR and its own
  evaluate_candidate_rear(), rather than reusing _sweep_common.py's
  evaluate_candidate() - _sweep_common.py itself is untouched, and every
  genuinely reusable piece (run_sweep, slope, span, converged, weighted_cost,
  MOTION_STEPS/BUMP_TRAVEL/ROLL_TRAVEL) still comes from there.

  Targets/scoring: roll centre height, roll steer and bump-steer weighting
  all differ from front - see TARGETS_REAR/WEIGHTS_REAR below. Static camber
  is Tier 4 (checked, not weighted) exactly like the front script. Bump
  steer is reported but never weighted (no rack here - "bump steer" is just
  toe change induced by the placeholder toe link, not a scored target).

  Track width: rear half-track is meant to be 700mm (1400mm full track) per
  the team brief, but configs/solidworks_points.yaml currently still says
  650mm (1300mm) - main() flags this mismatch every run (see
  REAR_HALF_TRACK_MM below).

Coordinates: ISO 8855 - X forward, Y left, Z up, millimetres.
The right corner is mirrored automatically (the `right` block is omitted).

Usage:
    python scripts/rear_mount_kpi_sweep.py --smoke              # one geometry, prove it runs
    python scripts/rear_mount_kpi_sweep.py --detail Z KPI        # full curves, one geometry
    python scripts/rear_mount_kpi_sweep.py --plots               # full grid sweep + all plots
    python scripts/rear_mount_kpi_sweep.py --render-best         # linkage drawing at the winner
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
CONFIGS_DIR = REPO_ROOT / "configs"
OUTPUT_DIR = REPO_ROOT / "outputs" / "rear" / "mount_height_kpi_sweep"

AXLE_POSITION = "rear"

# Target scrub radius (mm, positive = kingpin axis meets the ground inboard
# of the contact patch) - same target and same lateral (Y) shift of p, q as
# the front script uses. See build_upright().
TARGET_SCRUB_RADIUS_MM = 10.0

# Target static camber (deg, solver convention: negative = top of wheel
# tilts inward) - same as front. See build_upright().
TARGET_CAMBER_DEG = -3.5

# Team brief: rear track 1400mm (half-track 700mm). configs/solidworks_points.yaml
# still has the OLD figure (650mm half-track / 1300mm full) - see main()'s
# verification print. Front and rear tracks are deliberately separate
# constants (not shared), since they differ physically.
REAR_HALF_TRACK_MM = 700.0

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
# Placeholder generation - toe link and axle (5th constraint + spin axis)
# ===========================================================================

def _toe_link_points(hardpoints: dict, q: tuple) -> tuple[tuple, tuple, bool]:
    """Real toe_link_inboard/outboard if the rear geometry has them
    (non-null), else PLACEHOLDER positions per the team's recipe: outboard
    at q's height and Y, 50mm rearward of q in X; inboard the same height
    and X offset, Y=200mm (toward centreline). Returns
    (inboard, outboard, is_placeholder)."""
    inboard_raw = hardpoints.get("toe_link_inboard")
    outboard_raw = hardpoints.get("toe_link_outboard")
    if inboard_raw is not None and outboard_raw is not None:
        return sc.hp(hardpoints, "toe_link_inboard"), sc.hp(hardpoints, "toe_link_outboard"), False

    toe_link_outboard = (q[0] - 50.0, q[1], q[2])
    toe_link_inboard = (q[0] - 50.0, 200.0, q[2])
    return toe_link_inboard, toe_link_outboard, True


def _axle_points(hardpoints: dict) -> tuple[tuple, tuple, bool]:
    """Real axle_inboard/outboard if the rear geometry has them (non-null),
    else PLACEHOLDER positions straddling the nominal wheel centre
    (x=-wheelbase, y=REAR_HALF_TRACK_MM, z=tyre radius) by 40mm in Y total -
    matching the front corner's real, measured hub span. Returns
    (inboard, outboard, is_placeholder)."""
    inboard_raw = hardpoints.get("axle_inboard")
    outboard_raw = hardpoints.get("axle_outboard")
    if inboard_raw is not None and outboard_raw is not None:
        return sc.hp(hardpoints, "axle_inboard"), sc.hp(hardpoints, "axle_outboard"), False

    wheel_centre_nominal = (-WHEELBASE, REAR_HALF_TRACK_MM, TIRE_RADIUS_MM)
    axle_inboard = (wheel_centre_nominal[0], wheel_centre_nominal[1] - 20.0, wheel_centre_nominal[2])
    axle_outboard = (wheel_centre_nominal[0], wheel_centre_nominal[1] + 20.0, wheel_centre_nominal[2])
    return axle_inboard, axle_outboard, True


TOE_LINK_INBOARD_RAW, TOE_LINK_OUTBOARD_RAW, TOE_LINK_IS_PLACEHOLDER = _toe_link_points(_HARDPOINTS, LOWER_OUTBOARD)
_AXLE_INBOARD_CAD, _AXLE_OUTBOARD_CAD, AXLE_IS_PLACEHOLDER = _axle_points(_HARDPOINTS)


def print_placeholder_warnings() -> None:
    """Called once from every CLI entry point (main(), smoke(), etc.) - not
    at module level, since ProcessPoolExecutor workers re-import this module
    per-worker and would otherwise repeat the warning many times over."""
    if TOE_LINK_IS_PLACEHOLDER:
        print("WARNING: toe_link_inboard/toe_link_outboard are missing from the rear "
              "geometry - using PLACEHOLDER positions (50mm rearward of the lower ball "
              "joint q in X, q's own Z; Y=q's Y outboard, Y=200mm inboard). Replace with "
              "real CAD measurements in configs/solidworks_points.yaml once the toe link "
              "is designed.")
    if AXLE_IS_PLACEHOLDER:
        print("WARNING: axle_inboard/axle_outboard are missing from the rear geometry - "
              "using PLACEHOLDER positions straddling the nominal wheel centre "
              "(x=-wheelbase, y=half-track, z=tyre radius), 40mm apart in Y. Replace with "
              "real CAD measurements in configs/solidworks_points.yaml once the rear "
              "upright is designed.")


def verify_track_width() -> None:
    """Cross-check REAR_HALF_TRACK_MM against configs/solidworks_points.yaml
    (the canonical source of half_track for this corner - the generated
    geometry YAML this script loads doesn't carry that field at all)."""
    sw_path = CONFIGS_DIR / "solidworks_points.yaml"
    with open(sw_path) as f:
        sw_points = yaml.safe_load(f)
    sw_half_track = sw_points.get("rear", {}).get("half_track")
    if sw_half_track is None:
        print(f"WARNING: could not find rear.half_track in {sw_path} to verify against "
              f"REAR_HALF_TRACK_MM={REAR_HALF_TRACK_MM:.0f}.")
        return
    if abs(float(sw_half_track) - REAR_HALF_TRACK_MM) > 1e-6:
        print(f"WARNING: track width mismatch - this script uses REAR_HALF_TRACK_MM="
              f"{REAR_HALF_TRACK_MM:.0f} ({2*REAR_HALF_TRACK_MM:.0f}mm full track), but "
              f"{sw_path.name}'s rear.half_track is {sw_half_track:.0f} "
              f"({2*float(sw_half_track):.0f}mm full track). Update solidworks_points.yaml "
              f"once the real rear track width is confirmed.")
    else:
        print(f"Track width check OK: REAR_HALF_TRACK_MM={REAR_HALF_TRACK_MM:.0f} matches "
              f"{sw_path.name}.")


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
    the wheel-plane 'down' direction."""
    wheel_centre = np.array(_wheel_center_from_axle(axle_inboard, axle_outboard))
    wheel_down = _wheel_plane_down_from_axle(axle_inboard, axle_outboard)
    return tuple(wheel_centre + wheel_down * TIRE_RADIUS_MM)


def _kingpin_ground_intersection_y(p, q) -> float:
    """Y where the line through q (lower) and p (upper) crosses the world
    ground plane (Z = 0, ISO) - a simple planar approximation, good enough
    to pre-position a candidate before the solver's confirmatory re-solve
    reports the real number."""
    t = (0.0 - q[2]) / (p[2] - q[2])
    return q[1] + t * (p[1] - q[1])


def _scrub_shift_y(p, q, axle_inboard, axle_outboard, target_scrub_mm: float) -> float:
    """Lateral (Y) shift to apply to BOTH p and q so the kingpin axis's
    ground intersection lands target_scrub_mm inboard of the TRUE contact
    patch centre Y. A pure Y shift of both endpoints moves the ground
    intersection Y by that same amount, so this is a direct closed-form
    solve, not an iteration."""
    contact_patch = _contact_patch_from_axle(axle_inboard, axle_outboard)
    intersection_y = _kingpin_ground_intersection_y(p, q)
    current_scrub = contact_patch[1] - intersection_y
    return current_scrub - target_scrub_mm


def _camber_from_axle(axle_inboard, axle_outboard, side: float = 1.0) -> float:
    """Static camber (deg) computed directly from the axle hardpoints - an
    exact reimplementation of the solver's own calculate_camber(). side=+1.0
    matches ctx.side_sign for the left corner this file builds."""
    axle_in, axle_out = np.array(axle_inboard, dtype=float), np.array(axle_outboard, dtype=float)
    axle = axle_out - axle_in
    axle = axle / np.linalg.norm(axle)
    wheel_up = np.cross(axle, np.array([1.0, 0.0, 0.0])) * -side
    angle = math.atan2(wheel_up[1], wheel_up[2])
    camber_rad = angle if side > 0 else -angle
    return math.degrees(camber_rad)


def _camber_correction_delta(axle_inboard, axle_outboard, target_camber_deg: float) -> float:
    """Front-view rotation angle to apply to axle_inboard/axle_outboard
    (about the wheel centre) so static camber lands on target_camber_deg."""
    current_camber = _camber_from_axle(axle_inboard, axle_outboard)
    return current_camber - target_camber_deg


def build_upright(kpi_deg: float):
    """Rotate all upright-mounted points as a rigid body around q so the
    kingpin axis (p-q) sits at kpi_deg from vertical, then rotate
    axle_inboard/axle_outboard about the wheel centre so static camber hits
    TARGET_CAMBER_DEG, then shift p and q together laterally (Y) so the
    kingpin axis hits TARGET_SCRUB_RADIUS_MM of scrub radius (measured
    against the TRUE, post-camber-correction contact patch - see
    scripts/mount_kpi_sweep.py's build_upright() docstring for why camber
    correction must run before the scrub shift is calculated).

    toe_link_outboard: if real CAD data, rotates with the upright rigid body
    (like the front's trackrod_outboard) about q. If a PLACEHOLDER, it's
    instead recomputed fresh from q's FINAL (post-scrub-shift) position -
    the placeholder recipe defines it AS an offset from q, not as an
    independent point with its own CAD distance to preserve under rotation.
    toe_link_inboard: real CAD data stays fixed (like the front's
    trackrod_inboard); a placeholder is likewise recomputed from q's final
    Z/X (only Y is fixed at 200mm, per the recipe).

    Returns (upper_outboard, lower_outboard, toe_link_inboard,
    toe_link_outboard, axle_inboard, axle_outboard)."""
    delta = kpi_deg - BASELINE_KPI_DEG
    upper_outboard = _rotate_about_pivot(_UPPER_OUTBOARD_CAD, LOWER_OUTBOARD, delta)
    axle_inboard = _rotate_about_pivot(_AXLE_INBOARD_CAD, LOWER_OUTBOARD, delta)
    axle_outboard = _rotate_about_pivot(_AXLE_OUTBOARD_CAD, LOWER_OUTBOARD, delta)
    if not TOE_LINK_IS_PLACEHOLDER:
        toe_link_outboard_rotated = _rotate_about_pivot(TOE_LINK_OUTBOARD_RAW, LOWER_OUTBOARD, delta)

    wheel_centre = _wheel_center_from_axle(axle_inboard, axle_outboard)
    camber_delta = _camber_correction_delta(axle_inboard, axle_outboard, TARGET_CAMBER_DEG)
    axle_inboard = _rotate_about_pivot(axle_inboard, wheel_centre, camber_delta)
    axle_outboard = _rotate_about_pivot(axle_outboard, wheel_centre, camber_delta)

    shift_y = _scrub_shift_y(upper_outboard, LOWER_OUTBOARD, axle_inboard, axle_outboard,
                              TARGET_SCRUB_RADIUS_MM)
    lower_outboard = (LOWER_OUTBOARD[0], LOWER_OUTBOARD[1] + shift_y, LOWER_OUTBOARD[2])
    upper_outboard = (upper_outboard[0], upper_outboard[1] + shift_y, upper_outboard[2])

    if TOE_LINK_IS_PLACEHOLDER:
        toe_link_outboard = (lower_outboard[0] - 50.0, lower_outboard[1], lower_outboard[2])
        toe_link_inboard = (lower_outboard[0] - 50.0, 200.0, lower_outboard[2])
    else:
        toe_link_outboard = toe_link_outboard_rotated
        toe_link_inboard = TOE_LINK_INBOARD_RAW

    return upper_outboard, lower_outboard, toe_link_inboard, toe_link_outboard, axle_inboard, axle_outboard


# Z capped at the upright's height at KPI=0 (vertical - the tallest it ever
# is across the swept KPI range); infeasible (Z, KPI) pairs are rejected
# per-candidate in build_geometry(). Same formula/ranges as the front script.
_Z_CEILING = LOWER_OUTBOARD[2] + FV_UPRIGHT_LENGTH
Z_RANGE = np.arange(240.0, _Z_CEILING + 1e-9, 10.0)
KPI_RANGE = np.arange(5.0, 10.0 + 1e-9, 1.0)

# ===========================================================================
# Rear-specific scoring targets (see module docstring)
# name -> (accept_min, accept_max, optimal_lo, optimal_hi)
# ===========================================================================

TARGETS_REAR = {
    "roll_centre_z_static":     (25.0, 55.0, 41.0, 41.0),
    "camber_gain_bump_per25mm": sc.TARGETS_FULL["camber_gain_bump_per25mm"],   # same as front
    "camber_gain_roll_pct":     sc.TARGETS_FULL["camber_gain_roll_pct"],       # same as front
    "kpi_static":                sc.TARGETS_FULL["kpi_static"],                # same as front
    "scrub_radius_static":       sc.TARGETS_FULL["scrub_radius_static"],       # same as front
    "roll_steer_front":          (0.02, 0.03, 0.02, 0.03),   # stabilising toe-in per deg roll
    # camber_static is Tier 4 here too: checked (PASS/FAIL) but not weighted -
    # it's driven directly by build_upright()'s camber-correction rotation.
    "camber_static":             sc.TARGETS_FULL["camber_static"],
    # bump_steer_max is intentionally NOT a target here at all (see module
    # docstring: no rack, no rack-driven bump steer to score) - it's still
    # in every result row as a plain reported column, just never banded.
}

WEIGHTS_REAR = {
    k: sc.WEIGHTS_FULL[k] for k in
    ("roll_centre_z_static", "camber_gain_bump_per25mm", "camber_gain_roll_pct",
     "kpi_static", "scrub_radius_static", "roll_steer_front")
    # camber_static: Tier 4, no weight (same pattern as front).
    # bump_steer_max: not scored at all for the rear (see module docstring).
}


# ===========================================================================
# Geometry builder (axle scope, solver schema)
# ===========================================================================

BASE_AXLE = {
    "type": "double_wishbone",
    "scope": "axle",
    "name": "rear_mount_kpi_sweep",
    "version": "0.0.1",
    "units": "millimeters",
    "vehicle_config": {"cg_position": sc.pt(CG_POSITION), "wheelbase": WHEELBASE},
    "axle_config": {
        "axle_position": AXLE_POSITION,
        "steering": {"type": "none"},
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
            "toe_link_inboard": sc.pt((0, 0, 0)),
            "toe_link_outboard": sc.pt((0, 0, 0)),
            "axle_inboard": sc.pt((0, 0, 0)),
            "axle_outboard": sc.pt((0, 0, 0)),
        }
    },
}


def build_geometry(z_up: float, kpi_deg: float) -> dict:
    """Place the upper chassis mount at height z_up and rotate the upright
    rigid body to kpi_deg (degrees from vertical, front view)."""
    upper_outboard, lower_outboard, toe_link_inboard, toe_link_outboard, axle_inboard, axle_outboard = \
        build_upright(kpi_deg)

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
    hp["toe_link_inboard"] = sc.pt(toe_link_inboard)
    hp["toe_link_outboard"] = sc.pt(toe_link_outboard)
    hp["axle_inboard"] = sc.pt(axle_inboard)
    hp["axle_outboard"] = sc.pt(axle_outboard)
    return g


def write_cad_schema_geometry(path: Path, z_up: float, kpi_deg: float) -> None:
    """Write the winning (z_up, kpi_deg) geometry in the SAME schema as
    configs/uct_rear_geometry_generated.yaml (flat hardpoints, config
    block). All 10 hardpoints are written explicitly (not just whatever the
    source file already had), so the output is self-contained even though
    the source is missing 4 of them."""
    upper_outboard, lower_outboard, toe_link_inboard, toe_link_outboard, axle_inboard, axle_outboard = \
        build_upright(kpi_deg)

    out = copy.deepcopy(_GEOMETRY)
    hp = out["hardpoints"]
    hp["upper_wishbone_inboard_front"] = sc.pt((UPPER_INBOARD_X_FRONT, CHASSIS_MOUNT_Y_FRONT, z_up))
    hp["upper_wishbone_inboard_rear"] = sc.pt((UPPER_INBOARD_X_REAR, CHASSIS_MOUNT_Y_REAR, z_up))
    hp["upper_wishbone_outboard"] = sc.pt(upper_outboard)
    hp["lower_wishbone_outboard"] = sc.pt(lower_outboard)
    hp["toe_link_inboard"] = sc.pt(toe_link_inboard)
    hp["toe_link_outboard"] = sc.pt(toe_link_outboard)
    hp["axle_inboard"] = sc.pt(axle_inboard)
    hp["axle_outboard"] = sc.pt(axle_outboard)
    out["name"] = "UCT Racing FSA rear left - mount/KPI sweep winner"

    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)


# ===========================================================================
# Rear-specific motion sweeps and candidate evaluation
# (own copies, not _sweep_common.py's BUMP_SWEEP/ROLL_SWEEP/evaluate_candidate -
# see module docstring for why: those pin trackrod_inboard, which doesn't
# exist on a toe-link corner. Travel amounts/step counts are still pulled
# from _sweep_common.py so they stay in sync with the front script.)
# ===========================================================================

def _axle_sweep_2target(steps: int, left_z=(0.0, 0.0), right_z=(0.0, 0.0)) -> dict:
    """Axle-scope motion sweep spec with only the two wheel_center Z
    targets - no 3rd (rack/actuator) target, since the toe link's inboard
    point has no free DOF to pin (verified empirically - a 3rd target
    referencing trackrod_inboard is rejected outright, and omitting it
    entirely solves cleanly)."""
    return {
        "version": 1,
        "steps": steps,
        "targets": [
            {"point": "wheel_center", "side": "left", "direction": {"axis": "z"},
             "mode": "relative", "start": left_z[0], "stop": left_z[1]},
            {"point": "wheel_center", "side": "right", "direction": {"axis": "z"},
             "mode": "relative", "start": right_z[0], "stop": right_z[1]},
        ],
    }


BUMP_SWEEP_REAR = _axle_sweep_2target(sc.MOTION_STEPS, left_z=(-sc.BUMP_TRAVEL, sc.BUMP_TRAVEL),
                                       right_z=(-sc.BUMP_TRAVEL, sc.BUMP_TRAVEL))
ROLL_SWEEP_REAR = _axle_sweep_2target(sc.MOTION_STEPS, left_z=(-sc.ROLL_TRAVEL, sc.ROLL_TRAVEL),
                                       right_z=(sc.ROLL_TRAVEL, -sc.ROLL_TRAVEL))


def evaluate_candidate_rear(geometry: dict) -> dict:
    """Rear equivalent of _sweep_common.py's evaluate_candidate() - same
    metric extraction, same early-termination heuristic, just using
    BUMP_SWEEP_REAR/ROLL_SWEEP_REAR (2-target, no rack pin) instead of the
    shared BUMP_SWEEP/ROLL_SWEEP (which target trackrod_inboard and can't
    solve a toe-link corner)."""
    bump = sc.run_sweep(geometry, BUMP_SWEEP_REAR)
    mid = len(bump) // 2

    static = bump[mid]

    heave_mm = [r.get("heave") for r in bump]
    camber_bump = [r.get("camber_left") for r in bump]
    rc_z_bump = [r.get("roll_center_z") for r in bump]
    camber_gain_heave = sc.slope(heave_mm, camber_bump)
    camber_gain_bump_per25mm = camber_gain_heave * 25.0 if camber_gain_heave is not None else None
    rc_migration_heave = sc.slope(heave_mm, rc_z_bump)

    hp_left = geometry["hardpoints"]["left"]
    mount = sc.hp(hp_left, "upper_wishbone_inboard_front")
    outboard = sc.hp(hp_left, "upper_wishbone_outboard")
    upper_arm_length = float(np.linalg.norm(np.array(outboard) - np.array(mount)))

    bump_steer_max = sc.span([r.get("roadwheel_angle_left") for r in bump])

    if bump_steer_max is not None and bump_steer_max > 0.5:
        return {
            "camber_static": static.get("camber_left"),
            "kpi_static": static.get("kpi_left"),
            "caster_static": static.get("caster_left"),
            "scrub_radius_static": static.get("scrub_radius_left"),
            "mechanical_trail_static": static.get("mechanical_trail_left"),
            "roll_centre_z_static": static.get("roll_center_z"),
            "bump_steer_max": bump_steer_max,
            "camber_gain_bump_per25mm": camber_gain_bump_per25mm,
            "camber_gain_roll_pct": np.nan,
            "roll_steer_front": np.nan,
            "rc_migration_roll": np.nan,
            "rc_migration_heave": rc_migration_heave,
            "rc_migration_roll_y": np.nan,
            "fvsa_length_left": static.get("fvsa_length_left"),
            "upper_arm_length": upper_arm_length,
            "weighted_cost": sum(WEIGHTS_REAR.values()) * sc.MAX_NORMALIZED_ERROR,
        }

    roll = sc.run_sweep(geometry, ROLL_SWEEP_REAR)

    roll_deg = [r.get("roll") for r in roll]
    camber_l = [r.get("camber_left") for r in roll]
    rc_y_r = [r.get("roll_center_y") for r in roll]
    rc_z_r = [r.get("roll_center_z") for r in roll]
    roadwheel_l = [r.get("roadwheel_angle_left") for r in roll]

    dcam_droll = sc.slope(roll_deg, camber_l)
    camber_gain_roll_pct = (1.0 - abs(dcam_droll)) if dcam_droll is not None else None
    rc_migration_roll = sc.slope(roll_deg, rc_z_r)
    rc_migration_roll_y = sc.span(rc_y_r)
    roll_steer_front = sc.slope(roll_deg, roadwheel_l)

    return {
        "camber_static": static.get("camber_left"),
        "kpi_static": static.get("kpi_left"),
        "caster_static": static.get("caster_left"),
        "scrub_radius_static": static.get("scrub_radius_left"),
        "mechanical_trail_static": static.get("mechanical_trail_left"),
        "roll_centre_z_static": static.get("roll_center_z"),
        "bump_steer_max": bump_steer_max,
        "camber_gain_bump_per25mm": camber_gain_bump_per25mm,
        "camber_gain_roll_pct": camber_gain_roll_pct,
        "roll_steer_front": roll_steer_front,
        "rc_migration_roll": rc_migration_roll,
        "rc_migration_heave": rc_migration_heave,
        "rc_migration_roll_y": rc_migration_roll_y,
        "fvsa_length_left": static.get("fvsa_length_left"),
        "upper_arm_length": upper_arm_length,
    }


def evaluate(args) -> dict:
    z_up, kpi_deg = args
    out = {"z_up": z_up, "kpi_deg": kpi_deg, "feasible": False, "error": ""}
    try:
        geometry = build_geometry(z_up, kpi_deg)
        metrics = evaluate_candidate_rear(geometry)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    out["feasible"] = True
    out.update(metrics)
    out["weighted_cost"] = sc.weighted_cost(out, TARGETS_REAR, WEIGHTS_REAR)
    return out


# ===========================================================================
# Drivers
# ===========================================================================

def smoke() -> None:
    print_placeholder_warnings()
    verify_track_width()
    z, kpi = float(np.median(Z_RANGE)), float(np.median(KPI_RANGE))
    print(f"\nsmoke test at upper mount z={z:.0f}, KPI={kpi:.1f} deg")
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
    bump = sc.run_sweep(geometry, BUMP_SWEEP_REAR)
    roll = sc.run_sweep(geometry, ROLL_SWEEP_REAR)
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
    fig.suptitle(f"Rear upper mount z={z:.0f}, KPI={kpi:.1f} deg   top row: bump    bottom row: roll")
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
          f"(against {len(WEIGHTS_REAR)} weighted targets):\n")
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
         ("roll_steer_front", "Rear roll steer (deg/deg)", "RdBu_r"),
         ("rc_migration_roll", "RC migration in roll (mm/deg)", "magma_r"),
         ("rc_migration_roll_y", "Lateral RC excursion in roll (mm)", "magma_r"),
         ("weighted_cost", "Weighted cost (lower better)", "YlOrRd_r")],
        "Roll behaviour - rear Z x KPI sweep", "rear_mount_kpi_heatmaps_roll.png", (16, 9),
    )
    heatmap_fig(
        [("kpi_static", "KPI (deg) - sanity check vs input", "viridis"),
         ("scrub_radius_static", "Scrub radius (mm)", "RdBu_r")],
        "KPI and scrub radius - rear Z x KPI sweep", "rear_mount_kpi_heatmaps_kpi_scrub.png", (10, 4.5),
    )
    heatmap_fig(
        [("bump_steer_max", "Bump steer, max toe change (deg) - reported only, not scored", "RdYlGn_r"),
         ("camber_gain_bump_per25mm", "Camber gain in bump (deg/25mm)", "RdBu_r"),
         ("rc_migration_heave", "RC migration in heave (mm/mm)", "magma_r")],
        "Bump behaviour - rear Z x KPI sweep", "rear_mount_kpi_heatmaps_bump.png", (15, 4.5),
    )
    heatmap_fig(
        [("upper_arm_length", "Upper wishbone front-leg length (mm)", "viridis"),
         ("fvsa_length_left", "Front-view swing arm length (mm)", "viridis")],
        "How arm lengths change across the sweep - rear Z x KPI", "rear_arm_lengths_vs_zkpi.png", (10, 4.5),
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
    out_path = OUTPUT_DIR / "rear_mount_kpi_best_geometry.png"
    result = visualize_geometry(suspension, out_path)
    print(f"wrote {result.output_path}  (contact patch on ground: {result.contact_patch_on_ground})")

    detail(z, kpi, prefix=str(OUTPUT_DIR / "rear_mount_kpi_best"),
           curves_name=str(OUTPUT_DIR / "rear_mount_kpi_best_vs_travel.png"))

    geom_out = OUTPUT_DIR / "best_zkpi_geometry_rear.yaml"
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
    p.add_argument("--out", default=str(OUTPUT_DIR / "rear_mount_kpi_results.csv"))
    a = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if a.smoke:
        smoke(); return
    if a.detail:
        print_placeholder_warnings()
        verify_track_width()
        print()
        detail(a.detail[0], a.detail[1]); return
    if a.render_best:
        print_placeholder_warnings()
        verify_track_width()
        print()
        render_best(a.out); return

    print_placeholder_warnings()
    verify_track_width()
    print()

    # Two-pass grid search: a coarse pass to find the promising (Z, KPI)
    # region, then a fine pass confined to that region - same approach as
    # the front script.
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
