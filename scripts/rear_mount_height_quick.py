"""
Quick, no-solver approximation of the rear upper-wishbone chassis-mount
height (Z only - mirrors the FIRST half of the front's Stage A in
scripts/mount_kpi_sweep.py, minus the KPI sweep and minus the solver).

Why closed-form instead of the real solver: the rear corner is still
missing axle_inboard/axle_outboard and toe_link_inboard/outboard in
configs/solidworks_points.yaml (confirmed - see the "INCOMPLETE: 4
hardpoints still null" header in configs/uct_rear_geometry_generated.yaml),
so `kinematics`/build_suspension() cannot solve this corner at all yet.
Front-view instant-centre and roll-centre height only need the wishbone
hardpoints and the contact patch, both of which ARE known for the rear, so
this sidesteps the missing data entirely - closed-form vector algebra, one
line-line intersection per candidate, no Newton iteration, effectively
instant even swept over a fine grid.

FVIC method (general, non-parallel wishbone axes - same construction the
solver itself uses, see suspension-model's compute_front_view_instant_center):
for each wishbone, parametrise the inboard pivot axis A->B, find the point
on it at the SAME X as the outboard ball joint C, then the wishbone's
front-view line is the one through that point and C (both projected to the
Y-Z plane). FVIC = intersection of the upper and lower wishbone's front-view
lines. Roll centre height = Z where the line from the contact patch through
FVIC crosses the vehicle centreline (Y=0).

Sanity-checked against the front corner's raw CAD geometry (which the real
solver has already scored): this method gives roll_centre_z_static ~50.7mm
vs the solver's 51.57mm - a sub-2% difference, well inside "quick
approximation" territory.

Only the upper wishbone's chassis-mount HEIGHT (Z) is swept here, applied
to both legs (mirrors mount_kpi_sweep.py's convention exactly: same z_up on
both legs, each leg's Y pinned to its own lower-wishbone counterpart's Y,
X unchanged from CAD). KPI/caster are NOT touched - out of scope for this
pass per the user's request; only re-run mount_kpi_sweep-equivalent full
optimisation once the rear's missing hardpoints are measured.

Usage:
    python scripts/rear_mount_height_quick.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

FRONT_OPTIMAL_ROLL_CENTRE_Z = 31.981   # outputs/optimal/optimal_summary.csv, roll_centre_z_static
REAR_ROLL_CENTRE_TARGET = FRONT_OPTIMAL_ROLL_CENTRE_Z + 6.0   # "slightly higher than front"


def load_rear_hardpoints() -> dict:
    with open(CONFIGS_DIR / "uct_rear_geometry_generated.yaml") as f:
        geo = yaml.safe_load(f)
    hp = geo["hardpoints"]
    return {name: np.array([p["x"], p["y"], p["z"]]) for name, p in hp.items()}


def front_view_line_point(inboard_front: np.ndarray, inboard_rear: np.ndarray,
                           outboard: np.ndarray) -> np.ndarray:
    """Point on the inboard pivot axis at the same X as `outboard` (Y,Z only matter)."""
    t = (outboard[0] - inboard_front[0]) / (inboard_rear[0] - inboard_front[0])
    return inboard_front + t * (inboard_rear - inboard_front)


def line_intersect_yz(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """Intersection of line p1-p2 and line p3-p4 in the Y-Z plane."""
    y1, z1 = p1[1], p1[2]
    y2, z2 = p2[1], p2[2]
    y3, z3 = p3[1], p3[2]
    y4, z4 = p4[1], p4[2]
    denom = (y1 - y2) * (z3 - z4) - (z1 - z2) * (y3 - y4)
    y = ((y1 * z2 - z1 * y2) * (y3 - y4) - (y1 - y2) * (y3 * z4 - z3 * y4)) / denom
    z = ((y1 * z2 - z1 * y2) * (z3 - z4) - (z1 - z2) * (y3 * z4 - z3 * y4)) / denom
    return np.array([y, z])


def compute_fvic(hp: dict) -> np.ndarray:
    lower_ref = front_view_line_point(hp["lower_wishbone_inboard_front"],
                                       hp["lower_wishbone_inboard_rear"],
                                       hp["lower_wishbone_outboard"])
    upper_ref = front_view_line_point(hp["upper_wishbone_inboard_front"],
                                       hp["upper_wishbone_inboard_rear"],
                                       hp["upper_wishbone_outboard"])
    return line_intersect_yz(lower_ref, hp["lower_wishbone_outboard"],
                              upper_ref, hp["upper_wishbone_outboard"])


def roll_centre_z(hp: dict, half_track: float) -> float:
    contact_patch = np.array([hp["lower_wishbone_outboard"][0], half_track, 0.0])
    fvic_yz = compute_fvic(hp)
    fvic_3d = np.array([contact_patch[0], fvic_yz[0], fvic_yz[1]])
    # z at y=0 on the line contact_patch -> fvic (Y-Z plane only)
    y1, z1 = contact_patch[1], contact_patch[2]
    y2, z2 = fvic_3d[1], fvic_3d[2]
    t = (0.0 - y1) / (y2 - y1)
    return float(z1 + t * (z2 - z1))


def main() -> None:
    hp = load_rear_hardpoints()
    half_track = 650.0   # solidworks_points.yaml: rear.half_track

    lower_front_y = hp["lower_wishbone_inboard_front"][1]
    lower_rear_y = hp["lower_wishbone_inboard_rear"][1]
    upper_front_x = hp["upper_wishbone_inboard_front"][0]
    upper_rear_x = hp["upper_wishbone_inboard_rear"][0]
    upper_outboard_z = hp["upper_wishbone_outboard"][2]
    lower_z = hp["lower_wishbone_inboard_front"][2]   # ~= lower_wishbone_inboard_rear z too

    print(f"Fixed (unchanged from CAD): lower wishbone Z = {lower_z:.2f} mm, "
          f"upper wishbone outboard Z = {upper_outboard_z:.2f} mm (packaging ceiling)")
    print(f"Front's OPTIMISED static roll centre: {FRONT_OPTIMAL_ROLL_CENTRE_Z:.2f} mm")
    print(f"Rear target (slightly higher): ~{REAR_ROLL_CENTRE_TARGET:.2f} mm\n")

    z_range = np.arange(200.0, upper_outboard_z - 5.0, 2.0)
    results = []
    for z_up in z_range:
        hp_candidate = dict(hp)
        hp_candidate["upper_wishbone_inboard_front"] = np.array([upper_front_x, lower_front_y, z_up])
        hp_candidate["upper_wishbone_inboard_rear"] = np.array([upper_rear_x, lower_rear_y, z_up])
        rc_z = roll_centre_z(hp_candidate, half_track)
        results.append((z_up, rc_z))

    results = np.array(results)
    best_idx = np.argmin(np.abs(results[:, 1] - REAR_ROLL_CENTRE_TARGET))
    best_z, best_rc = results[best_idx]

    print(f"{'Z_up (mm)':>10} {'roll_centre_z (mm)':>20}")
    for z_up, rc_z in results[::5]:
        marker = "  <-- closest to target" if abs(z_up - best_z) < 1e-6 else ""
        print(f"{z_up:10.1f} {rc_z:20.2f}{marker}")

    print(f"\nRecommended rear upper wishbone chassis-mount Z = {best_z:.1f} mm "
          f"-> roll centre = {best_rc:.2f} mm")
    print(f"  upper_wishbone_inboard_front: x={upper_front_x:.2f}  y={lower_front_y:.2f}  z={best_z:.1f}")
    print(f"  upper_wishbone_inboard_rear:  x={upper_rear_x:.2f}  y={lower_rear_y:.2f}  z={best_z:.1f}")

    z_gap_front_leg = best_z - lower_z
    z_gap_rear_leg = best_z - lower_z
    print(f"\nZ distance between wishbones at this mount height: {z_gap_front_leg:.1f} mm "
          f"(both legs, since lower wishbone Z is ~equal front/rear: "
          f"{hp['lower_wishbone_inboard_front'][2]:.2f} / {hp['lower_wishbone_inboard_rear'][2]:.2f})")


if __name__ == "__main__":
    main()
