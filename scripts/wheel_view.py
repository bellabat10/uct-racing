"""
Annotated side-view (XZ) + front-view (YZ) diagram of one corner's kingpin
axis and wheel, with caster / KPI / camber / scrub radius / mechanical trail
labeled directly on the drawing.

Exists because the submodule's 4-view linkage renderer
(kinematics.cli.visualization.api.visualize_geometry) scales every view to
the WHOLE suspension assembly's bounding box, so a few degrees of caster or
KPI tilt over a ~150mm kingpin axis is sub-pixel and invisible - and it has
no metric annotations at all. This module draws just the upright + wheel,
zoomed to a fixed margin around them, with the solver's own metric values
labeled on top - no independent recomputation, so the diagram always
matches whatever the solver actually reports for this state.

Coordinates: ISO 8855 - X forward, Y left, Z up, millimetres.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle

from kinematics.core.enums import Axis
from kinematics.core.metrics.angles import (
    calculate_camber,
    calculate_caster,
    calculate_kpi,
    calculate_toe,
)
from kinematics.core.metrics.context import MetricContext
from kinematics.core.metrics.steering_geometry import (
    calculate_mechanical_trail,
    calculate_scrub_radius,
)

COLOR_AXIS = "#ff6f4f"      # coral - the kingpin axis itself, the "headline" geometry
COLOR_WHEEL = "#1abc9c"     # teal - wheel / dimension brackets, "supporting measurement"
COLOR_REF = "#888888"       # gray dashed - vertical references
MARGIN_MM = 70.0


def _xy(point, axis_a: Axis, axis_b: Axis) -> tuple[float, float]:
    return float(point[axis_a]), float(point[axis_b])


def _extend(q_xy, p_xy, past_q: float, past_p: float):
    """Return (bottom, top) points along the q->p line, extended `past_q`
    beyond q (toward/through the ground) and `past_p` beyond p."""
    dx, dz = p_xy[0] - q_xy[0], p_xy[1] - q_xy[1]
    length = float(np.hypot(dx, dz))
    ux, uz = dx / length, dz / length
    bottom = (q_xy[0] - ux * past_q, q_xy[1] - uz * past_q)
    top = (p_xy[0] + ux * past_p, p_xy[1] + uz * past_p)
    return bottom, top


_LABEL_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5)
_SUMMARY_BBOX = dict(facecolor="white", edgecolor="#888888", alpha=0.8, pad=6)


def _angle_arc(ax, pivot_xy, ref_angle_deg: float, axis_angle_deg: float,
               radius: float, label: str, color: str) -> None:
    """Draw an angle arc at `pivot_xy` from the vertical reference to the
    axis direction, with a text label pushed well clear of both lines."""
    theta1, theta2 = sorted((ref_angle_deg, axis_angle_deg))
    ax.add_patch(Arc(pivot_xy, 2 * radius, 2 * radius, angle=0,
                      theta1=theta1, theta2=theta2, color=color, linewidth=1.5))
    # Label offset to the side (away from the near-vertical lines it sits
    # between), not along the arc's own midpoint angle - for a small angle
    # (a few degrees) that midpoint sits almost exactly on the axis line.
    mid = np.radians((theta1 + theta2) / 2.0)
    side = 1.0 if np.cos(mid) >= 0 else -1.0
    lx = pivot_xy[0] + side * (radius + 45)
    lz = pivot_xy[1] + (radius + 15) * np.sin(mid)
    ax.text(lx, lz, label, color=color, fontsize=9, ha="center", va="center",
             fontweight="bold", bbox=_LABEL_BBOX, zorder=6)


def _dimension_bracket(ax, y: float, x0: float, x1: float, label: str, color: str) -> None:
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=1.3))
    ax.text((x0 + x1) / 2.0, y - 6, label, color=color, fontsize=9,
             ha="center", va="top", fontweight="bold", bbox=_LABEL_BBOX, zorder=6)


def _point_marker(ax, xy: tuple[float, float], label: str, coords_mm: tuple[float, float, float],
                   color: str, offset: tuple[float, float], ha: str = "left") -> None:
    """Mark a hardpoint (P or Q) with a dot and an adjacent
    'LABEL (x, y, z) mm' tag giving its full 3D ISO 8855 coordinates."""
    ax.plot(xy[0], xy[1], marker="o", ms=5, color=color, zorder=7,
            markeredgecolor="black", markeredgewidth=0.6)
    ax.text(xy[0] + offset[0], xy[1] + offset[1],
            f"{label} ({coords_mm[0]:.1f}, {coords_mm[1]:.1f}, {coords_mm[2]:.1f})",
            color=color, fontsize=7.5, ha=ha, va="center", fontweight="bold",
            bbox=_LABEL_BBOX, zorder=7)


def build_summary_lines(kpi: float, caster: float, scrub: float, trail: float, camber: float,
                         toe: float, roll_centre_z: float | None, weighted_cost: float | None) -> list[str]:
    """The 8 headline metrics, one 'Parameter: value unit' line each - shared
    between wheel_view.png's own text box and any other diagram (e.g. the
    submodule's 4-view linkage render) annotated via annotate_png()."""
    roll_centre_str = f"{roll_centre_z:.1f} mm" if roll_centre_z is not None else "n/a"
    weighted_cost_str = f"{weighted_cost:.3f}" if weighted_cost is not None else "n/a"
    return [
        f"KPI: {kpi:.1f} deg",
        f"Caster: {caster:.1f} deg",
        f"Scrub radius: {scrub:.1f} mm",
        f"Mechanical trail: {trail:.1f} mm",
        f"Static camber: {camber:.2f} deg",
        f"Static toe: {toe:.2f} deg",
        f"Roll centre height: {roll_centre_str}",
        f"Weighted cost: {weighted_cost_str}",
    ]


def annotate_png(png_path: Path, lines: list[str], corner_frac: tuple[float, float] = (0.44, 0.04)) -> None:
    """Overlay a semi-transparent summary box directly onto an
    already-rendered PNG (e.g. the submodule's 4-view linkage render) by
    drawing on the pixels with Pillow - doesn't touch the renderer that
    produced it, so kinematics.cli.visualization.api stays untouched.
    `corner_frac` anchors the box's top-left corner as a fraction of the
    image size; the default sits in the wide blank gutter between the
    linkage renderer's 4 subplots."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("consola.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    pad = 14
    line_gap = 6
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_w = max(b[2] - b[0] for b in bboxes)
    line_h = max(b[3] - b[1] for b in bboxes)
    box_w = text_w + 2 * pad
    box_h = len(lines) * (line_h + line_gap) - line_gap + 2 * pad

    x0 = corner_frac[0] * img.width
    y0 = corner_frac[1] * img.height
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h],
                   fill=(255, 255, 255, 200), outline=(136, 136, 136, 255), width=1)
    y = y0 + pad
    for line in lines:
        draw.text((x0 + pad, y), line, font=font, fill=(0, 0, 0, 255))
        y += line_h + line_gap

    img.save(png_path)
    print(f"annotated {png_path}")


def _set_bounds(ax, xs, zs, margin=MARGIN_MM) -> None:
    """Square (equal-aspect) bounds around the given extents, expanding the
    narrower axis rather than letting matplotlib silently override
    explicit xlim/ylim to force equal aspect."""
    x0, x1 = min(xs) - margin, max(xs) + margin
    z0, z1 = min(zs) - margin, max(zs) + margin
    xc, zc = (x0 + x1) / 2.0, (z0 + z1) / 2.0
    half = max(x1 - x0, z1 - z0) / 2.0
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(zc - half, zc + half)
    ax.set_aspect("equal")


def render(ctx: MetricContext, output_path: Path, dpi: int = 150,
           roll_centre_z: float | None = None, weighted_cost: float | None = None,
           corner_label: str = "front-left") -> dict:
    caster = calculate_caster(ctx)
    kpi = calculate_kpi(ctx)
    camber = calculate_camber(ctx)
    toe = calculate_toe(ctx)
    scrub = calculate_scrub_radius(ctx)
    trail = calculate_mechanical_trail(ctx)

    q, p = ctx.steering_axis_pivots  # (lower, upper)
    p_xyz = (float(p[Axis.X]), float(p[Axis.Y]), float(p[Axis.Z]))
    q_xyz = (float(q[Axis.X]), float(q[Axis.Y]), float(q[Axis.Z]))
    wc = ctx.wheel_center
    cp = ctx.contact_patch_center
    gi = ctx.steering_axis_ground_intersection
    wa = ctx.wheel_axis
    gz = ctx.ground_z
    r = ctx.tire_radius

    fig, (ax_side, ax_front) = plt.subplots(1, 2, figsize=(14, 7), dpi=dpi)

    # =======================================================================
    # Side view (XZ)
    # =======================================================================
    q_xz, p_xz = _xy(q, Axis.X, Axis.Z), _xy(p, Axis.X, Axis.Z)
    wc_xz = _xy(wc, Axis.X, Axis.Z)
    cp_x = float(cp[Axis.X])
    gi_x = float(gi[Axis.X]) if gi is not None else cp_x

    bottom, top = _extend(q_xz, p_xz, past_q=40, past_p=25)
    ax_side.plot([bottom[0], top[0]], [bottom[1], top[1]], color=COLOR_AXIS,
                 linewidth=2.2, zorder=4, label="Kingpin axis")
    ax_side.plot([q_xz[0], q_xz[0]], [q_xz[1], p_xz[1] + 25], "--",
                 color=COLOR_REF, linewidth=1, zorder=2)

    xr = [wc_xz[0] - r, wc_xz[0] + r, gi_x, cp_x, q_xz[0], p_xz[0]]
    zr = [wc_xz[1] - r, wc_xz[1] + r, gz, q_xz[1], p_xz[1]]
    ax_side.plot([min(xr) - 10, max(xr) + 10], [gz, gz], color="k", linewidth=1, zorder=1)
    ax_side.add_patch(Circle(wc_xz, r, fill=False, edgecolor=COLOR_WHEEL, linewidth=2, zorder=3))
    ax_side.plot(cp_x, gz, marker="o", ms=6, color=COLOR_WHEEL, zorder=5)

    axis_angle = float(np.degrees(np.arctan2(p_xz[1] - q_xz[1], p_xz[0] - q_xz[0])))
    _angle_arc(ax_side, q_xz, 90.0, axis_angle, 40.0, f"Caster: {caster:.1f}°", COLOR_AXIS)
    _dimension_bracket(ax_side, gz - 18, gi_x, cp_x, f"Mechanical trail: {trail:.1f} mm", COLOR_WHEEL)
    _point_marker(ax_side, q_xz, "Q", q_xyz, COLOR_AXIS, offset=(-15, -25), ha="right")
    _point_marker(ax_side, p_xz, "P", p_xyz, COLOR_AXIS, offset=(-15, 20), ha="right")

    _set_bounds(ax_side, xr, zr + [gz - 30])
    ax_side.set_xlabel("X - forward (mm)")
    ax_side.set_ylabel("Z - up (mm)")
    ax_side.set_title("Side view (XZ)")
    ax_side.grid(alpha=0.12)

    # =======================================================================
    # Front view (YZ)
    # =======================================================================
    q_yz, p_yz = _xy(q, Axis.Y, Axis.Z), _xy(p, Axis.Y, Axis.Z)
    wc_yz = _xy(wc, Axis.Y, Axis.Z)
    cp_y = float(cp[Axis.Y])
    gi_y = float(gi[Axis.Y]) if gi is not None else cp_y

    bottom, top = _extend(q_yz, p_yz, past_q=40, past_p=25)
    ax_front.plot([bottom[0], top[0]], [bottom[1], top[1]], color=COLOR_AXIS,
                  linewidth=2.2, zorder=4, label="Kingpin axis")
    ax_front.plot([q_yz[0], q_yz[0]], [q_yz[1], p_yz[1] + 25], "--",
                  color=COLOR_REF, linewidth=1, zorder=2)

    # Wheel "disc" as seen from the front: a line through the wheel centre,
    # PERPENDICULAR to the spin axis (wheel_axis), length = 2x tire radius.
    # Its tilt from vertical is exactly the camber angle.
    way, waz = float(wa[Axis.Y]), float(wa[Axis.Z])
    wa_len = float(np.hypot(way, waz))
    perp = (-waz / wa_len, way / wa_len) if wa_len > 1e-9 else (0.0, 1.0)
    disc_bottom = (wc_yz[0] - perp[0] * r, wc_yz[1] - perp[1] * r)
    disc_top = (wc_yz[0] + perp[0] * r, wc_yz[1] + perp[1] * r)
    ax_front.plot([disc_bottom[0], disc_top[0]], [disc_bottom[1], disc_top[1]],
                  color=COLOR_WHEEL, linewidth=3, zorder=3, label="Wheel (front view)")
    ax_front.text(wc_yz[0], disc_top[1] + 10, f"Camber: {camber:.2f}°",
                  color=COLOR_WHEEL, fontsize=9, ha="center", fontweight="bold",
                  bbox=_LABEL_BBOX, zorder=6)

    yr = [disc_bottom[0], disc_top[0], gi_y, cp_y, q_yz[0], p_yz[0]]
    zr = [disc_bottom[1], disc_top[1], gz, q_yz[1], p_yz[1]]
    ax_front.plot([min(yr) - 10, max(yr) + 10], [gz, gz], color="k", linewidth=1, zorder=1)
    ax_front.plot(cp_y, gz, marker="o", ms=6, color=COLOR_WHEEL, zorder=5)

    axis_angle_fv = float(np.degrees(np.arctan2(p_yz[1] - q_yz[1], p_yz[0] - q_yz[0])))
    _angle_arc(ax_front, q_yz, 90.0, axis_angle_fv, 40.0, f"KPI: {kpi:.1f}°", COLOR_AXIS)
    _dimension_bracket(ax_front, gz - 18, gi_y, cp_y, f"Scrub radius: {scrub:.1f} mm", COLOR_WHEEL)
    _point_marker(ax_front, q_yz, "Q", q_xyz, COLOR_AXIS, offset=(12, -25))
    _point_marker(ax_front, p_yz, "P", p_xyz, COLOR_AXIS, offset=(12, 20))

    _set_bounds(ax_front, yr, zr + [gz - 30])
    ax_front.set_xlabel("Y - left (mm)")
    ax_front.set_ylabel("Z - up (mm)")
    ax_front.set_title("Front view (YZ)")
    ax_front.grid(alpha=0.12)

    # =======================================================================
    # Summary text box - headline metrics, off to one side of the linkage
    # =======================================================================
    summary_lines = build_summary_lines(kpi, caster, scrub, trail, camber, toe,
                                         roll_centre_z, weighted_cost)
    ax_front.text(0.97, 0.97, "\n".join(summary_lines), transform=ax_front.transAxes,
                  fontsize=8, family="monospace", ha="right", va="top",
                  bbox=_SUMMARY_BBOX, zorder=7)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.suptitle(
        f"Optimal {corner_label} corner - caster={caster:.1f}°, KPI={kpi:.1f}°, "
        f"camber={camber:.2f}°, scrub={scrub:.1f}mm, trail={trail:.1f}mm",
        fontsize=12, y=0.99,
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {output_path}")

    return {
        "kpi": kpi, "caster": caster, "scrub": scrub, "trail": trail,
        "camber": camber, "toe": toe, "roll_centre_z": roll_centre_z,
        "weighted_cost": weighted_cost, "p_xyz": p_xyz, "q_xyz": q_xyz,
    }
