"""
UCT Racing FSA - SolidWorks to ISO 8855 Coordinate Converter

Reads solidworks_points.yaml, applies the axis swap and origin translation,
and writes solver-ready geometry.yaml files for front and rear corners.

SolidWorks convention:  X backward, Y up, Z right
ISO 8855 convention:    x forward,  y left, z up

Axis swap:    iso_x = -SW_X,  iso_y = -SW_Z,  iso_z = SW_Y
Origin:       translated so the wheel centre lands at the correct world-frame
              position (x = axle ref, y = half-track, z = tyre free radius).

Usage:
    python configs/convert_coordinates.py
"""

import sys
from pathlib import Path

import yaml


# ── Tyre geometry ──────────────────────────────────────────────────────────

def compute_free_radius(section_width_mm, aspect_ratio, rim_diameter_inches):
    """Compute the tyre free radius in mm."""
    rim_radius = (rim_diameter_inches * 25.4) / 2.0
    sidewall = section_width_mm * aspect_ratio
    return rim_radius + sidewall


# ── Coordinate transforms ─────────────────────────────────────────────────

def sw_to_iso_axes(point_sw):
    """Apply the axis swap only (no translation)."""
    X, Y, Z = point_sw
    return [-X, Z, Y]


def compute_translation(wheel_center_sw, half_track_mm, free_radius_mm, axle_x_iso):
    """
    Compute the translation vector that places the wheel centre at the
    correct world-frame position after the axis swap.

    Parameters
    ----------
    wheel_center_sw : list of 3 floats
        Wheel centre in SolidWorks coordinates [X, Y, Z].
    half_track_mm : float
        Half-track width (lateral distance from centreline to wheel centre).
    free_radius_mm : float
        Tyre free radius (height of wheel centre above ground).
    axle_x_iso : float
        Target x position for this axle in ISO frame (0 for front, -wheelbase
        for rear).

    Returns
    -------
    list of 3 floats
        Translation vector [tx, ty, tz] to apply after axis swap.
    """
    swapped = sw_to_iso_axes(wheel_center_sw)
    # Left-side corner: positive ISO y
    target = [axle_x_iso, half_track_mm, free_radius_mm]
    return [target[i] - swapped[i] for i in range(3)]


def transform_point(point_sw, translation):
    """Full transform: axis swap + translation."""
    swapped = sw_to_iso_axes(point_sw)
    return [round(swapped[i] + translation[i], 3) for i in range(3)]


# ── YAML reading helpers ──────────────────────────────────────────────────

def get_point(entry):
    """Extract the position from a point entry (handles both fixed and bar)."""
    if isinstance(entry, dict):
        # Bar-type or box-type point: use initial_guess
        if 'initial_guess' in entry:
            return entry['initial_guess']
        # Fixed point: use position
        if 'position' in entry:
            return entry['position']
    # Direct list (like wheel_center)
    if isinstance(entry, list):
        return entry
    return None


# ── Geometry YAML writer ──────────────────────────────────────────────────

def build_geometry_yaml(corner_name, hardpoints_iso, vehicle_config, is_steered):
    """
    Build a geometry YAML dict matching the solver's schema.

    Parameters
    ----------
    corner_name : str
        'front' or 'rear'.
    hardpoints_iso : dict
        Point name -> [x, y, z] in ISO 8855 mm.
    vehicle_config : dict
        Vehicle configuration from the input file.
    is_steered : bool
        True for rack-steered (front), False for fixed toe link (rear).
    """
    tire = vehicle_config['tire']
    wheelbase = vehicle_config['wheelbase']

    # CG position in ISO frame
    # For now, place CG at mid-wheelbase on the centreline at CG height.
    # The exact longitudinal CG position needs confirming from the team.
    cg_x = -wheelbase / 2.0  # placeholder: mid-wheelbase
    cg_z = vehicle_config['cg_height']

    steering_type = 'rack' if is_steered else 'none'

    # Build the heading link point names
    if is_steered:
        heading_inboard = 'trackrod_inboard'
        heading_outboard = 'trackrod_outboard'
    else:
        heading_inboard = 'toe_link_inboard'
        heading_outboard = 'toe_link_outboard'

    geometry = {
        'name': f'UCT Racing FSA {corner_name} left',
        'version': '1.0.0',
        'units': 'millimeters',
        'type': 'double_wishbone',
        'scope': 'corner',
        'side': 'left',
        'actuation': {
            'type': 'direct',
            'mount': 'lower_wishbone',
        },
        'spring': {'type': 'none'},
        'config': {
            'steering': {'type': steering_type},
            'wheel': {
                'offset': vehicle_config['wheel'].get('offset', 0) or 0,
                'tire': {
                    'section_width': tire['section_width'],
                    'aspect_ratio': tire['aspect_ratio'],
                    'rim_diameter': tire['rim_diameter'],
                },
            },
            'cg_position': {
                'x': round(cg_x, 1),
                'y': 0,
                'z': round(cg_z, 1),
            },
            'wheelbase': wheelbase,
        },
        'hardpoints': {},
    }

    # Map our point names to the solver's expected names
    solver_names = [
        'lower_wishbone_inboard_front',
        'lower_wishbone_inboard_rear',
        'lower_wishbone_outboard',
        'upper_wishbone_inboard_front',
        'upper_wishbone_inboard_rear',
        'upper_wishbone_outboard',
        heading_inboard,
        heading_outboard,
        'axle_inboard',
        'axle_outboard',
    ]

    missing = []
    for name in solver_names:
        if name in hardpoints_iso and hardpoints_iso[name] is not None:
            x, y, z = hardpoints_iso[name]
            geometry['hardpoints'][name] = {'x': x, 'y': y, 'z': z}
        else:
            missing.append(name)

    return geometry, missing


# ── Sanity checks ─────────────────────────────────────────────────────────

def distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5


def print_sanity_checks(corner_name, hardpoints_iso, free_radius):
    """Print quick verification numbers."""
    print(f'\n  Sanity checks ({corner_name}):')

    # Check wheel centre height
    wc = hardpoints_iso.get('_wheel_center')
    if wc:
        print(f'    Wheel centre:  x={wc[0]:.1f}  y={wc[1]:.1f}  z={wc[2]:.1f}')
        print(f'    Contact patch Z (should be ~0): {wc[2] - free_radius:.1f}')

    # Check link lengths
    length_checks = [
        ('Upper wishbone front leg',
         'upper_wishbone_inboard_front', 'upper_wishbone_outboard'),
        ('Upper wishbone rear leg',
         'upper_wishbone_inboard_rear', 'upper_wishbone_outboard'),
        ('Lower wishbone front leg',
         'lower_wishbone_inboard_front', 'lower_wishbone_outboard'),
        ('Lower wishbone rear leg',
         'lower_wishbone_inboard_rear', 'lower_wishbone_outboard'),
        ('Upright (p to q)',
         'upper_wishbone_outboard', 'lower_wishbone_outboard'),
    ]

    for label, name_a, name_b in length_checks:
        pa = hardpoints_iso.get(name_a)
        pb = hardpoints_iso.get(name_b)
        if pa is not None and pb is not None:
            d = distance(pa, pb)
            print(f'    {label}: {d:.1f} mm')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).parent
    input_path = script_dir / 'solidworks_points.yaml'

    if not input_path.exists():
        print(f'ERROR: {input_path} not found.')
        sys.exit(1)

    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)

    # ── Vehicle config ────────────────────────────────────────────────
    vehicle = data['vehicle']
    tire = vehicle['tire']
    wheelbase = vehicle['wheelbase']

    free_radius = compute_free_radius(
        tire['section_width'], tire['aspect_ratio'], tire['rim_diameter']
    )
    print(f'Tyre free radius: {free_radius:.1f} mm')
    print(f'Wheelbase: {wheelbase} mm')

    # ── Process each corner ───────────────────────────────────────────
    corners = {
        'front': {'axle_x': 0.0, 'is_steered': True},
        'rear':  {'axle_x': -wheelbase, 'is_steered': False},
    }

    for corner_name, corner_cfg in corners.items():
        corner_data = data.get(corner_name)
        if corner_data is None:
            print(f'\n  Skipping {corner_name}: not found in input file.')
            continue

        print(f'\n{"="*60}')
        print(f'  Processing {corner_name} corner')
        print(f'{"="*60}')

        # Get wheel centre and half-track for this corner
        wc_sw = corner_data.get('wheel_center')
        half_track = corner_data.get('half_track')

        if wc_sw is None or half_track is None:
            print(f'  ERROR: wheel_center or half_track missing for {corner_name}.')
            continue

        # Compute translation
        translation = compute_translation(
            wc_sw, half_track, free_radius, corner_cfg['axle_x']
        )

        print(f'  Wheel centre (SW): {wc_sw}')
        print(f'  Half-track: {half_track} mm')
        print(f'  Translation (after axis swap): '
              f'[{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]')

        # Determine which point names to look for
        if corner_cfg['is_steered']:
            heading_names = ['trackrod_inboard', 'trackrod_outboard']
        else:
            heading_names = ['toe_link_inboard', 'toe_link_outboard']

        point_names = [
            'lower_wishbone_inboard_front',
            'lower_wishbone_inboard_rear',
            'lower_wishbone_outboard',
            'upper_wishbone_inboard_front',
            'upper_wishbone_inboard_rear',
            'upper_wishbone_outboard',
            *heading_names,
            'axle_inboard',
            'axle_outboard',
        ]

        # Transform each point
        hardpoints_iso = {}
        print(f'\n  Transformed hardpoints (ISO 8855):')
        print(f'  {"Point":<35s} {"x":>8s} {"y":>8s} {"z":>8s}')
        print(f'  {"-"*35} {"-"*8} {"-"*8} {"-"*8}')

        for name in point_names:
            entry = corner_data.get(name)
            if entry is None:
                print(f'  {name:<35s}   -- not in input file --')
                continue

            point_sw = get_point(entry)
            if point_sw is None:
                print(f'  {name:<35s}   -- null (needs measuring) --')
                hardpoints_iso[name] = None
                continue

            point_iso = transform_point(point_sw, translation)
            hardpoints_iso[name] = point_iso
            print(f'  {name:<35s} {point_iso[0]:8.2f} {point_iso[1]:8.2f} {point_iso[2]:8.2f}')

        # Transform wheel centre too (for sanity checks, not a hardpoint)
        hardpoints_iso['_wheel_center'] = transform_point(wc_sw, translation)

        # Sanity checks
        print_sanity_checks(corner_name, hardpoints_iso, free_radius)

        # Build and write geometry YAML
        geometry, missing = build_geometry_yaml(
            corner_name, hardpoints_iso, vehicle, corner_cfg['is_steered']
        )

        if missing:
            print(f'\n  WARNING: {len(missing)} points still missing - '
                  f'solver will reject this geometry:')
            for m in missing:
                print(f'    - {m}')

        output_name = f'uct_{corner_name}_geometry_generated.yaml'
        output_path = script_dir / output_name
        with open(output_path, 'w') as f:
            # Write a header comment
            f.write(f'# UCT Racing FSA - {corner_name} left corner\n')
            f.write(f'# Generated by convert_coordinates.py from solidworks_points.yaml\n')
            f.write(f'# Coordinate system: ISO 8855 (x fwd, y left, z up), millimetres\n')
            if missing:
                f.write(f'# INCOMPLETE: {len(missing)} hardpoints still null\n')
            f.write(f'#\n')
            yaml.dump(geometry, f, default_flow_style=False, sort_keys=False)

        print(f'\n  Wrote: {output_path}')
        if missing:
            print(f'  (incomplete - fill in null points in solidworks_points.yaml and re-run)')

    print(f'\n{"="*60}')
    print(f'  Done.')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()