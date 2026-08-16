# RECOMMENDATION (rear)
- Mount height Z: 335 mm
- KPI angle: 6.0 degrees
- Caster: 7.0 degrees
- Axis offset: 18.6 mm (rearward)
- Mechanical trail: 15.0 mm
- Upper ball joint (p) shifts 4.6 mm from current CAD position
- Lower ball joint (q) shifts 21.7 mm from current CAD position

**Both wishbones' outboard ball joints move** - not just the upper one. The caster/trail refinement (Stage B) repositions both the upper and lower ball joints along the newly-oriented kingpin axis.

## Target check (against confirmatory re-solve, bump steer excluded - see module docstring)

7/9 targets pass. Weighted cost: 9.358.

## Drift check (rear Stage A prediction vs Stage C confirmatory re-solve)

                  metric  stage_a_predicted  stage_c_actual   delta  tolerance  flagged
    roll_centre_z_static            41.4798         39.1732 -2.3066     5.0000    False
camber_gain_bump_per25mm            -0.5356         -0.5662 -0.0307     0.1000    False
    camber_gain_roll_pct                NaN             NaN     NaN     0.0500    False
              kpi_static             6.0000          6.0447  0.0447     0.5000    False
     scrub_radius_static            11.4253         11.2200 -0.2053     2.0000    False
        roll_steer_front                NaN             NaN     NaN     0.0500    False
           camber_static            -3.5000         -3.5000  0.0000     0.5000    False

All metrics within tolerance - no significant drift from Stage A's prediction.

CHECK: does this shift fit inside the upright and clear the wheel rim?
Take these numbers to the CAD model before locking anything in. Roll centre and motion ratio have been re-checked by this stage's confirmatory solve. Bump steer is not scored for the rear (it's a placeholder-toe-link artifact, not real geometry - see scripts/rear_mount_kpi_sweep.py's module docstring), and axle_inboard/axle_outboard/toe_link_inboard/toe_link_outboard are all still PLACEHOLDERS until the real upright/toe-link is measured, per the note in scripts/rear_caster_trail_analysis.py.
