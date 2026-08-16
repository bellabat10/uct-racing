# RECOMMENDATION
- Mount height Z: 330 mm
- KPI angle: 6.0 degrees
- Caster: 7.0 degrees
- Axis offset: 18.6 mm (rearward)
- Mechanical trail: 15.0 mm
- Upper ball joint (p) shifts 46.0 mm from current CAD position
- Lower ball joint (q) shifts 5.9 mm from current CAD position

**Both wishbones' outboard ball joints move** - not just the upper one. The caster/trail refinement (Stage B) repositions both the upper and lower ball joints along the newly-oriented kingpin axis.

## Target check (full 9, against confirmatory re-solve)

7/10 targets pass. Weighted cost: 10.353.

## Drift check (Stage A prediction vs Stage C confirmatory re-solve)

                  metric  stage_a_predicted  stage_c_actual   delta  tolerance  flagged
          bump_steer_max             8.3339          6.9912 -1.3427     0.5000     True
    camber_gain_roll_pct             0.8033          0.5410 -0.2623     0.0500     True
    roll_centre_z_static            33.8782         32.8655 -1.0127     5.0000    False
              kpi_static             6.0000          6.0289  0.0289     0.5000    False
     scrub_radius_static             9.9918          9.8524 -0.1394     2.0000    False
        roll_steer_front             1.5119          1.2709 -0.2410     0.0500     True
camber_gain_bump_per25mm            -0.4493         -1.0497 -0.6004     0.1000     True
           camber_static            -3.5000         -3.5000 -0.0000     0.5000    False

4 metric(s) drifted beyond tolerance - review before locking in.

CHECK: does this shift fit inside the upright and clear the wheel rim?
Take these numbers to the CAD model before locking anything in. Roll centre, bump steer and motion ratio have been re-checked by this stage's confirmatory solve, but the full optimiser (all three stages together) should be re-run once hardpoints actually move in CAD, per the note in scripts/caster_trail_analysis.py.
