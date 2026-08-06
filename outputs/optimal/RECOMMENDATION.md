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

5/9 targets pass. Weighted cost: 10.853.

## Drift check (Stage A prediction vs Stage C confirmatory re-solve)

                  metric  stage_a_predicted  stage_c_actual   delta  tolerance  flagged
          bump_steer_max             7.9267          6.7007 -1.2260     0.5000     True
    camber_gain_roll_pct             0.8093          0.5548 -0.2545     0.0500     True
    roll_centre_z_static            33.0044         31.9811 -1.0232     5.0000    False
              kpi_static             6.0000          6.0289  0.0289     0.5000    False
     scrub_radius_static            -1.3909         -1.5300 -0.1391     2.0000    False
        roll_steer_front             1.4524          1.2307 -0.2217     0.0500     True
camber_gain_bump_per25mm            -0.4308         -1.0072 -0.5763     0.1000     True

4 metric(s) drifted beyond tolerance - review before locking in.

CHECK: does this shift fit inside the upright and clear the wheel rim?
Take these numbers to the CAD model before locking anything in. Roll centre, bump steer and motion ratio have been re-checked by this stage's confirmatory solve, but the full optimiser (all three stages together) should be re-run once hardpoints actually move in CAD, per the note in scripts/caster_trail_analysis.py.
