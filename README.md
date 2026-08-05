# UCT Racing – Suspension Hardpoint Optimisation

Kinematic optimisation of suspension hardpoints, built on the `kinematics`
solver package.

## Repo layout

- `configs/solidworks_points.yaml` — raw hardpoints exported from SolidWorks
- `configs/convert_coordinates.py` — converts SolidWorks points to ISO 8855
- `configs/uct_front_geometry_generated.yaml`, `uct_rear_geometry_generated.yaml`
  — converter output; these are what the scripts actually read
- `suspension-model/` — git submodule, the `kinematics` solver (installed
  editable via `pip install -e .`)
- `iteration1_axle_sweep.py` — current sweep script (see below)
- `run_sweep.py` — earlier one-off sweep using the `kinematics` CLI directly
- `outputs/` — generated CSVs/plots, gitignored (not committed — run the
  scripts locally to regenerate)

## Current iteration: Iteration 1 (axle scope) — upper wishbone chassis mount

Sweeps the upper wishbone's chassis-side mount position and scores each
candidate against kinematic targets (camber compensation, roll centre
height, RC migration, camber gain).

All fixed hardpoints (lower wishbone, upper ball joint, track rod, axle,
CG, tire, wheelbase) are loaded directly from
`configs/uct_front_geometry_generated.yaml` — nothing is hand-typed, so the
sweep can't drift from the CAD.

Constraints baked into the sweep:
- **Y is fixed per leg**, not swept — the upper wishbone's front and rear
  legs each bolt to the same chassis structure as their lower-wishbone
  counterpart, so each leg's Y is pinned to its own real lower-wishbone Y
  (209.64mm front, 254.34mm rear — the real chassis does **not** put them on
  one shared vertical plate).
- **Z is capped** at the real upper ball-joint height (~375.75mm) — the
  chassis-side mount can never sit higher than the upright-side ball joint —
  and sweeps downward from there in 10mm steps.
- **The tie rod is forced level and straight**: its rack-side (inboard)
  attachment is pinned to the same X and Z as its upright-side (outboard)
  mount, since the CAD only ever had placeholder values there. Rack and tie
  rod now form one straight, level line — no kink, no rake.

Latest run: best candidate at Z=320mm, total score 0.810, roll centre
height 44mm — inside the 20–50mm target band for the first time. No
candidate satisfies every target yet: `rc_migration_roll_y` (lateral roll
centre excursion) spikes sharply past Z≈330mm (up to ~4000mm by Z=350),
which looks like a near-singular FVSA condition rather than a search
failure. Worth investigating next.

### Running it

```
python iteration1_axle_sweep.py --smoke         # one geometry, sanity check
python iteration1_axle_sweep.py --plots         # full sweep + all plots
python iteration1_axle_sweep.py --render-best   # linkage drawing at the top-scoring point
python iteration1_axle_sweep.py --detail Y Z    # full travel curves for one geometry
```

All outputs write to `outputs/`.

## Changelog

### 2026-08-05
- Fixed the mount grid reporting 2205 positions instead of the intended
  step size — `Y_RANGE`/`Z_RANGE` had a 5mm step, not 20mm.
- Constrained the sweep to physically valid mount positions: Y pinned to
  the lower wishbone's chassis plane, Z capped at the upright ball-joint
  height.
- Added `--render-best`: draws the actual linkage geometry at the
  top-scoring mount position.
- Outputs now default to `outputs/` instead of the repo root.
- Fixed the solver's geometry-plot legend (was covering the drawing;
  listed every mirrored left/right part twice) — submodule change,
  `suspension-model@6f6a72d`.
- Replaced every hand-typed fixed hardpoint (lower wishbone, upper ball
  joint, track rod, axle, CG, tire, wheelbase) with values loaded straight
  from `configs/uct_front_geometry_generated.yaml`. The hand-typed lower
  wishbone had the chassis-side mount ~35mm *above* the upright mount; the
  real CAD has it ~10mm *below* — an inverted relationship that was
  inflating the computed roll centre height. Fixing it dropped roll centre
  height at the sweep's best point from over 100mm to 44mm, inside the
  20–50mm target band, and raised the best total score from 0.488 to 0.770.
- Discovered the real chassis puts the upper wishbone's front/rear legs at
  different Y (not one shared mounting plate) — each leg is now pinned to
  its own lower-wishbone counterpart's Y instead of a single shared value.
- Forced the tie rod level and colinear with the rack (both pinned to the
  outboard mount's X and Z) — the CAD's rack-side point was an unmeasured
  placeholder, and a kinked or raked tie rod doesn't steer cleanly. Raised
  the best total score to 0.810.
