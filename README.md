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

Constraints baked into the sweep:
- **Y is fixed**, not swept — the upper wishbone bolts to the same
  chassis-side plate as the lower wishbone, so its lateral position is
  pinned to the lower wishbone's inboard Y (215mm).
- **Z is capped** at the upright's upper ball-joint height (350mm) — the
  chassis-side mount can never sit higher than the upright-side ball joint —
  and sweeps downward from there in 10mm steps.

Latest run: best candidate at Z=320mm, total score 0.488. No candidate
satisfies every target — `roll_centre_z` never reaches its target band
(20–50mm; actual range is 90–195mm across the swept Z), which looks like
it's unreachable at the current `LOWER_OUTBOARD` height rather than a
search failure. Worth revisiting the lower ball-joint height or the
`roll_centre_z` target band next.

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
