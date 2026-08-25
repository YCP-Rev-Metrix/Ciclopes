# Isaac Sim synthetic data tools

These scripts generate controlled bowling-lane segmentation data and trajectory ground truth in NVIDIA Isaac Sim, then convert Replicator output into Ultralytics YOLO segmentation labels. They supplement the primary Roboflow dataset; they do not replace its real labeled images.

## Files

| File | Purpose |
|---|---|
| `run_ball_script_data_gen.py` | Basic Replicator capture from an existing lane/ball/camera stage |
| `run_ball_script_data_gen_constant_vel.py` | Straight/constant-velocity trajectory generation |
| `run_realistic_traj_gen.py` | Variable-speed, decelerating, late-hook trajectory plus per-frame CSV ground truth |
| `repo_to_yolo_seg.py` | Converts RGB + semantic/instance bundles to normalized YOLO polygons and optionally splits train/validation |
| `visualize_segmentation.py` | Inspects raw segmentation output |
| `visualize_yolo_polygons.py` | Overlays generated YOLO polygons for label QA |
| `WORKFLOW.md` | Detailed Isaac Sim execution workflow |
| `requirements.txt` | Dependencies for conversion/visualization scripts outside Isaac Sim |

## Algorithm

The realistic generator assumes an existing stage with `/World/Lane`, `/World/Sphere`, and `/World/Camera`. It assigns stable `lane` and `ball` semantics, samples starting position, initial longitudinal speed, deceleration, and signed hook amplitude, then advances the ball at 30 FPS until it leaves the usable lane or reaches the end.

Longitudinal speed decreases each step. Lateral displacement follows a cubic lane-fraction profile, producing a late break instead of a straight line. Finite differences yield velocity and acceleration. Replicator writes RGB, instance segmentation, and semantic segmentation; a CSV records frame/episode IDs, position, velocity, acceleration, lane fraction, lateral break, and terminal summaries.

`repo_to_yolo_seg.py` groups matching Replicator outputs, reads class/color mappings, extracts contours for target semantics, normalizes polygon coordinates to `[0,1]`, writes YOLO segmentation label rows, and optionally creates train/validation folders. It can parallelize across CPU workers.

## Workflow

1. Open/configure the expected lane scene in Isaac Sim.
2. Review hard-coded prim paths, scale, camera, frame rate, episode count, and output directory.
3. Paste/run the desired generator in Isaac Sim's Script Editor as described in `WORKFLOW.md`.
4. Inspect raw masks before conversion.
5. Convert output outside Isaac Sim:

   ```bash
   python repo_to_yolo_seg.py --in_dir path/to/replicator-output --out_dir path/to/yolo-output --train_val_split 0.8
   ```

6. Overlay YOLO polygons and spot-check class IDs, alignment, holes, tiny contours, and empty labels.
7. Merge/upload approved synthetic data as a separately versioned Roboflow dataset version; do not silently mix it into a real-data export.

## Important class warning

The converter's generated YAML may reflect the synthetic script's available classes (historically lane/ball), while the current production model uses `0: ball`, `1: lane`, `2: pins`. Before mixing datasets, remap labels into the production class order and add/handle pins consistently. Never concatenate label files from datasets with different class maps.

## Limitations

- The generator moves a scripted sphere; it is not a full bowling physics simulation.
- A cubic hook and constant deceleration are controlled approximations, not learned ball dynamics.
- Camera, lighting, materials, occlusion, lane reflections, and pin behavior create a synthetic-to-real domain gap.
- Hard-coded output paths and stage prim names must be reviewed on every workstation.
- Isaac Sim scripts use its bundled Python/runtime; install `requirements.txt` only for the external conversion/inspection tools.

Use synthetic data for coverage, failure injection, and regression tests. Validate final checkpoints on held-out real Roboflow data and real shot videos.
