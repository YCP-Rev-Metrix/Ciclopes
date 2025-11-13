"""
Synthetic Data Generation Script for Isaac Sim 5.1 (Constant Velocity)

Generates RGB + instance/semantic segmentation data of a ball rolling on a bowling lane,
using a constant x-velocity across all episodes. Logs per-frame (x, y) of the ball's
center for homography testing, and prints the constant velocity used.

Setup Requirements:
  1. Scene must have a Lane prim at /World/Lane (18 units long, 1 unit wide, pointing in -X)
  2. Scene must have a Sphere prim at /World/Sphere (0.3 scale diameter ball)
  3. Scene must have a Camera prim at /World/Camera (already positioned/configured)
  4. Semantics will be set automatically by script for 'lane' and 'ball' classes
"""
# Run inside Isaac Sim Script Editor (or as a kit app)
import os, random, csv
import omni.replicator.core as rep
import omni.usd
import omni.kit.app
from pxr import UsdGeom, Gf, Sdf

# ---------------------- CONFIG ----------------------
# Episodes and timing
NUM_EPISODES = 2
EPISODE_LEN  = 30            # frames per episode
FPS          = 30            # used to report velocity in units/sec

# Constant velocity along -X (negative direction)
# Choose approximately 17 units/sec so a start near +8.x reaches ~-8.x in ~1s (30 frames)
SPEED_UNITS_PER_SEC = 17.0   # magnitude in stage units per second
DX_PER_FRAME        = -SPEED_UNITS_PER_SEC / FPS  # negative to move toward -X

# Output directory (different from the default used in the other script)
OUT_DIR = os.path.join(os.getcwd(), r"\isaacsim\synth_data_scripts\isaac_raw_output_constant_vel_v1")

# Scene prim paths (must exist in your stage)
LANE_PRIM    = "/World/Lane"
BALL_PRIM    = "/World/Sphere"
CAMERA_PRIM  = "/World/Camera"  # existing camera (already positioned)

# Lane configuration: 18 units long, 1 unit wide, pointing in -X direction
# Lane runs from X=+9 (start, positive end) to X=-9 (end, negative end)
LANE_X_START =  +9.0   # positive end (where ball starts)
LANE_X_END   =  -9.0   # negative end (target direction)
LANE_Y_CENTER=   0.0   # lane centerline Y
LANE_WIDTH   =   1.0   # lane width (Y dimension)
LANE_Z       =   0.0008  # lane surface height

# Ball configuration: 0.3 scale = 0.3 diameter, so radius = 0.15
BALL_RADIUS  =   0.15  # ball radius (scale/2)
BALL_Z       =   LANE_Z + 0.001 + BALL_RADIUS  # lane surface + lane thickness + ball radius

# Margins to keep ball fully on lane
MARGIN_X     =   0.5   # keep random start away from very edge (X direction)
MARGIN_Y     =   BALL_RADIUS + 0.05  # keep ball on lane (Y direction, radius + small buffer)
# ----------------------------------------------------

# Helpers to set a prim's world transform cleanly
def set_world_transform(prim_path, pos: Gf.Vec3d):
    """Set prim position only (preserves existing scale/rotation for rigid bodies)."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim:
        raise RuntimeError(f"Prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    xform_ops = xformable.GetOrderedXformOps()

    translate_op = None
    for op in xform_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break

    if translate_op is None:
        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "")

    translate_op.Set(Gf.Vec3d(*pos))

def set_ball_scale(prim_path, diameter):
    """Set ball scale to ensure correct diameter/radius."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim:
        raise RuntimeError(f"Prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    xform_ops = xformable.GetOrderedXformOps()

    scale_op = None
    for op in xform_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break

    if scale_op is None:
        scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "")

    # Set uniform scale (diameter value for a unit sphere)
    scale_op.Set(Gf.Vec3d(diameter, diameter, diameter))

# ---------- Stage / camera / semantics --------------
stage = omni.usd.get_context().get_stage()
assert stage.GetPrimAtPath(LANE_PRIM),   f"Missing prim: {LANE_PRIM}"
assert stage.GetPrimAtPath(BALL_PRIM),   f"Missing prim: {BALL_PRIM}"
assert stage.GetPrimAtPath(CAMERA_PRIM), f"Missing prim: {CAMERA_PRIM}"

# Use existing camera (already positioned in your scene)
camera_prim_path = CAMERA_PRIM

# ========== BALL INITIALIZATION (Health Check & Setup) ==========
print("Initializing ball with proper scale and constant-velocity motion...")

BALL_DIAMETER = BALL_RADIUS * 2.0  # 0.3
set_ball_scale(BALL_PRIM, BALL_DIAMETER)
print(f"  ✓ Ball scale set to {BALL_DIAMETER} (radius = {BALL_RADIUS})")
print(f"  ✓ Constant X velocity: dx/frame = {DX_PER_FRAME:.6f} units/frame; "
      f"|v| = {abs(SPEED_UNITS_PER_SEC):.6f} units/sec at {FPS} FPS")
print()

# Create render product from existing camera
render_product = rep.create.render_product(camera_prim_path, resolution=(1920, 1080))

# Clean, explicit semantics so your labels are stable
rep.modify.semantics(semantics=[("class", "ball")], input_prims=[BALL_PRIM], mode="replace")
rep.modify.semantics(semantics=[("class", "lane")], input_prims=[LANE_PRIM], mode="replace")

# BasicWriter for rgb + masks (your converter will read these)
def make_writer(out_dir):
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=out_dir,
        rgb=True,
        instance_segmentation=True,
        semantic_segmentation=True
    )
    writer.attach([render_product])
    return writer

def compute_positions_constant_velocity(
    num_episodes: int,
    episode_len: int,
    x_start: float,
    y_center: float,
    lane_width: float,
    z_value: float,
    dx_per_frame: float,
):
    """Precompute positions so we can warm-up without losing a saved frame."""
    positions = []
    episode_indices = []
    frame_in_episode_indices = []

    for e in range(num_episodes):
        # Random start position near the positive end (with margin from edge)
        start_x = random.uniform(x_start - MARGIN_X, x_start - 1.0)

        # Random Y position across lane width (accounting for ball radius)
        y_min = y_center - (lane_width / 2.0 - MARGIN_Y)
        y_max = y_center + (lane_width / 2.0 - MARGIN_Y)
        y = random.uniform(y_min, y_max)

        for k in range(episode_len):
            x = start_x + dx_per_frame * k
            positions.append((x, y, z_value))
            episode_indices.append(e)
            frame_in_episode_indices.append(k)

    return positions, episode_indices, frame_in_episode_indices

# ------------- Render loop -------------
async def run_generation():
    """Generate 2 episodes x 30 frames with constant velocity; log (x,y) each frame."""
    os.makedirs(OUT_DIR, exist_ok=True)

    total_saved_frames = NUM_EPISODES * EPISODE_LEN
    positions, epi_idx, epi_frame_idx = compute_positions_constant_velocity(
        NUM_EPISODES, EPISODE_LEN, LANE_X_START, LANE_Y_CENTER, LANE_WIDTH, BALL_Z, DX_PER_FRAME
    )

    print(f"Generating {total_saved_frames} frames ({NUM_EPISODES} episodes x {EPISODE_LEN} frames) at {FPS} FPS...")
    print(f"  Constant velocity: {DX_PER_FRAME:.6f} units/frame  ({-DX_PER_FRAME * FPS:.6f} units/sec toward -X)")
    print(f"  Ball will cover lane width: Y ∈ "
          f"[{LANE_Y_CENTER - LANE_WIDTH/2 + MARGIN_Y:.3f}, {LANE_Y_CENTER + LANE_WIDTH/2 - MARGIN_Y:.3f}]")
    print(f"  Output dir: {OUT_DIR}")
    print()

    # Warm-up: set first position and render once WITHOUT saving (keep same first position for saved frame 0)
    print("  Warming up (skipping initial render to ensure ball is in frame)...")
    first_pos = positions[0]
    set_world_transform(BALL_PRIM, first_pos)
    await omni.kit.app.get_app().next_update_async()
    await rep.orchestrator.step_async()  # render but don't save (writer not attached yet)

    # NOW attach the writer so saving starts at what we'll label as saved frame 0
    writer = make_writer(OUT_DIR)

    # Prepare CSV logging
    csv_path = os.path.join(OUT_DIR, "constant_velocity_log.csv")
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Header for file and console
        header = ["frame_idx","episode_idx","frame_in_episode","t_sec","x","y","vx_units_per_sec","dx_units_per_frame"]
        csv_writer.writerow(header)
        print("\n" + ",".join(header))

        # Generate and save each frame: set ball position, render, write, and log
        for i in range(total_saved_frames):
            pos = positions[i]
            set_world_transform(BALL_PRIM, pos)
            await omni.kit.app.get_app().next_update_async()
            await rep.orchestrator.step_async()  # renders & writes one frame (RGB + masks)

            # Logging for homography testing: x,y center of ball on lane (world coords)
            x, y, _ = pos
            t_sec = i / float(FPS)
            row = [i, epi_idx[i], epi_frame_idx[i], f"{t_sec:.6f}", f"{x:.6f}", f"{y:.6f}", f"{-DX_PER_FRAME*FPS:.6f}", f"{DX_PER_FRAME:.6f}"]
            # Write file row and print console row
            csv_writer.writerow(row)
            print(",".join(map(str, row)))

    # Ensure all frames are written
    await rep.orchestrator.wait_until_complete_async()
    print(f"\nDone! Generated {total_saved_frames} saved frames in '{OUT_DIR}'")
    print(f"CSV log saved to: {csv_path}")
    print("Copy the CSV lines above or use the saved CSV for your postprocessing tests.")

# Run the generation
import asyncio
asyncio.ensure_future(run_generation())


