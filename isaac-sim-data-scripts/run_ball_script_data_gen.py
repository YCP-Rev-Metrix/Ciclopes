"""
Synthetic Data Generation Script for Isaac Sim 5.1

Generates RGB + instance/semantic segmentation data of a ball rolling on a bowling lane.
Ball position is randomized in both X (along lane) and Y (across lane width) to provide
comprehensive training data coverage of all possible ball positions on the lane.

Setup Requirements:
  1. Scene must have a Lane prim at /World/Lane (18 units long, 1 unit wide, pointing in -X)
  2. Scene must have a Sphere prim at /World/Sphere (0.3 scale diameter ball)
  3. Scene must have a Camera prim at /World/Camera (already positioned/configured)
  4. Semantics will be set automatically by script for 'lane' and 'ball' classes

Motion Pattern:
  - Each episode (30 frames): ball starts at random (X,Y) position, rolls in -X direction
  - X randomization: starts near +X end (positive), rolls toward -X end (negative)
  - Y randomization: uniform distribution across lane width (accounts for ball radius)
  - 183 episodes = 183 unique ball starting positions

Usage:
  1. Open your scene in Isaac Sim 5.1
  2. Run this script in the Script Editor
  3. Wait for generation to complete (outputs to isaac_raw_output/)
  4. Convert to YOLO format:
     python repo_to_yolo_seg.py --in_dir isaac_raw_output --out_dir yolo_dataset --train_val_split 0.8
"""
# Run inside Isaac Sim Script Editor (or as a kit app)
import os, math, random
import omni.replicator.core as rep
import omni.usd
import omni.kit.app
from pxr import UsdGeom, Gf, Sdf

# ---------------------- CONFIG ----------------------
TOTAL_FRAMES = 5101          # total frames to generate (train/val split done in conversion)
EPISODE_LEN  = 30            # frames per episode (ball resets every 30 frames)
FPS          = 30            # for reference; not required for writer
OUT_DIR      = os.path.join(os.getcwd(), "\isaacsim\synth_data_scripts\isaac_raw_output_ds_v1")  # raw Isaac Sim output

# Scene prim paths (must exist in your stage)
LANE_PRIM    = "/World/Lane"
BALL_PRIM    = "/World/Sphere"
CAMERA_PRIM  = "/World/Camera"  # existing camera (already positioned)

# Lane configuration: 18 units long, 1 unit wide, pointing in -X direction
# Lane runs from X=+9 (start, positive end) to X=-9 (end, negative end)
LANE_X_START =  +9.0   # positive end (where ball starts)
LANE_X_END   =  -9.0   # negative end (where ball rolls to)
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
    
    # Get existing xform ops
    xform_ops = xformable.GetOrderedXformOps()
    
    # Look for translate op, or create individual ops (translate, rotate, scale)
    translate_op = None
    for op in xform_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    
    # If no translate op exists, add one (don't clear existing ops to preserve scale/rotation)
    if translate_op is None:
        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "")
    
    # Set only the position
    translate_op.Set(Gf.Vec3d(*pos))

def set_ball_scale(prim_path, diameter):
    """Set ball scale to ensure correct diameter/radius."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim:
        raise RuntimeError(f"Prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    xform_ops = xformable.GetOrderedXformOps()
    
    # Look for existing scale op
    scale_op = None
    for op in xform_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    
    # If no scale op exists, add one
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
print("Initializing ball with proper scale and random starting position...")

# 1. Set ball to correct diameter (0.3 = radius 0.15)
BALL_DIAMETER = BALL_RADIUS * 2.0  # 0.3
set_ball_scale(BALL_PRIM, BALL_DIAMETER)
print(f"  ✓ Ball scale set to {BALL_DIAMETER} (radius = {BALL_RADIUS})")

# 2. Set ball to random starting position on lane
# Random X near the positive end
initial_x = random.uniform(LANE_X_START - MARGIN_X, LANE_X_START - 1.0)
# Random Y across lane width
y_min = LANE_Y_CENTER - (LANE_WIDTH / 2.0 - MARGIN_Y)
y_max = LANE_Y_CENTER + (LANE_WIDTH / 2.0 - MARGIN_Y)
initial_y = random.uniform(y_min, y_max)
# Correct Z on lane surface
initial_z = BALL_Z

set_world_transform(BALL_PRIM, (initial_x, initial_y, initial_z))
print(f"  ✓ Ball positioned at ({initial_x:.3f}, {initial_y:.3f}, {initial_z:.4f})")
print(f"  ✓ Ball is on lane surface with proper radius clearance")
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

# ------------- Motion generator (kinematic) ----------
def episode_generator(total_frames, episode_len, x_start, x_end, y_center, lane_width, z):
    """
    Yields per-frame (x,y,z) positions.
    Every `episode_len` frames, pick a new random starting position:
      - X: near x_start (positive end)
      - Y: random position across lane width
    Then roll toward x_end in episode_len steps (Y stays constant per episode).
    
    Ball rolls from +X (positive) toward -X (negative) direction.
    Y position randomized to cover full lane width for training diversity.
    """
    frame = 0
    while frame < total_frames:
        # Random start position near the positive end (with margin from edge)
        start_x = random.uniform(x_start - MARGIN_X, x_start - 1.0)
        
        # Random Y position across lane width (accounting for ball radius)
        # Lane edges: y_center ± lane_width/2
        # Valid range: y_center ± (lane_width/2 - MARGIN_Y)
        y_min = y_center - (lane_width / 2.0 - MARGIN_Y)
        y_max = y_center + (lane_width / 2.0 - MARGIN_Y)
        y = random.uniform(y_min, y_max)
        
        # Roll toward the negative end (X decreases, Y constant)
        dx_per_frame = (x_end - start_x) / float(episode_len)
        
        for k in range(episode_len):
            if frame >= total_frames:
                break
            x = start_x + dx_per_frame * k
            yield (x, y, z)
            frame += 1

# ------------- Render loop -------------
async def run_generation():
    """Generate all frames in episodes with random ball starting positions (X and Y)."""
    os.makedirs(OUT_DIR, exist_ok=True)

    pos_iter = episode_generator(TOTAL_FRAMES, EPISODE_LEN, LANE_X_START, LANE_X_END, 
                                  LANE_Y_CENTER, LANE_WIDTH, BALL_Z)

    print(f"Generating {TOTAL_FRAMES} frames ({TOTAL_FRAMES // EPISODE_LEN} episodes)...")
    print(f"  Ball will cover lane width: Y ∈ [{LANE_Y_CENTER - LANE_WIDTH/2 + MARGIN_Y:.3f}, "
          f"{LANE_Y_CENTER + LANE_WIDTH/2 - MARGIN_Y:.3f}]")
    
    # Warm-up: Set first position and render once WITHOUT saving (duct-tape fix for frame 0)
    print("  Warming up (skipping frame 0 to ensure ball is in frame)...")
    pos = next(pos_iter)
    set_world_transform(BALL_PRIM, pos)
    await omni.kit.app.get_app().next_update_async()
    await rep.orchestrator.step_async()  # render but don't save
    
    # NOW attach the writer (so it starts saving from what will be labeled as frame 0)
    writer = make_writer(OUT_DIR)
    
    # Generate each frame: set ball position, render, write
    # This loop generates TOTAL_FRAMES-1 frames (since we already consumed one above)
    for i in range(TOTAL_FRAMES - 1):
        if i % 100 == 0:
            print(f"  Frame {i}/{TOTAL_FRAMES - 1}...")
        
        pos = next(pos_iter)
        set_world_transform(BALL_PRIM, pos)
        
        # Wait for USD stage to update before rendering
        await omni.kit.app.get_app().next_update_async()
        
        await rep.orchestrator.step_async()  # renders & writes one frame (RGB + masks)

    # Ensure all frames are written
    await rep.orchestrator.wait_until_complete_async()
    print(f"\n Done! Generated {TOTAL_FRAMES} frames in '{OUT_DIR}'")
    print(f"   Ball appeared at {TOTAL_FRAMES // EPISODE_LEN} different (X,Y) positions")
    print(f"   Next: Run conversion with repo_to_yolo_seg.py")

# Run the generation
import asyncio
asyncio.ensure_future(run_generation())
