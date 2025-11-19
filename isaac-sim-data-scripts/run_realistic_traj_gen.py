"""
Realistic (curved, non-constant-velocity) trajectory generator for Isaac Sim 5.1.

Runs inside the Isaac Sim Script Editor and produces:
  * RGB + instance/semantic segmentation frames for a ball rolling on a lane
  * A CSV log with per-frame kinematics and lane metrics suitable for
    downstream pre/post-processing tests.

Compared to `run_ball_script_data_gen_constant_vel.py`:
  * Velocity along the lane is NOT constant – each episode samples an initial
    speed and a constant deceleration.
  * The ball follows a smooth, randomly-curved path across the lane with
    realistic “break” (lateral hook) by the time it exits the lane.
  * Episodes do not have a fixed frame count – each episode starts on-lane
    and ends as soon as the ball leaves the lane bounds or reaches the end.
"""

import csv
import os
import random
from typing import List, Tuple, Dict

import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom, Gf, Sdf


# ---------------------- CONFIG ----------------------
# Episodes and timing
NUM_EPISODES = 3
FPS = 30.0
DT = 1.0 / FPS
MAX_STEPS_PER_EPISODE = 120  # hard safety cap

# Output directory
OUT_DIR = os.path.join(
    os.getcwd(),
    r"isaacsim\synth_data_scripts\isaac_raw_output_curved_traj_v1",
)

# Scene prim paths (must exist in your stage)
LANE_PRIM = "/World/Lane"
BALL_PRIM = "/World/Sphere"
CAMERA_PRIM = "/World/Camera"  # existing camera (already positioned)

# Lane configuration: 18 units long, 1 unit wide, pointing in -X direction
# Lane runs from X=+9 (start, positive end) to X=-9 (end, negative end)
LANE_X_START = +9.0  # positive end (where ball starts)
LANE_X_END = -9.0  # negative end
LANE_Y_CENTER = 0.0  # lane centerline Y
LANE_WIDTH = 1.0  # lane width (Y dimension)
LANE_Z = 0.0008  # lane surface height

LANE_LENGTH = LANE_X_START - LANE_X_END  # 18 units

# Ball configuration: 0.3 scale = 0.3 diameter, so radius = 0.15
BALL_RADIUS = 0.15  # ball radius (scale/2)
BALL_Z = LANE_Z + 0.001 + BALL_RADIUS  # lane surface + lane thickness + ball radius

# Margins to keep ball initially on lane
MARGIN_X = 0.5  # keep random start away from very edge (X direction)
MARGIN_Y = BALL_RADIUS + 0.05  # keep ball fully on lane (Y direction)


# ----------------------------------------------------
# Helpers to set prim transforms

def _get_stage():
    return omni.usd.get_context().get_stage()


def set_world_transform(prim_path: str, pos: Tuple[float, float, float]) -> None:
    """Set prim position only (preserves existing scale/rotation for rigid bodies)."""
    stage = _get_stage()
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


def set_ball_scale(prim_path: str, diameter: float) -> None:
    """Set ball scale to ensure correct diameter/radius."""
    stage = _get_stage()
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

    scale_op.Set(Gf.Vec3d(diameter, diameter, diameter))


# ---------- Stage / camera / semantics --------------
stage = _get_stage()
assert stage.GetPrimAtPath(LANE_PRIM), f"Missing prim: {LANE_PRIM}"
assert stage.GetPrimAtPath(BALL_PRIM), f"Missing prim: {BALL_PRIM}"
assert stage.GetPrimAtPath(CAMERA_PRIM), f"Missing prim: {CAMERA_PRIM}"

camera_prim_path = CAMERA_PRIM

print("Initializing ball and lane for curved, variable-velocity trajectories...")

BALL_DIAMETER = BALL_RADIUS * 2.0
set_ball_scale(BALL_PRIM, BALL_DIAMETER)
print(f"  ✓ Ball scale set to {BALL_DIAMETER} (radius = {BALL_RADIUS})")

# Create render product from existing camera
render_product = rep.create.render_product(camera_prim_path, resolution=(1920, 1080))

# Clean, explicit semantics so your labels are stable
rep.modify.semantics(semantics=[("class", "ball")], input_prims=[BALL_PRIM], mode="replace")
rep.modify.semantics(semantics=[("class", "lane")], input_prims=[LANE_PRIM], mode="replace")


def make_writer(out_dir: str):
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=out_dir,
        rgb=True,
        instance_segmentation=True,
        semantic_segmentation=True,
    )
    writer.attach([render_product])
    return writer


# ---------- Episode trajectory generator ------------

def _sample_episode_params() -> Dict[str, float]:
    """
    Sample a set of kinematic parameters for one episode.

    Returns a dict with:
      - x_start: initial X
      - y0: initial Y (lane center +/-)
      - v0: initial speed along lane (units/sec)
      - decel: constant positive deceleration magnitude (units/sec^2)
      - hook_ampl: lateral hook amplitude (units)
    """
    # Random start X a bit inside the positive end
    x_start = random.uniform(LANE_X_START - MARGIN_X, LANE_X_START - 1.0)

    # Random Y across lane width, ensuring fully on-lane initially
    y_min = LANE_Y_CENTER - (LANE_WIDTH / 2.0 - MARGIN_Y)
    y_max = LANE_Y_CENTER + (LANE_WIDTH / 2.0 - MARGIN_Y)
    y0 = random.uniform(y_min, y_max)

    # Initial speed and deceleration -> non-constant velocity
    v0 = random.uniform(14.0, 19.0)  # units/sec
    decel = random.uniform(3.0, 6.0)  # units/sec^2

    # Lateral hook amplitude so the ball meaningfully breaks toward a gutter
    hook_ampl = random.uniform(0.25, 0.40) * (1.0 if random.random() > 0.5 else -1.0)

    return {
        "x_start": x_start,
        "y0": y0,
        "v0": v0,
        "decel": decel,
        "hook_ampl": hook_ampl,
    }


def _hook_profile(p: float) -> float:
    """
    Smooth hook profile as a function of lane fraction p ∈ [0, 1].
    Small near the foul line, growing toward the pins.
    """
    p = max(0.0, min(1.0, p))
    # Late-breaking cubic curve; near zero at start, saturates near end.
    return p ** 3


def generate_episode_frames(params: Dict[str, float]) -> List[Dict[str, float]]:
    """
    Generate per-frame world positions and kinematics for one episode.

    Returns a list of dicts, each with keys:
      t, x, y, vx, vy, speed, ax, ay, accel, lane_s, lane_frac,
      lat_from_center, lat_from_start
    """
    x_start = params["x_start"]
    y0 = params["y0"]
    v0 = params["v0"]
    decel = params["decel"]
    hook_ampl = params["hook_ampl"]

    records: List[Dict[str, float]] = []

    t = 0.0
    lane_s = 0.0  # distance along lane from foul line (0 -> LANE_LENGTH)
    v_long = v0

    prev_x = None
    prev_y = None
    prev_vx = None
    prev_vy = None

    for _ in range(MAX_STEPS_PER_EPISODE):
        # Fraction along lane
        lane_frac = max(0.0, min(1.0, lane_s / LANE_LENGTH))

        x = x_start - lane_s
        y = y0 + hook_ampl * _hook_profile(lane_frac)

        # Basic on/off-lane condition (center exceeds lane bounds)
        if abs(y - LANE_Y_CENTER) > (LANE_WIDTH / 2.0):
            # Stop once the ball center has left the lane bounds
            break

        # Approximate velocities using finite differences (position-based)
        if prev_x is None:
            vx = -v_long  # initial guess: motion mostly along -X
            vy = 0.0
        else:
            vx = (x - prev_x) / DT
            vy = (y - prev_y) / DT

        # Approximate accelerations
        if prev_vx is None:
            ax = -decel  # along -X
            ay = 0.0
        else:
            ax = (vx - prev_vx) / DT
            ay = (vy - prev_vy) / DT

        speed = (vx ** 2 + vy ** 2) ** 0.5
        accel_mag = (ax ** 2 + ay ** 2) ** 0.5

        lat_from_center = y - LANE_Y_CENTER
        lat_from_start = y - y0

        records.append(
            {
                "t": t,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "speed": speed,
                "ax": ax,
                "ay": ay,
                "accel": accel_mag,
                "lane_s": lane_s,
                "lane_frac": lane_frac,
                "lat_from_center": lat_from_center,
                "lat_from_start": lat_from_start,
            }
        )

        prev_x, prev_y = x, y
        prev_vx, prev_vy = vx, vy

        # Update kinematics for next step
        v_long = max(0.5, v_long - decel * DT)  # do not let it hit exactly zero
        lane_s += v_long * DT
        t += DT

        # Stop if we reach the physical end of the lane
        if lane_s >= LANE_LENGTH:
            break

    return records


# ------------- Render loop & CSV logging -------------
async def run_generation():
    """Generate episodes with curved, non-constant-velocity trajectories."""
    os.makedirs(OUT_DIR, exist_ok=True)
    make_writer(OUT_DIR)

    csv_path = os.path.join(OUT_DIR, "curved_trajectory_log.csv")
    header = [
        "global_frame_idx",
        "episode_idx",
        "frame_in_episode",
        "t_sec",
        "x",
        "y",
        "vx_units_per_sec",
        "vy_units_per_sec",
        "speed_units_per_sec",
        "ax_units_per_sec2",
        "ay_units_per_sec2",
        "accel_units_per_sec2",
        "lane_s",
        "lane_frac",
        "lat_from_center",
        "lat_from_start",
        "episode_total_break",
        "episode_end_speed",
    ]

    global_frame_idx = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(header)

        print(
            f"Generating {NUM_EPISODES} episodes with curved, variable-velocity "
            f"trajectories at {FPS:.1f} FPS..."
        )
        print(f"  Output dir: {OUT_DIR}")
        print(f"  CSV log:   {csv_path}")

        for episode_idx in range(NUM_EPISODES):
            params = _sample_episode_params()
            print(
                f"\nEpisode {episode_idx}: "
                f"x_start={params['x_start']:.3f}, "
                f"y0={params['y0']:.3f}, "
                f"v0={params['v0']:.3f} units/s, "
                f"decel={params['decel']:.3f} units/s^2, "
                f"hook_ampl={params['hook_ampl']:.3f}"
            )

            episode_records = generate_episode_frames(params)
            if not episode_records:
                print("  (No frames generated for this episode; skipping)")
                continue

            # Episode-level ground truths
            total_break = abs(episode_records[-1]["lat_from_start"])
            end_speed = episode_records[-1]["speed"]

            for frame_in_episode, rec in enumerate(episode_records):
                # Set ball transform and render frame
                set_world_transform(BALL_PRIM, (rec["x"], rec["y"], BALL_Z))
                await omni.kit.app.get_app().next_update_async()
                await rep.orchestrator.step_async()

                row = [
                    global_frame_idx,
                    episode_idx,
                    frame_in_episode,
                    f"{rec['t']:.6f}",
                    f"{rec['x']:.6f}",
                    f"{rec['y']:.6f}",
                    f"{rec['vx']:.6f}",
                    f"{rec['vy']:.6f}",
                    f"{rec['speed']:.6f}",
                    f"{rec['ax']:.6f}",
                    f"{rec['ay']:.6f}",
                    f"{rec['accel']:.6f}",
                    f"{rec['lane_s']:.6f}",
                    f"{rec['lane_frac']:.6f}",
                    f"{rec['lat_from_center']:.6f}",
                    f"{rec['lat_from_start']:.6f}",
                    f"{total_break:.6f}",
                    f"{end_speed:.6f}",
                ]
                writer_csv.writerow(row)
                global_frame_idx += 1

    await rep.orchestrator.wait_until_complete_async()
    print(f"\nDone! Generated {global_frame_idx} frames across {NUM_EPISODES} episodes.")
    print(f"CSV log saved to: {csv_path}")


# Run the generation
import asyncio

asyncio.ensure_future(run_generation())

