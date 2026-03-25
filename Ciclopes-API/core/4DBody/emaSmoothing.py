from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, List

logger = logging.getLogger("ciclopes.ema_smoothing")

# Default EMA alpha — higher = less smoothing, lower = more smoothing.
# 0.4 keeps the skeleton responsive while killing per-frame jitter.
DEFAULT_ALPHA = 0.4


def ema_smooth_skeleton_frames(
    frames: List[List[dict[str, Any]]],
    alpha: float = DEFAULT_ALPHA,
) -> List[List[dict[str, Any]]]:
    """
    Apply exponential moving average smoothing to skeleton joint positions
    across frames to reduce temporal jitter.

    Each joint is tracked by its joint_id.  For every frame the smoothed
    position is:

        smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]

    Joints that appear or disappear between frames are handled gracefully —
    a joint's EMA state is initialised the first time it's seen and kept
    until the end.

    Args:
        frames: List of frames, each frame is a list of joint dicts
                with keys: joint_id (int), x, y, z (float).
        alpha:  Smoothing factor in (0, 1].  1.0 = no smoothing.

    Returns:
        Smoothed frames in the same structure.
    """
    if not frames or alpha >= 1.0:
        return frames

    alpha = max(min(alpha, 1.0), 0.01)

    # Running EMA state per joint_id: {joint_id: (x, y, z)}
    state: dict[int, tuple[float, float, float]] = {}

    smoothed_frames: List[List[dict[str, Any]]] = []

    for frame_joints in frames:
        smoothed_joints: List[dict[str, Any]] = []

        # Index current frame joints by joint_id for lookup
        current: dict[int, dict[str, Any]] = {}
        for j in frame_joints:
            current[int(j["joint_id"])] = j

        for jid, j in current.items():
            raw_x = float(j["x"])
            raw_y = float(j["y"])
            raw_z = float(j["z"])

            if jid in state:
                prev_x, prev_y, prev_z = state[jid]
                sx = alpha * raw_x + (1.0 - alpha) * prev_x
                sy = alpha * raw_y + (1.0 - alpha) * prev_y
                sz = alpha * raw_z + (1.0 - alpha) * prev_z
            else:
                # First appearance — initialise with raw value
                sx, sy, sz = raw_x, raw_y, raw_z

            state[jid] = (sx, sy, sz)

            smoothed_joints.append({
                "joint_id": jid,
                "x": sx,
                "y": sy,
                "z": sz,
            })

        smoothed_frames.append(smoothed_joints)

    logger.info(
        "EMA smoothing: %d frames, %d unique joints, alpha=%.2f",
        len(frames),
        len(state),
        alpha,
    )
    return smoothed_frames
