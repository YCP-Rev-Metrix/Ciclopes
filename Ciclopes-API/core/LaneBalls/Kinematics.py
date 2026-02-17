from __future__ import annotations

from typing import List

import numpy as np

from core.LaneBalls.Postprocessing import LANE_LENGTH_M
from core.LaneBalls.models import BallPos, Kinematics, QuarterKinematics


def compute_kinematics_per_quarter(
    ball_positions: List[BallPos],
    lane_length_m: float = LANE_LENGTH_M,
) -> Kinematics:
    """
    Compute mean speed and acceleration for each quarter of the lane.
    """
    if len(ball_positions) < 2:
        return Kinematics(
            quarters=[
                QuarterKinematics(
                    quarter=i + 1,
                    start_m=i * lane_length_m / 4.0,
                    end_m=(i + 1) * lane_length_m / 4.0,
                    mean_speed_mps=0.0,
                    mean_acceleration_mps2=0.0,
                    sample_count=0,
                )
                for i in range(4)
            ]
        )

    positions = sorted(ball_positions, key=lambda p: p.frame_index)

    sample_y: List[float] = []
    speeds: List[float] = []
    accels: List[float] = []

    last_speed = None
    for i in range(1, len(positions)):
        prev = positions[i - 1]
        curr = positions[i]
        dt = max(curr.timestamp_s - prev.timestamp_s, 1e-6)
        dx = curr.x_m - prev.x_m
        dy = curr.y_m - prev.y_m
        speed = float(np.hypot(dx, dy) / dt)
        accel = 0.0 if last_speed is None else float((speed - last_speed) / dt)
        last_speed = speed

        sample_y.append(float(np.clip(curr.y_m, 0.0, lane_length_m)))
        speeds.append(speed)
        accels.append(accel)

    quarters: List[QuarterKinematics] = []
    q_len = lane_length_m / 4.0
    for i in range(4):
        q_start = i * q_len
        q_end = (i + 1) * q_len
        idx = [j for j, y in enumerate(sample_y) if q_start <= y <= q_end]

        if idx:
            q_speeds = np.array([speeds[j] for j in idx], dtype=np.float64)
            q_accels = np.array([accels[j] for j in idx], dtype=np.float64)
            mean_speed = float(np.mean(q_speeds))
            mean_accel = float(np.mean(q_accels))
            count = len(idx)
        else:
            mean_speed = 0.0
            mean_accel = 0.0
            count = 0

        quarters.append(
            QuarterKinematics(
                quarter=i + 1,
                start_m=q_start,
                end_m=q_end,
                mean_speed_mps=mean_speed,
                mean_acceleration_mps2=mean_accel,
                sample_count=count,
            )
        )

    return Kinematics(quarters=quarters)
