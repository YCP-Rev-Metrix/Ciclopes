from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class BallPos:
    frame_index: int
    timestamp_s: float
    x_m: float
    y_m: float


@dataclass
class BallPosList:
    ball_positions: List[BallPos] = field(default_factory=list)


@dataclass
class QuarterKinematics:
    quarter: int
    start_m: float
    end_m: float
    mean_speed_mps: float
    mean_acceleration_mps2: float
    sample_count: int


@dataclass
class Kinematics:
    quarters: List[QuarterKinematics] = field(default_factory=list)
