from __future__ import annotations
from typing import Optional, List, Any
from dataclasses import dataclass

@dataclass
class BallPos:
    x: float
    y: float

@dataclass
class BallPosList:
    ball_positions: List[BallPos] = []

@dataclass
class Kinematics:
    q1_velocity: float
    q1_acceleration: float
    q2_velocity: float
    q2_acceleration: float
    q3_velocity: float
    q3_acceleration: float
    q4_velocity: float
    q4_acceleration: float