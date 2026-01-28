from __future__ import annotations
from typing import Optional, List, Any
from dataclasses import dataclass

@dataclass
class JointObj:
    x: int
    y: int
    z: int
    joint_id: int

@dataclass
class Skeleton:
    joints: List[JointObj] = []

@dataclass
class VideoSkeleton:
    skeletons: List[Skeleton] = []