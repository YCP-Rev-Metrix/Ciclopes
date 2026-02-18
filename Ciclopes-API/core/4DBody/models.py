from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class JointObj:
    """Single joint in 3D space, identified by joint_id (MHR70 ordering)"""
    x: float
    y: float
    z: float
    joint_id: int


@dataclass
class Skeleton:
    joints: List[JointObj] = field(default_factory=list)


@dataclass
class VideoSkeleton:
    skeletons: List[Skeleton] = field(default_factory=list)