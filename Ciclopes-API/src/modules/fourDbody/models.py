from __future__ import annotations

from pydantic import BaseModel, Field


class Sam3DBodyRunInput(BaseModel):
    video_key: str = Field(..., description="Object key for input video in bucket")


class SkeletonPoint(BaseModel):
    joint_id: int = Field(..., description="MHR70 joint index (0-69)")
    x: float
    y: float
    z: float


class Sam3DBodyRunOutput(BaseModel):
    skeleton_points: list[list[SkeletonPoint]]
