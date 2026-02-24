from __future__ import annotations

from pydantic import BaseModel, Field


class AggRunInput(BaseModel):
    video_key: str = Field(..., description="Object key for input video in bucket")
    sd_key: str = Field(..., description="Skeleton data key placeholder for SAM3D stage")


class BallPoint(BaseModel):
    x: float
    y: float


class SkeletonPoint(BaseModel):
    joint_id: int = Field(..., description="MHR70 joint index (0-69), used as rendering index for skeleton links")
    x: float
    y: float
    z: float


class KinematicsRow(BaseModel):
    quarter: int
    start_m: float
    end_m: float
    mean_speed_mps: float
    mean_acceleration_mps2: float
    sample_count: int


class AggRunOutput(BaseModel):
    ball_points: list[BallPoint]
    kinematics_table: list[KinematicsRow]
    skeleton_points: list[list[SkeletonPoint]]
