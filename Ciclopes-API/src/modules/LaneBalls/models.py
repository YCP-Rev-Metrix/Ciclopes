from __future__ import annotations

from pydantic import BaseModel, Field


class LaneBallRunInput(BaseModel):
    video_key: str = Field(..., description="Object key for input video in bucket")


class BallPoint(BaseModel):
    x: float
    y: float


class KinematicsRow(BaseModel):
    quarter: int
    start_m: float
    end_m: float
    mean_speed_mps: float
    mean_acceleration_mps2: float
    sample_count: int


class LaneBallHealthInfo(BaseModel):
    frames_scanned_for_h: int = 0
    frames_with_lane: int = 0
    frames_with_ball: int = 0
    lane_polygon_count_at_h: int = 0
    homography_determinant: float = 0.0
    homography_condition_number: float = 0.0
    mean_lane_coverage_ratio: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0


class LaneBallRunOutput(BaseModel):
    fps: float = Field(..., description="Source video frame rate — use for playback speed on the frontend")
    ball_points: list[BallPoint]
    kinematics_table: list[KinematicsRow]
    is_trapezoid: bool = False
    homography_frame: int | None = None
    health: LaneBallHealthInfo
