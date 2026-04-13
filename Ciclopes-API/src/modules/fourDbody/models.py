from __future__ import annotations

from pydantic import BaseModel, Field


class Sam3DBodyRunInput(BaseModel):
    video_key: str = Field(..., description="Object key for input video in bucket")
    sd_key: str = Field("key", description="Sensor data key in bucket; 'key' = test mode (use all frames)")


class SkeletonPoint(BaseModel):
    joint_id: int = Field(..., description="MHR70 joint index (0-69)")
    x: float
    y: float
    z: float


class Sam3DBodyRunOutput(BaseModel):
    fps: float = Field(..., description="Source video frame rate — use for playback speed on the frontend")
    skeleton_points: list[list[SkeletonPoint]]
