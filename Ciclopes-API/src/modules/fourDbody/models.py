from __future__ import annotations

from pydantic import BaseModel, Field


class Sam3DBodyRunInput(BaseModel):
    video_key: str = Field(..., description="Object key for input video in bucket")
    sd_key: str = Field("key", description="Sensor data key in bucket; 'key' = test mode (use all frames)")
    save_name: str | None = Field(None, description="Optional local name for saving this run for later query")


class SkeletonPoint(BaseModel):
    joint_id: int = Field(..., description="MHR70 joint index (0-69)")
    x: float
    y: float
    z: float


class Sam3DBodyRunOutput(BaseModel):
    fps: float = Field(..., description="Source video frame rate — use for playback speed on the frontend")
    skeleton_points: list[list[SkeletonPoint]]


class Sam3DBodyQueryInput(BaseModel):
    shot_numbers: list[int] = Field(default_factory=list, description="Legacy shot numbers to retrieve from the mock DB")
    names: list[str] = Field(default_factory=list, description="Saved run names to retrieve from the local mock DB")


class Sam3DBodyQueryOutput(BaseModel):
    shots: dict[int, Sam3DBodyRunOutput] = Field(default_factory=dict)
    runs: dict[str, Sam3DBodyRunOutput] = Field(default_factory=dict)
