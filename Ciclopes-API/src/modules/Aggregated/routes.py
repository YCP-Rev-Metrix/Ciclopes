from __future__ import annotations

import asyncio
import logging

import cv2
from fastapi import APIRouter, HTTPException, Request

from src.modules.Aggregated.models import AggRunInput, AggRunOutput, BallPoint, KinematicsRow, SkeletonPoint

logger = logging.getLogger("ciclopes.aggregated_routes")
HARDCODED_START_FRAME = 120

router = APIRouter(
    prefix="/agg",
    tags=["Aggregated"],
)


def _get_engine(request: Request):
    engine = getattr(request.app.state, "inference_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="InferenceEngine is not initialized")
    return engine


def _get_settings(request: Request):
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=503, detail="Application settings are not initialized")
    return settings


@router.post("/run", response_model=AggRunOutput)
async def run_aggregate_pipeline(request: Request, payload: AggRunInput):
    from core.VideoUtil.FrameSplit import split_video_into_frames
    from core.VideoUtil.SpacesApiClient import query_video_via_api_to_temp_file

    engine = _get_engine(request)
    settings = _get_settings(request)

    if not settings.username or not settings.password:
        raise HTTPException(
            status_code=500,
            detail="Missing API credentials. Set username/password in Ciclopes-API/.env or environment.",
        )

    logger.info(
        "Running /agg/run for video_key=%s sd_key=%s",
        payload.video_key,
        payload.sd_key,
    )

    try:
        temp_ctx = query_video_via_api_to_temp_file(
            base=settings.api_base,
            verify_api=settings.verify_api,
            username=settings.username,
            password=settings.password,
            key=payload.video_key,
            ttl_seconds=settings.presign_ttl_seconds,
            verify_presigned=settings.verify_presigned,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Video query failed: {exc}")

    with temp_ctx as temp_path:
        split_video = split_video_into_frames(str(temp_path))

    if not split_video.frames:
        return AggRunOutput(ball_points=[], kinematics_table=[], skeleton_points=[])

    fps = float(split_video.fps) if split_video.fps > 0 else 30.0
    rgb_frames = [cv2.cvtColor(vf.image, cv2.COLOR_BGR2RGB) for vf in split_video.frames]

    # Run both pipelines concurrently — they use separate thread-pool slots.
    lane_ball_coro = engine.forward_lane_ball(
        frames_rgb=rgb_frames,
        fps=fps,
        start_frame=HARDCODED_START_FRAME,
    )
    sam3d_coro = engine.forward_sam3d_body(frames_rgb=rgb_frames)

    try:
        lane_ball_output, sam3d_output = await asyncio.gather(
            lane_ball_coro, sam3d_coro
        )
    except Exception as exc:
        logger.exception("/agg/run pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # ── Lane-ball results ──────────────────────────────────────────────────
    health = lane_ball_output.get("health", {})
    pipeline_error = health.get("error")
    if pipeline_error:
        raise HTTPException(status_code=500, detail=str(pipeline_error))

    positions = lane_ball_output.get("positions", [])
    quarters = lane_ball_output.get("kinematics", {}).get("quarters", [])

    ball_points = [
        BallPoint(x=float(point.get("x_m", 0.0)), y=float(point.get("y_m", 0.0)))
        for point in positions
    ]

    kinematics_table = [
        KinematicsRow(
            quarter=int(row.get("quarter", 0)),
            start_m=float(row.get("start_m", 0.0)),
            end_m=float(row.get("end_m", 0.0)),
            mean_speed_mps=float(row.get("mean_speed_mps", 0.0)),
            mean_acceleration_mps2=float(row.get("mean_acceleration_mps2", 0.0)),
            sample_count=int(row.get("sample_count", 0)),
        )
        for row in quarters
    ]

    # ── SAM3D Body results ─────────────────────────────────────────────────
    # sam3d_output: list[list[dict]] — outer=frames, inner=joints per frame
    skeleton_points = [
        [
            SkeletonPoint(
                joint_id=int(j["joint_id"]),
                x=float(j["x"]),
                y=float(j["y"]),
                z=float(j["z"]),
            )
            for j in frame_joints
        ]
        for frame_joints in sam3d_output
    ]

    return AggRunOutput(
        ball_points=ball_points,
        kinematics_table=kinematics_table,
        skeleton_points=skeleton_points,
    )
