from __future__ import annotations

import logging
import sys
import traceback

import cv2
from fastapi import APIRouter, HTTPException, Request
from core.VideoUtil.FrameSplit import split_video_into_frames
from core.VideoUtil.SpacesApiClient import query_video_via_api_to_temp_file

from src.modules.LaneBalls.models import (
    BallPoint,
    KinematicsRow,
    LaneBallHealthInfo,
    LaneBallRunInput,
    LaneBallRunOutput,
)

logger = logging.getLogger("ciclopes.laneballs_routes")

router = APIRouter(
    prefix="/laneballs",
    tags=["LaneBalls"],
)


def _empty_output(fps: float = 30.0) -> LaneBallRunOutput:
    return LaneBallRunOutput(
        fps=fps,
        ball_points=[],
        kinematics_table=[],
        is_trapezoid=False,
        homography_frame=None,
        health=LaneBallHealthInfo(),
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


@router.post("/run", response_model=LaneBallRunOutput)
async def run_lane_ball_pipeline(request: Request, payload: LaneBallRunInput):
    """
    Run the lane-ball detection + kinematics pipeline on a video

    This endpoint runs asynchronously, it does not block SAM3D Body or other concurrent requests because the GPU work is dispatched to a thread-pool executor inside InferenceEngine
    """
    fps = 30.0

    try:
        return await _run_lane_ball_pipeline_inner(request, payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "UNHANDLED EXCEPTION in /laneballs/run:\n%s",
            traceback.format_exc(),
        )
        sys.stdout.flush()
        return _empty_output(fps=fps)


async def _run_lane_ball_pipeline_inner(request: Request, payload: LaneBallRunInput) -> LaneBallRunOutput:
    engine = _get_engine(request)
    settings = _get_settings(request)

    if not settings.username or not settings.password:
        raise HTTPException(
            status_code=500,
            detail="Missing API credentials. Set username/password in Ciclopes-API/.env or environment.",
        )

    logger.info("Running /laneballs/run for video_key=%s", payload.video_key)

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
        split_video = split_video_into_frames(
            str(temp_path),
            max_frames=settings.max_video_frames,
            max_dimension=settings.max_video_dimension,
        )

    if not split_video.frames:
        logger.warning("No frames extracted for video_key=%s", payload.video_key)
        return _empty_output()

    fps = float(split_video.fps) if split_video.fps > 0 else 30.0
    rgb_frames = [cv2.cvtColor(vf.image, cv2.COLOR_BGR2RGB) for vf in split_video.frames]

    # Free the raw BGR frames — we only need rgb_frames from here on
    split_video.frames.clear()
    del split_video

    try:
        lane_ball_output = await engine.forward_lane_ball(
            frames_rgb=rgb_frames,
            fps=fps,
            start_frame=settings.lane_ball_start_frame,
            batch_size=settings.lane_ball_batch_size,
        )
    except Exception as exc:
        logger.exception("/laneballs/run pipeline failed")
        return _empty_output(fps=fps)
    finally:
        rgb_frames.clear()
        del rgb_frames

    # ── Parse results ─────────────────────────────────────────────────────
    health_raw = lane_ball_output.get("health", {})

    positions = lane_ball_output.get("positions", [])
    quarters = lane_ball_output.get("kinematics", {}).get("quarters", [])

    ball_points = [
        BallPoint(x=float(p.get("x_m", 0.0)), y=float(p.get("y_m", 0.0)))
        for p in positions
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

    health = LaneBallHealthInfo(
        frames_scanned_for_h=health_raw.get("frames_scanned_for_h", 0),
        frames_with_lane=health_raw.get("frames_with_lane", 0),
        frames_with_ball=health_raw.get("frames_with_ball", 0),
        lane_polygon_count_at_h=health_raw.get("lane_polygon_count_at_h", 0),
        homography_determinant=health_raw.get("homography_determinant", 0.0),
        homography_condition_number=health_raw.get("homography_condition_number", 0.0),
        mean_lane_coverage_ratio=health_raw.get("mean_lane_coverage_ratio", 0.0),
        inference_ms=health_raw.get("inference_ms", 0.0),
        postprocess_ms=health_raw.get("postprocess_ms", 0.0),
    )

    return LaneBallRunOutput(
        fps=fps,
        ball_points=ball_points,
        kinematics_table=kinematics_table,
        is_trapezoid=lane_ball_output.get("is_trapezoid", False),
        homography_frame=lane_ball_output.get("homography_frame"),
        health=health,
    )
