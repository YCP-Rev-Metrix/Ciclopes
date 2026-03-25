from __future__ import annotations

import logging

import cv2
from fastapi import APIRouter, HTTPException, Request

import importlib as _il

from src.modules.fourDbody.models import Sam3DBodyRunInput, Sam3DBodyRunOutput, SkeletonPoint
from core.VideoUtil.FrameSplit import split_video_into_frames

_ema_mod = _il.import_module("core.4DBody.emaSmoothing")
ema_smooth_skeleton_frames = _ema_mod.ema_smooth_skeleton_frames
from core.VideoUtil.SpacesApiClient import query_video_via_api_to_temp_file

logger = logging.getLogger("ciclopes.fourdbody_routes")

router = APIRouter(
    prefix="/fourdbody",
    tags=["fourDbody"],
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


@router.post("/run", response_model=Sam3DBodyRunOutput)
async def run_sam3d_body_pipeline(request: Request, payload: Sam3DBodyRunInput):
    """
    Run SAM 3D Body skeleton estimation on a video

    This endpoint runs asynchronously, it does not block lane-ball or other concurrent requests because the GPU work is dispatched to a thread-pool executor inside InferenceEngine
    """

    engine = _get_engine(request)
    settings = _get_settings(request)

    if not settings.username or not settings.password:
        raise HTTPException(
            status_code=500,
            detail="Missing API credentials. Set username/password in Ciclopes-API/.env or environment.",
        )

    logger.info("Running /fourdbody/run for video_key=%s", payload.video_key)

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

    fps = float(split_video.fps) if split_video.fps > 0 else 30.0

    if not split_video.frames:
        return Sam3DBodyRunOutput(fps=fps, skeleton_points=[])

    rgb_frames = [cv2.cvtColor(vf.image, cv2.COLOR_BGR2RGB) for vf in split_video.frames]

    try:
        sam3d_output = await engine.forward_sam3d_body(
            frames_rgb=rgb_frames,
            batch_size=settings.sam3d_body_batch_size,
        )
    except Exception as exc:
        logger.exception("/fourdbody/run pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # ── EMA smoothing to reduce jitter ────────────────────────────────────
    smoothed_output = ema_smooth_skeleton_frames(sam3d_output)

    # ── Parse results ─────────────────────────────────────────────────────
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
        for frame_joints in smoothed_output
    ]

    return Sam3DBodyRunOutput(fps=fps, skeleton_points=skeleton_points)
