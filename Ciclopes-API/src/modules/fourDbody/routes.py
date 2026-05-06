from __future__ import annotations

import logging
import sys
import traceback

import cv2
from fastapi import APIRouter, HTTPException, Request

import importlib as _il

from core.SensorData.sensor_parser import compute_ball_contact_frame
from core.VideoUtil.FrameSplit import split_video_into_frames

from src.modules.fourDbody.models import (
    Sam3DBodyQueryInput,
    Sam3DBodyQueryOutput,
    Sam3DBodyRunInput,
    Sam3DBodyRunOutput,
    SkeletonPoint,
)

_ema_mod = _il.import_module("core.4DBody.emaSmoothing")
ema_smooth_skeleton_frames = _ema_mod.ema_smooth_skeleton_frames
from core.MockDB.mock_db_reader import load_named_runs, load_shots
from core.MockDB.mock_db_writer import default_run_name, save_named_run_section
from core.VideoUtil.SpacesApiClient import query_json_via_api, query_video_via_api_to_temp_file

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


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@router.post("/run", response_model=Sam3DBodyRunOutput)
async def run_sam3d_body_pipeline(request: Request, payload: Sam3DBodyRunInput):
    """
    Run SAM 3D Body skeleton estimation on a video

    This endpoint runs asynchronously, it does not block lane-ball or other concurrent requests because the GPU work is dispatched to a thread-pool executor inside InferenceEngine
    """
    fps = 30.0

    try:
        return await _run_sam3d_body_pipeline_inner(request, payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "UNHANDLED EXCEPTION in /fourdbody/run:\n%s",
            traceback.format_exc(),
        )
        sys.stdout.flush()
        return Sam3DBodyRunOutput(fps=fps, skeleton_points=[])


async def _run_sam3d_body_pipeline_inner(request: Request, payload: Sam3DBodyRunInput) -> Sam3DBodyRunOutput:
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
        split_video = split_video_into_frames(
            str(temp_path),
            max_frames=settings.max_video_frames,
            max_dimension=settings.max_video_dimension,
        )

    fps = float(split_video.fps) if split_video.fps > 0 else 30.0

    if not split_video.frames:
        return Sam3DBodyRunOutput(fps=fps, skeleton_points=[])

    rgb_frames = [cv2.cvtColor(vf.image, cv2.COLOR_BGR2RGB) for vf in split_video.frames]

    # Free the raw BGR frames — we only need rgb_frames from here on
    split_video.frames.clear()
    del split_video

    # ── Determine frame range from sensor data (or use all frames) ────────
    fourdbody_frames = rgb_frames
    if payload.sd_key != "key":
        logger.info("Fetching sensor JSON from bucket: sd_key=%s", payload.sd_key)
        try:
            sensor_data = query_json_via_api(
                base=settings.api_base,
                verify_api=settings.verify_api,
                username=settings.username,
                password=settings.password,
                key=payload.sd_key,
                ttl_seconds=settings.presign_ttl_seconds,
                verify_presigned=settings.verify_presigned,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Sensor JSON query failed: {exc}")

        sensor_info = compute_ball_contact_frame(sensor_data, fps)
        if sensor_info is not None:
            fourdbody_end = sensor_info.ball_contact_frame + int(fps)
            fourdbody_frames = rgb_frames[:fourdbody_end]
            logger.info(
                "Sensor-derived fourdbody_end=%d (of %d total)",
                fourdbody_end,
                len(rgb_frames),
            )
        else:
            logger.warning("Could not parse sensor data; falling back to all frames")

    try:
        sam3d_output = await engine.forward_sam3d_body(
            frames_rgb=fourdbody_frames,
            batch_size=settings.sam3d_body_batch_size,
        )
    except Exception as exc:
        logger.exception("/fourdbody/run pipeline failed")
        return Sam3DBodyRunOutput(fps=fps, skeleton_points=[])
    finally:
        rgb_frames.clear()
        del rgb_frames

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

    output = Sam3DBodyRunOutput(fps=fps, skeleton_points=skeleton_points)
    run_name = payload.save_name or default_run_name(payload.video_key)
    try:
        save_named_run_section(
            name=run_name,
            section="fourdbody",
            response=_model_to_dict(output),
            video_key=payload.video_key,
            sd_key=payload.sd_key,
        )
    except Exception:
        logger.exception("Failed to save fourdbody run name=%s", run_name)
    return output


@router.post("/query", response_model=Sam3DBodyQueryOutput)
async def query_sam3d_body_shots(payload: Sam3DBodyQueryInput):
    """
    Return saved pose data for one or more shot numbers from the mock DB.
    Missing shot numbers are silently omitted from the response.
    """
    records = load_shots("fourdbody", payload.shot_numbers)

    shots: dict[int, Sam3DBodyRunOutput] = {}
    for shot_number, rec in records.items():
        skeleton_points = [
            [
                SkeletonPoint(
                    joint_id=j["joint_id"],
                    x=j["x"],
                    y=j["y"],
                    z=j["z"],
                )
                for j in frame
            ]
            for frame in rec.get("skeleton_frames", [])
        ]
        shots[shot_number] = Sam3DBodyRunOutput(
            fps=rec.get("fps", 30.0),
            skeleton_points=skeleton_points,
        )

    runs: dict[str, Sam3DBodyRunOutput] = {}
    for name, rec in load_named_runs(payload.names).items():
        section = rec.get("sections", {}).get("fourdbody")
        if not section:
            continue
        try:
            runs[name] = Sam3DBodyRunOutput(**section)
        except Exception:
            logger.exception("Failed to parse saved fourdbody section for name=%s", name)

    return Sam3DBodyQueryOutput(shots=shots, runs=runs)
