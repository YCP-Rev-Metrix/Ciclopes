from __future__ import annotations

import logging
import time
from typing import Any

import cv2
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("ciclopes.test_routes")

router = APIRouter(
    prefix="/test",
    tags=["test"],
)


# ── Request model ─────────────────────────────────────────────────────────────

class InferenceTestInput(BaseModel):
    """
    JSON body for the end-to-end inference test.

    Provide RevMetrix API credentials + video key.  TLS verification and TTL
    are hardcoded for this test endpoint.
    """
    api_base: str = Field(..., description="Base API URL, e.g. https://api.revmetrix.io")
    username: str = Field(..., description="API username for /api/posts/Authorize")
    password: str = Field(..., description="API password")
    video_key: str = Field(..., description="Object key for /api/videos/presign, e.g. videos/abc.mp4")


class LaneBallPostprocessTestInput(InferenceTestInput):
    start_frame: int = Field(
        0,
        ge=0,
        description="Frame index where homography search begins",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_engine(request: Request):
    """Retrieve the InferenceEngine from app state or 503."""
    engine = getattr(request.app.state, "inference_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="InferenceEngine is not initialized")
    return engine


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    """Simple liveness check — no model interaction."""
    return {"ok": True}


@router.get("/engine_status")
async def engine_status(request: Request):
    """Return InferenceEngine device info and VRAM usage."""
    engine = _get_engine(request)
    return {"ok": True, "engine_status": engine.status()}


@router.post("/run_inference")
async def run_inference_test(request: Request, payload: InferenceTestInput):
    """
    End-to-end inference test.

    Pipeline:
        1. Query video via RevMetrix API auth → presigned download
        2. Split video into frames (OpenCV)
        3. Convert frames to RGB numpy arrays
        4. Run InferenceEngine.forward() — YOLO seg on first frame
        5. Return structured JSON with first-frame masks/detections

    Hardcoded settings for this test:
        verify_api      = True
        verify_presigned = True
        ttl_seconds     = 3600
    """
    # ── Lazy imports so the module loads fast even without all deps ────────
    from core.VideoUtil.FrameSplit import split_video_into_frames
    from core.VideoUtil.SpacesApiClient import query_video_via_api_to_temp_file

    engine = _get_engine(request)

    # ── 1. Query video via API auth flow ──────────────────────────────────
    logger.info(
        "Querying video: api_base=%s video_key=%s",
        payload.api_base,
        payload.video_key,
    )

    try:
        temp_ctx = query_video_via_api_to_temp_file(
            base=payload.api_base,
            verify_api=True,
            username=payload.username,
            password=payload.password,
            key=payload.video_key,
            ttl_seconds=3600,
            verify_presigned=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Video query failed: {exc}")

    # ── 2. Split video into frames ────────────────────────────────────────
    with temp_ctx as temp_path:
        logger.info("Splitting video: %s", temp_path)
        split_video = split_video_into_frames(str(temp_path))

    frame_count = len(split_video.frames)
    if frame_count == 0:
        return {
            "ok": False,
            "error": "Video produced zero frames",
            "video_info": {
                "fps": split_video.fps,
                "width": split_video.width,
                "height": split_video.height,
                "frame_count": 0,
            },
        }

    # ── 3. Convert frames to RGB numpy arrays ────────────────────────────
    # OpenCV reads as BGR — convert each frame to RGB for YOLO inference.
    rgb_frames: list = []
    for vf in split_video.frames:
        rgb_frames.append(cv2.cvtColor(vf.image, cv2.COLOR_BGR2RGB))

    logger.info(
        "Video split complete: %d frames, %dx%d @ %.1f fps",
        frame_count,
        split_video.width,
        split_video.height,
        split_video.fps,
    )

    # ── 4. Run inference ──────────────────────────────────────────────────
    logger.info("Running LaneBall inference on %d frames (first-frame infer)...", frame_count)

    try:
        inference_output = await engine.forward(rgb_frames)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    # ── 5. Build response ─────────────────────────────────────────────────
    return {
        "ok": True,

        "video_info": {
            "fps": float(split_video.fps),
            "width": int(split_video.width),
            "height": int(split_video.height),
            "frame_count": frame_count,
        },

        # First-frame segmentation masks from YOLO (ball / lane / pins)
        "first_frame_segmentation": inference_output["segmentation"],

        "engine_status": engine.status(),
    }


@router.post("/run_lane_ball_postprocessing")
async def run_lane_ball_postprocessing_test(
    request: Request, payload: LaneBallPostprocessTestInput
):
    """
    End-to-end lane-ball test:
      1. Query and split video
      2. Run lane/ball segmentation on all frames
      3. Find first trapezoidal center-lane segmentation from start_frame
      4. Compute homography, project ball positions (x,y in meters), and kinematics
    """
    from core.VideoUtil.FrameSplit import split_video_into_frames
    from core.VideoUtil.SpacesApiClient import query_video_via_api_to_temp_file

    request_t0 = time.perf_counter()
    engine = _get_engine(request)

    try:
        temp_ctx = query_video_via_api_to_temp_file(
            base=payload.api_base,
            verify_api=True,
            username=payload.username,
            password=payload.password,
            key=payload.video_key,
            ttl_seconds=3600,
            verify_presigned=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Video query failed: {exc}")

    with temp_ctx as temp_path:
        split_video = split_video_into_frames(str(temp_path))

    frame_count = len(split_video.frames)
    fps = float(split_video.fps) if split_video.fps > 0 else 30.0

    if frame_count == 0:
        return {
            "ok": False,
            "error": "Video produced zero frames",
            "video_info": {
                "fps": fps,
                "width": int(split_video.width),
                "height": int(split_video.height),
                "frame_count": 0,
                "start_frame": payload.start_frame,
            },
        }

    if payload.start_frame >= frame_count:
        raise HTTPException(
            status_code=422,
            detail=f"start_frame={payload.start_frame} is out of range (frame_count={frame_count})",
        )

    rgb_frames = [cv2.cvtColor(vf.image, cv2.COLOR_BGR2RGB) for vf in split_video.frames]

    try:
        lane_ball_output = await engine.forward_lane_ball(
            frames_rgb=rgb_frames,
            fps=fps,
            start_frame=payload.start_frame,
        )
    except Exception as exc:
        logger.exception("LaneBall postprocessing test failed")
        raise HTTPException(status_code=500, detail=f"LaneBall pipeline error: {exc}")

    total_ms = (time.perf_counter() - request_t0) * 1000.0
    pipeline_ok = "error" not in lane_ball_output.get("health", {})

    return {
        "ok": pipeline_ok,
        "video_info": {
            "fps": fps,
            "width": int(split_video.width),
            "height": int(split_video.height),
            "frame_count": frame_count,
            "start_frame": payload.start_frame,
        },
        "error": lane_ball_output["health"].get("error"),
        "is_trapezoid": lane_ball_output["is_trapezoid"],
        "homography_frame": lane_ball_output["homography_frame"],
        "ball_positions": lane_ball_output["positions"],
        "kinematics": lane_ball_output["kinematics"],
        "homography": {
            "src_corners": lane_ball_output["homography_src_corners"],
            "dst_corners_m": lane_ball_output["homography_dst_corners_m"],
            "matrix": lane_ball_output["homography_matrix"],
        },
        "health": {
            **lane_ball_output["health"],
            "total_ms": round(total_ms, 2),
        },
        "engine_status": engine.status(),
    }
