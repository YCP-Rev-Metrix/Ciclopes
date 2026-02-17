from __future__ import annotations

import logging
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
