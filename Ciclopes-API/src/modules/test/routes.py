from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pathlib import Path

from pydantic import BaseModel, Field
import torch

router = APIRouter(
    prefix="/test",
    tags=["test"],
)


def _get_inference_engine(request: Request):
    engine = getattr(request.app.state, "inference_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="InferenceEngine is not initialized")
    return engine


def _summarize_tensor_list(tensors: list[torch.Tensor]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for tensor in tensors:
        summaries.append(
            {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
        )
    return summaries

def assert_video_split():
    """
    Lightweight runtime test endpoint:
    - loads `test_video/test_vid.mp4`
    - splits into frames via OpenCV
    - returns basic assertions + diagnostics
    AI Generated test
    """
    try:
        from core.VideoUtil.TestVideoQuery import query_video as run_video_query
    except Exception as e:
        return {"ok": False, "stage": "import", "error": repr(e)}

    api_root = Path(__file__).resolve().parents[3]  # .../Ciclopes-API
    video_path = api_root / "test_video" / "test_vid.mp4"

    if not video_path.exists():
        return {
            "ok": False,
            "stage": "path",
            "error": "test video not found",
            "expected_path": str(video_path),
        }

    try:
        split_video = run_video_query(str(video_path))
    except Exception as e:
        return {
            "ok": False,
            "stage": "split",
            "error": repr(e),
            "video_path": str(video_path),
        }

    frame_count = len(split_video.frames)
    ok = (
        split_video.fps > 0
        and split_video.width > 0
        and split_video.height > 0
        and frame_count > 0
    )

    # Validate first frame basics (image is expected to be an OpenCV/numpy array).
    first_frame_shape = None
    if frame_count > 0:
        img = split_video.frames[0].image
        first_frame_shape = getattr(img, "shape", None)
        if first_frame_shape is not None and len(first_frame_shape) >= 2:
            ok = ok and int(first_frame_shape[0]) == int(split_video.height)
            ok = ok and int(first_frame_shape[1]) == int(split_video.width)

    return {
        "ok": bool(ok),
        "video_path": str(video_path),
        "fps": split_video.fps,
        "width": split_video.width,
        "height": split_video.height,
        "frame_count": frame_count,
        "first_frame_shape": first_frame_shape,
        "first_timestamp_s": split_video.frames[0].timestamp if frame_count else None,
        "last_timestamp_s": split_video.frames[-1].timestamp if frame_count else None,
    }


class QueryVideoInferenceInput(BaseModel):
    input_path_or_url: str | None = Field(
        default=None,
        description="Local filesystem path or direct HTTP(S) URL",
    )
    verify_tls: bool | None = Field(
        default=None,
        description="Optional TLS verification override for direct HTTP(S) URLs",
    )
    api_base: str | None = Field(
        default=None,
        description="Base API URL for auth/presign flow, e.g. https://api.revmetrix.io",
    )
    username: str | None = Field(default=None, description="API username for authorize endpoint")
    password: str | None = Field(default=None, description="API password for authorize endpoint")
    video_key: str | None = Field(
        default=None,
        description="Object key/path passed to /api/videos/presign (ex: videos/abc.mp4)",
    )
    verify_api: bool = Field(default=True, description="TLS verification for authorize/presign requests")
    verify_presigned: bool | None = Field(
        default=None,
        description="Optional TLS verification override for the presigned download URL",
    )
    ttl_seconds: int = Field(default=600, ge=1, le=86400)


def assert_query_split_batch16_infer(request: Request, payload: QueryVideoInferenceInput):
    try:
        from core.VideoUtil.FrameSplit import batch_split_video, split_video_into_frames
        from core.VideoUtil.SpacesApiClient import (
            query_video_to_temp_file,
            query_video_via_api_to_temp_file,
        )
    except Exception as e:
        return {"ok": False, "stage": "import", "error": repr(e)}

    engine = _get_inference_engine(request)
    batch_size = 16
    use_api_query = bool(payload.api_base and payload.username and payload.password and payload.video_key)
    use_direct_query = bool(payload.input_path_or_url)
    if not use_api_query and not use_direct_query:
        return {
            "ok": False,
            "stage": "input",
            "error": (
                "Provide either input_path_or_url for direct query OR "
                "api_base + username + password + video_key for auth/presign query"
            ),
        }

    source_mode = "api_presign" if use_api_query else "direct"
    try:
        if use_api_query:
            temp_video_ctx = query_video_via_api_to_temp_file(
                base=payload.api_base or "",
                verify_api=payload.verify_api,
                username=payload.username or "",
                password=payload.password or "",
                key=payload.video_key or "",
                ttl_seconds=payload.ttl_seconds,
                verify_presigned=payload.verify_presigned,
            )
        else:
            temp_video_ctx = query_video_to_temp_file(
                input_path_or_url=payload.input_path_or_url or "",
                verify=payload.verify_tls,
            )

        with temp_video_ctx as temp_video_path:
            split_video = split_video_into_frames(str(temp_video_path))
            frame_count = len(split_video.frames)
            batches = batch_split_video(
                split_video,
                batch_size=batch_size,
                rgb=True,
                normalize=True,
                drop_last=True,
            )
            output = engine.forward(batches)
    except Exception as e:
        return {"ok": False, "stage": "query/split/batch/infer", "error": repr(e)}

    output_batches = output.get("test")
    if not isinstance(output_batches, list) or not all(isinstance(t, torch.Tensor) for t in output_batches):
        return {"ok": False, "stage": "infer", "error": "InferenceEngine returned unexpected structure"}

    expected_batches = frame_count // batch_size
    same_count = len(output_batches) == len(batches) == expected_batches
    all_shapes_ok = all(
        list(t.shape) == [batch_size, 3, split_video.height, split_video.width] for t in output_batches
    )
    input_path = (payload.input_path_or_url or "").strip()
    is_http = input_path.startswith("http://") or input_path.startswith("https://")

    return {
        "ok": bool(frame_count > 0 and same_count and all_shapes_ok),
        "source_mode": source_mode,
        "input_path_or_url": input_path if input_path else None,
        "input_type": ("url" if is_http else "path") if input_path else None,
        "api_base": payload.api_base,
        "video_key": payload.video_key,
        "engine_status": engine.status(),
        "fps": float(split_video.fps),
        "width": int(split_video.width),
        "height": int(split_video.height),
        "frame_count": int(frame_count),
        "batch_size": batch_size,
        "expected_batches": int(expected_batches),
        "input_batches": int(len(batches)),
        "output_batches": int(len(output_batches)),
        "output_batch_summaries": _summarize_tensor_list(output_batches),
        "note": "drop_last=True; videos with fewer than 16 frames yield zero batches",
    }


def assert_inference_engine_batch_roundtrip(
    request: Request,
    batch_count: int = 2,
    batch_size: int = 4,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
):
    if batch_count <= 0 or batch_size <= 0 or channels <= 0 or height <= 0 or width <= 0:
        return {"ok": False, "error": "All dimensions and counts must be > 0"}

    engine = _get_inference_engine(request)
    input_batches = [
        torch.rand(batch_size, channels, height, width, dtype=torch.float32)
        for _ in range(batch_count)
    ]

    output = engine.forward(input_batches)
    output_batches = output.get("test")
    if not isinstance(output_batches, list) or not all(isinstance(t, torch.Tensor) for t in output_batches):
        return {"ok": False, "error": "InferenceEngine returned unexpected structure"}

    same_count = len(output_batches) == len(input_batches)
    same_shapes = all(out.shape == inp.shape for out, inp in zip(output_batches, input_batches))

    return {
        "ok": bool(same_count and same_shapes),
        "engine_status": engine.status(),
        "input_batch_count": len(input_batches),
        "output_batch_count": len(output_batches),
        "input_batches": _summarize_tensor_list(input_batches),
        "output_batches": _summarize_tensor_list(output_batches),
    }


def assert_inference_engine_echo_payload(request: Request, payload: dict[str, Any]):
    engine = _get_inference_engine(request)
    output = engine.forward(payload)
    return {
        "ok": True,
        "engine_status": engine.status(),
        "input": payload,
        "output": output,
    }

@router.get("/health")
async def health():
    return {"ok": True}

@router.get("/query_video")
async def query_video():
    return assert_video_split()

@router.post("/query_split_batch16_inference_test")
async def query_split_batch16_inference_test(
    request: Request, payload: QueryVideoInferenceInput
):
    return assert_query_split_batch16_infer(request=request, payload=payload)


@router.get("/inference_engine_status")
async def inference_engine_status(request: Request):
    engine = _get_inference_engine(request)
    return {"ok": True, "engine_status": engine.status()}


@router.get("/inference_engine_batch_roundtrip")
async def inference_engine_batch_roundtrip(
    request: Request,
    batch_count: int = 2,
    batch_size: int = 4,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
):
    return assert_inference_engine_batch_roundtrip(
        request=request,
        batch_count=batch_count,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
    )


@router.post("/inference_engine_echo")
async def inference_engine_echo(request: Request, payload: dict[str, Any]):
    return assert_inference_engine_echo_payload(request=request, payload=payload)
