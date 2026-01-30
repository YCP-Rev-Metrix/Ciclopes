from fastapi import APIRouter
from pathlib import Path

router = APIRouter(
    prefix="/test",
    tags=["test"],
)

def assert_video_split():
    """
    Lightweight runtime test endpoint:
    - loads `test_video/test_vid.mp4`
    - splits into frames via OpenCV
    - returns basic assertions + diagnostics
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

@router.get("/health")
async def health():
    return {"ok": True}

@router.get("/query_video")
async def query_video():
    return assert_video_split()