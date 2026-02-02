from fastapi import APIRouter
import os
import tempfile
from pathlib import Path

router = APIRouter(
    prefix="/test",
    tags=["test"],
)

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

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


def assert_example_bucket_setup(batch_size: int = 8, ttl_seconds: int = 600):
    """
    End-to-end *bucket* smoke test for the inference plumbing:

    - authenticate (JWT)
    - presign a Spaces object (optionally upload a test video first)
    - download to a temp file
    - split into frames via OpenCV (SplitVideo)
    - batch frames into torch tensors ready for inference
    - assert shapes/dtypes/ranges

    Env (no secrets in code):
    - REV_TEST_BASE (default: https://api.revmetrix.io)
    - REV_TEST_VERIFY (default: true) - TLS verify when calling REV_TEST_BASE
    - REV_TEST_USERNAME / REV_TEST_PASSWORD

    Optional:
    - REV_TEST_VIDEO_KEY: if set, skip upload and just presign this key
    - REV_TEST_UPLOAD_FOLDER (default: videos)
    - REV_TEST_PRESIGNED_VERIFY (default: true) - TLS verify for the presigned download URL

    AI Generated test
    """
    try:
        from core.VideoUtil.SpacesApiClient import (
            authorize,
            upload_video,
            get_presigned_url,
            download_presigned_url_to_file,
        )
        from core.VideoUtil.FrameSplit import split_video_into_frames, batch_split_video
    except Exception as e:
        return {"ok": False, "stage": "import", "error": repr(e)}

    base = "https://api.revmetrix.io"
    verify_api = True
    verify_presigned = True

    username = "string"
    password = "string"
    if not username or not password:
        return {"ok": False, "stage": "env", "error": "Missing REV_TEST_USERNAME / REV_TEST_PASSWORD"}

    if batch_size <= 0:
        return {"ok": False, "stage": "params", "error": "batch_size must be > 0"}

    existing_key = None
    upload_folder = "videos"

    api_root = Path(__file__).resolve().parents[3]  # .../Ciclopes-API
    input_video = api_root / "test_video" / "test_vid.mp4"
    if not input_video.exists() and not existing_key:
        return {
            "ok": False,
            "stage": "path",
            "error": "test video not found (and no REV_TEST_VIDEO_KEY provided)",
            "expected_path": str(input_video),
        }

    try:
        token = authorize(base=base, verify=verify_api, username=username, password=password)
    except Exception as e:
        return {"ok": False, "stage": "authorize", "error": repr(e), "base": base}

    try:
        if existing_key:
            key = existing_key
            uploaded = False
        else:
            key = upload_video(
                base=base,
                verify=verify_api,
                token=token,
                file_path=input_video,
                folder=upload_folder,
            )
            uploaded = True
    except Exception as e:
        return {"ok": False, "stage": "upload", "error": repr(e), "base": base}

    try:
        url = get_presigned_url(
            base=base,
            verify=verify_api,
            token=token,
            key=key,
            ttl_seconds=ttl_seconds,
        )
    except Exception as e:
        return {"ok": False, "stage": "presign", "error": repr(e), "base": base, "key": key}

    # Windows-friendly: create a real temp path, close it, then let OpenCV open by path.
    try:
        with tempfile.TemporaryDirectory(prefix="ciclopes-video-") as td:
            temp_path = Path(td) / "downloaded.mp4"
            download_presigned_url_to_file(url=url, out_path=temp_path, verify=verify_presigned)

            split_video = split_video_into_frames(str(temp_path))
            frame_count = len(split_video.frames)

            # Convert frames into inference-ready tensor batches.
            batches = batch_split_video(split_video, batch_size=batch_size, rgb=True, normalize=True)
    except Exception as e:
        return {"ok": False, "stage": "download/split/batch", "error": repr(e)}

    # --- Assertions / diagnostics ---
    ok = True
    ok = ok and split_video.fps > 0
    ok = ok and split_video.width > 0 and split_video.height > 0
    ok = ok and frame_count > 0

    expected_batches = frame_count // batch_size
    ok = ok and len(batches) == expected_batches

    first_frame_shape = None
    if frame_count:
        img0 = split_video.frames[0].image
        first_frame_shape = getattr(img0, "shape", None)
        if first_frame_shape is not None and len(first_frame_shape) >= 2:
            ok = ok and int(first_frame_shape[0]) == int(split_video.height)
            ok = ok and int(first_frame_shape[1]) == int(split_video.width)

    first_batch_shape = None
    first_batch_dtype = None
    first_batch_minmax = None
    if batches:
        b0 = batches[0]
        first_batch_shape = list(b0.shape)
        first_batch_dtype = str(b0.dtype)

        # Expected: (B, 3, H, W) float32 normalized to [0, 1].
        ok = ok and b0.ndim == 4
        ok = ok and int(b0.shape[0]) == int(batch_size)
        ok = ok and int(b0.shape[1]) == 3
        ok = ok and int(b0.shape[2]) == int(split_video.height)
        ok = ok and int(b0.shape[3]) == int(split_video.width)

        # Sample to keep this endpoint cheap even for long videos.
        sample = b0[:1, :, :32, :32]
        mn = float(sample.min().item())
        mx = float(sample.max().item())
        first_batch_minmax = [mn, mx]
        ok = ok and mn >= 0.0 and mx <= 1.0

    return {
        "ok": bool(ok),
        "base": base,
        "verify_api": bool(verify_api),
        "verify_presigned": bool(verify_presigned),
        "uploaded_test_video": bool(uploaded),
        "key": key,
        "ttl_seconds": int(ttl_seconds),
        "fps": float(split_video.fps),
        "width": int(split_video.width),
        "height": int(split_video.height),
        "frame_count": int(frame_count),
        "batch_size": int(batch_size),
        "expected_batches": int(expected_batches),
        "batches": int(len(batches)),
        "first_frame_shape": first_frame_shape,
        "first_batch_shape": first_batch_shape,
        "first_batch_dtype": first_batch_dtype,
        "first_batch_sample_minmax": first_batch_minmax,
        "first_timestamp_s": split_video.frames[0].timestamp if frame_count else None,
        "last_timestamp_s": split_video.frames[-1].timestamp if frame_count else None,
    }

@router.get("/health")
async def health():
    return {"ok": True}

@router.get("/query_video")
async def query_video():
    return assert_video_split()

@router.get("/example_bucket_setup_test")
async def example_bucket_setup_test(batch_size: int = 8, ttl_seconds: int = 600):
    return assert_example_bucket_setup(batch_size=batch_size, ttl_seconds=ttl_seconds)