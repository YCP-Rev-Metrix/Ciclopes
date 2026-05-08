from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    api_base: str
    username: str
    password: str
    verify_api: bool = True
    verify_presigned: bool = True
    presign_ttl_seconds: int = 3600
    lane_ball_batch_size: int = 32
    sam3d_body_batch_size: int = 4
    lane_ball_start_frame: int = 40  # demo default — skip first N frames
    force_lane_ball_start_frame: bool = True  # if True, ignore sd_key sensor-derived start and always use lane_ball_start_frame
    max_video_frames: int = 600     # OOM protection: cap extracted frames
    max_video_dimension: int = 1024 # OOM protection: downscale longest edge
    lane_ball_max_video_dimension: int = 1280
    multi_gpu: bool = False


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key)
        if value is not None and value != "":
            return value
    return default


def load_app_settings() -> AppSettings:
    api_root = Path(__file__).resolve().parents[1]
    _load_dotenv(api_root / ".env")

    ttl_raw = _env("PRESIGN_TTL_SECONDS", "presign_ttl_seconds", default="3600")
    try:
        ttl = int(ttl_raw)
    except ValueError:
        ttl = 3600

    lb_bs_raw = _env("LANE_BALL_BATCH_SIZE", default="32")
    try:
        lb_bs = max(int(lb_bs_raw), 1)
    except ValueError:
        lb_bs = 32

    sam3d_bs_raw = _env("SAM3D_BODY_BATCH_SIZE", default="4")
    try:
        sam3d_bs = max(int(sam3d_bs_raw), 1)
    except ValueError:
        sam3d_bs = 4

    max_frames_raw = _env("MAX_VIDEO_FRAMES", default="600")
    try:
        max_frames = max(int(max_frames_raw), 1)
    except ValueError:
        max_frames = 600

    max_dim_raw = _env("MAX_VIDEO_DIMENSION", default="1024")
    try:
        max_dim = max(int(max_dim_raw), 128)
    except ValueError:
        max_dim = 1024

    lb_max_dim_raw = _env("LANE_BALL_MAX_VIDEO_DIMENSION", default="1280")
    try:
        lb_max_dim = max(int(lb_max_dim_raw), 128)
    except ValueError:
        lb_max_dim = 1280

    multi_gpu = _parse_bool(_env("MULTI_GPU", default="false"), default=False)
    force_lb_start = _parse_bool(
        _env("FORCE_LANE_BALL_START_FRAME", default="false"), default=False
    )

    return AppSettings(
        api_base=_env("API_BASE", "api_base", default="https://api.revmetrix.io"),
        username=_env("API_USERNAME", "USERNAME", "username"),
        password=_env("API_PASSWORD", "PASSWORD", "password"),
        verify_api=_parse_bool(_env("VERIFY_API", "verify_api", default="true"), default=True),
        verify_presigned=_parse_bool(
            _env("VERIFY_PRESIGNED", "verify_presigned", default="true"), default=True
        ),
        presign_ttl_seconds=max(ttl, 1),
        lane_ball_batch_size=lb_bs,
        sam3d_body_batch_size=sam3d_bs,
        max_video_frames=max_frames,
        max_video_dimension=max_dim,
        lane_ball_max_video_dimension=lb_max_dim,
        multi_gpu=multi_gpu,
        force_lane_ball_start_frame=force_lb_start,
    )
