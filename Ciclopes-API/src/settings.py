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
    lane_ball_start_frame: int = 10  # demo default — skip first N frames


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
    )
