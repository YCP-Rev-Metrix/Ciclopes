from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("ciclopes.mock_db_writer")

_MOCK_DB_DIR = Path(__file__).parent / "data"
_RUNS_DIR = _MOCK_DB_DIR / "runs"


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name.strip())
    return safe.strip("._") or "run"


def default_run_name(video_key: str) -> str:
    """
    Derive a stable local run name from the input object key.
    """
    stem = Path(video_key).stem or video_key
    return _safe_name(stem)


def save_named_run_section(
    *,
    name: str,
    section: str,
    response: dict[str, Any],
    video_key: str,
    sd_key: str | None = None,
) -> Path:
    """
    Persist one API response section under a named run.
    Re-running a section with the same name updates that section in-place while
    preserving any other sections already saved for the same run.
    """
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_name(name)
    out_path = _RUNS_DIR / f"{safe_name}.json"
    now = datetime.now(timezone.utc).isoformat()

    record: dict[str, Any]
    if out_path.exists():
        try:
            with out_path.open() as fh:
                record = json.load(fh)
        except Exception:
            logger.warning("mock_db: replacing unreadable saved run %s", out_path)
            record = {}
    else:
        record = {}

    record.setdefault("id", str(uuid.uuid4()))
    record.setdefault("created_at", now)
    record["updated_at"] = now
    record["name"] = name
    record["slug"] = safe_name
    record["video_key"] = video_key
    if sd_key is not None:
        record["sd_key"] = sd_key

    sections = record.setdefault("sections", {})
    sections[section] = response

    with out_path.open("w") as fh:
        json.dump(record, fh, indent=2)

    logger.info("mock_db: wrote named run section %s:%s", safe_name, section)
    return out_path


def save_mock_request(
    *,
    source: str,
    shot_number: int,
    video_key: str,
    fps: float,
    ball_positions: list[dict[str, float]],
    skeleton_frames: list[list[dict[str, Any]]],
) -> Path:
    """
    Persist ball positions and pose data for a single shot to a JSON file.
    Kinematics are intentionally excluded — they are recomputed at query time from ball_positions + fps.
    Files are keyed by source + shot_number and overwrite any prior save for that shot.

    Returns the path of the written file.
    """
    _MOCK_DB_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "shot_number": shot_number,
        "video_key": video_key,
        "fps": fps,
        "ball_positions": ball_positions,
        "skeleton_frames": skeleton_frames,
    }

    filename = f"{source}_shot{shot_number}.json"
    out_path = _MOCK_DB_DIR / filename

    with out_path.open("w") as fh:
        json.dump(record, fh, indent=2)

    logger.info("mock_db: wrote %s", out_path)
    return out_path
