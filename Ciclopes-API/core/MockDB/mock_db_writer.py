from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("ciclopes.mock_db_writer")

_MOCK_DB_DIR = Path(__file__).parent / "data"


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
