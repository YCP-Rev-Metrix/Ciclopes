from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("ciclopes.mock_db_reader")

_MOCK_DB_DIR = Path(__file__).parent / "data"


def load_shot(source: str, shot_number: int) -> dict[str, Any] | None:
    """
    Load a saved mock-DB record for the given source and shot number.
    Returns None if the file does not exist.
    """
    path = _MOCK_DB_DIR / f"{source}_shot{shot_number}.json"
    if not path.exists():
        logger.warning("mock_db: no file for source=%s shot=%d", source, shot_number)
        return None

    with path.open() as fh:
        return json.load(fh)


def load_shots(source: str, shot_numbers: list[int]) -> dict[int, dict[str, Any]]:
    """
    Load multiple shots for a source. Missing shots are omitted from the result.
    """
    result: dict[int, dict[str, Any]] = {}
    for n in shot_numbers:
        record = load_shot(source, n)
        if record is not None:
            result[n] = record
    return result
