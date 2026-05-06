from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("ciclopes.mock_db_reader")

_MOCK_DB_DIR = Path(__file__).parent / "data"
_RUNS_DIR = _MOCK_DB_DIR / "runs"


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


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name.strip())
    return safe.strip("._") or "run"


def list_saved_run_names() -> list[str]:
    """
    Return names for locally saved named run records.
    """
    if not _RUNS_DIR.exists():
        return []

    names: list[str] = []
    for path in sorted(_RUNS_DIR.glob("*.json")):
        try:
            with path.open() as fh:
                record = json.load(fh)
        except Exception:
            logger.warning("mock_db: failed to read saved run %s", path)
            continue
        name = record.get("name") or path.stem
        names.append(str(name))
    return names


def load_named_run(name: str) -> dict[str, Any] | None:
    """
    Load one named saved run record. Returns None if it does not exist.
    """
    path = _RUNS_DIR / f"{_safe_name(name)}.json"
    if not path.exists():
        logger.warning("mock_db: no saved run named=%s", name)
        return None

    with path.open() as fh:
        return json.load(fh)


def load_named_runs(names: list[str]) -> dict[str, dict[str, Any]]:
    """
    Load multiple named runs. Missing names are omitted.
    """
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        record = load_named_run(name)
        if record is not None:
            result[name] = record
    return result
