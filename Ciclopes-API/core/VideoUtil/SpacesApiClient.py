"""
Small helper for working with DigitalOcean Spaces via *your API's* presign endpoints.

Your teammate's `ExampleBucketSetup.py` proves the flow end-to-end by calling:
- POST /api/posts/Authorize            -> returns tokenA
- POST /api/videos/upload             -> returns object key
- GET  /api/videos/presign            -> returns a presigned download URL
- GET  <presigned URL>                -> downloads bytes directly from Spaces

This module wraps the same calls, but:
- takes base/verify as arguments (no globals)
- raises exceptions instead of sys.exit (so it can be used inside FastAPI endpoints)

AI Generated -- needs verified / cleaned
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests


class SpacesApiError(RuntimeError):
    pass


def authorize(*, base: str, verify: bool, username: str, password: str) -> str:
    resp = requests.post(
        f"{base}/api/posts/Authorize",
        json={"username": username, "password": password},
        headers={"Content-Type": "application/json"},
        verify=verify,
        timeout=30,
    )
    if resp.status_code != 200:
        raise SpacesApiError(f"Authorize failed: status={resp.status_code}, body={resp.text}")
    token = (resp.json() or {}).get("tokenA")
    if not token:
        raise SpacesApiError("Authorize returned no tokenA")
    return token


def upload_video(
    *,
    base: str,
    verify: bool,
    token: str,
    file_path: Path,
    folder: str = "videos",
) -> str:
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    headers = {"Authorization": f"Bearer {token}"}
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f, "video/mp4")}
        resp = requests.post(
            f"{base}/api/videos/upload",
            params={"folder": folder},
            headers=headers,
            files=files,
            verify=verify,
            timeout=60,
        )
    if resp.status_code != 200:
        raise SpacesApiError(f"Upload failed: status={resp.status_code}, body={resp.text}")
    key = (resp.json() or {}).get("key")
    if not key:
        raise SpacesApiError("Upload response missing 'key'")
    return key


def get_presigned_url(
    *,
    base: str,
    verify: bool,
    token: str,
    key: str,
    ttl_seconds: int = 600,
) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{base}/api/videos/presign",
        params={"key": key, "ttlSeconds": ttl_seconds},
        headers=headers,
        verify=verify,
        timeout=30,
    )
    if resp.status_code != 200:
        raise SpacesApiError(f"Presign failed: status={resp.status_code}, body={resp.text}")
    url = (resp.json() or {}).get("url")
    if not url:
        raise SpacesApiError("Presign response missing 'url'")
    return url


def download_presigned_url_to_file(
    *,
    url: str,
    out_path: Path,
    verify: Optional[bool] = None,
) -> None:
    kwargs = {"stream": True, "timeout": 60}
    if verify is not None:
        kwargs["verify"] = verify

    resp = requests.get(url, **kwargs)
    if resp.status_code != 200:
        raise SpacesApiError(f"Spaces download failed: status={resp.status_code}, body={resp.text}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                out.write(chunk)

