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

import json
from contextlib import contextmanager
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

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


@contextmanager
def query_video_to_temp_file(
    *,
    input_path_or_url: str,
    verify: Optional[bool] = None,
) -> Iterator[Path]:
    """
    Resolve an input video reference (local path or HTTP(S) URL) into a local temp file.
    The temp file is cleaned up automatically when the context exits.
    """
    with tempfile.TemporaryDirectory(prefix="ciclopes-video-") as td:
        temp_path = Path(td) / "queried_video.mp4"
        candidate = input_path_or_url.strip()
        is_http = candidate.startswith("http://") or candidate.startswith("https://")

        if is_http:
            kwargs = {"stream": True, "timeout": 60}
            if verify is not None:
                kwargs["verify"] = verify

            resp = requests.get(candidate, **kwargs)
            if resp.status_code != 200:
                raise SpacesApiError(
                    f"Query video download failed: status={resp.status_code}, body={resp.text}"
                )

            with temp_path.open("wb") as out:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        out.write(chunk)
        else:
            source_path = Path(candidate)
            if not source_path.exists():
                raise FileNotFoundError(str(source_path))
            shutil.copy2(source_path, temp_path)

        yield temp_path


@contextmanager
def query_video_via_api_to_temp_file(
    *,
    base: str,
    verify_api: bool,
    username: str,
    password: str,
    key: str,
    ttl_seconds: int = 600,
    verify_presigned: Optional[bool] = None,
) -> Iterator[Path]:
    """
    Resolve a protected video object key to a local temp file using:
    authorize -> presign -> download.
    """
    with tempfile.TemporaryDirectory(prefix="ciclopes-video-") as td:
        temp_path = Path(td) / "queried_video.mp4"
        token = authorize(
            base=base,
            verify=verify_api,
            username=username,
            password=password,
        )
        presigned_url = get_presigned_url(
            base=base,
            verify=verify_api,
            token=token,
            key=key,
            ttl_seconds=ttl_seconds,
        )
        download_presigned_url_to_file(
            url=presigned_url,
            out_path=temp_path,
            verify=verify_presigned,
        )
        yield temp_path


def query_json_via_api(
    *,
    base: str,
    verify_api: bool,
    username: str,
    password: str,
    key: str,
    ttl_seconds: int = 600,
    verify_presigned: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Fetch a JSON object from the bucket using authorize -> presign -> download,
    returning the parsed dict directly (no temp file needed).
    """
    token = authorize(
        base=base,
        verify=verify_api,
        username=username,
        password=password,
    )
    presigned_url = get_presigned_url(
        base=base,
        verify=verify_api,
        token=token,
        key=key,
        ttl_seconds=ttl_seconds,
    )
    kwargs: dict = {"timeout": 60}
    if verify_presigned is not None:
        kwargs["verify"] = verify_presigned
    resp = requests.get(presigned_url, **kwargs)
    if resp.status_code != 200:
        raise SpacesApiError(f"JSON download failed: status={resp.status_code}, body={resp.text}")
    return resp.json()
