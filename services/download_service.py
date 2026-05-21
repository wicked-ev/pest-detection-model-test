"""Robust HTTP download service with retries, backoff, streaming and progress.

Designed for constrained devices (Raspberry Pi 3 B+):
- Stream to disk in chunks
- Atomic writes via .part temporary files
- Exponential backoff retry
- Optional resume when server supports Range requests
- Lightweight terminal progress indicator
"""
from __future__ import annotations

import os
import time
import math
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    pass


class DownloadService:
    def __init__(self, *, timeout: int = 30, max_retries: int = 4, backoff_base: float = 1.0, chunk_size: int = 64 * 1024):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.chunk_size = chunk_size

    def _format_size(self, n: float) -> str:
        if n is None:
            return "?"
        for unit in ["B", "KB", "MB", "GB"]:
            if n < 1024.0:
                return f"{n:3.1f}{unit}"
            n /= 1024.0
        return f"{n:.1f}TB"

    def _print_progress(self, downloaded: int, total: Optional[int], start_ts: float):
        # Throttle updates so terminal isn't flooded
        now = time.monotonic()
        elapsed = now - start_ts if start_ts else 0.0
        speed = downloaded / elapsed if elapsed > 0 else 0.0
        if total:
            perc = downloaded / total * 100.0
            eta = (total - downloaded) / speed if speed > 0 else None
            eta_s = f"{int(eta)}s" if eta is not None else "?"
            line = f"Downloading: {perc:5.1f}% {self._format_size(downloaded)}/{self._format_size(total)} @ {self._format_size(speed)}/s ETA {eta_s}"
        else:
            line = f"Downloading: {self._format_size(downloaded)} downloaded @ {self._format_size(speed)}/s"
        # Use carriage return to overwrite line; ensure flush
        print(line + "\r", end="", flush=True)

    def download(self, url: str, dest_path: Path) -> None:
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        attempt = 0
        while attempt <= self.max_retries:
            attempt += 1
            try:
                headers = {}
                resume_pos = 0
                if temp_path.exists():
                    resume_pos = temp_path.stat().st_size
                    if resume_pos > 0:
                        headers["Range"] = f"bytes={resume_pos}-"

                with requests.get(url, stream=True, timeout=self.timeout, headers=headers) as r:
                    r.raise_for_status()

                    total = None
                    if "Content-Length" in r.headers:
                        total = int(r.headers.get("Content-Length", 0))
                        if "Range" in headers and r.status_code == 206:
                            # when resuming, Content-Length is remaining bytes
                            total = resume_pos + total

                    # start writing
                    mode = "ab" if resume_pos and r.status_code == 206 else "wb"
                    downloaded = resume_pos if mode == "ab" else 0
                    start_ts = time.monotonic()

                    with open(temp_path, mode) as fh:
                        last_update = 0.0
                        for chunk in r.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                fh.write(chunk)
                                downloaded += len(chunk)
                                now = time.monotonic()
                                # update progress at most 5 times/sec
                                if now - last_update >= 0.2:
                                    self._print_progress(downloaded, total, start_ts)
                                    last_update = now
                        # final progress
                        self._print_progress(downloaded, total, start_ts)
                        print("")

                # validate size
                if total is not None and temp_path.stat().st_size != total:
                    msg = f"Downloaded size mismatch (got={temp_path.stat().st_size} expected={total})"
                    logger.warning(msg)
                    raise DownloadError(msg)

                # atomic rename
                os.replace(str(temp_path), str(dest_path))
                logger.info("Downloaded %s -> %s", url, dest_path)
                return

            except Exception as exc:
                # cleanup partial if last attempt or non-resumable
                logger.exception("Download attempt %d failed for %s", attempt, url)
                if attempt > self.max_retries:
                    # final failure: remove partial file
                    try:
                        if temp_path.exists():
                            temp_path.unlink()
                    except Exception:
                        logger.exception("Failed to remove partial file %s", temp_path)
                    raise DownloadError(f"Failed to download {url}: {exc}")

                # exponential backoff with jitter
                backoff = self.backoff_base * (2 ** (attempt - 1))
                jitter = backoff * 0.1 * (0.5 - (time.time() % 1))
                wait = max(0.5, backoff + jitter)
                logger.info("Retrying in %.1f seconds... (attempt %d/%d)", wait, attempt, self.max_retries)
                time.sleep(wait)

*** End Patch