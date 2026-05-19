"""Camera management service optimized for Raspberry Pi 3 B+.

Responsibilities:
- Initialize and validate camera configuration
- Provide a low-latency capture loop that always exposes the latest frame
- Lightweight test mode for health checks and optional debug frame saving
- Graceful shutdown, reconnect logic, and safety protections

Design notes:
- A single-frame slot (latest frame) is used to avoid queue buildup and copies.
- A short-grace reconnect loop prevents dead capture streams from stalling.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from pathlib import Path

import cv2
import numpy as np

import configs

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    device: int = configs.CAMERA_DEVICE
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1  # keep only the newest frame
    frame_format: str = "bgr"  # or 'rgb'
    read_timeout: float = 3.0  # seconds to wait for a first frame

    def __post_init__(self) -> None:
        if not isinstance(self.device, int) or self.device < 0:
            raise ValueError(f"Camera device must be a non-negative integer, got {self.device}")
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError(f"Camera width must be a positive integer, got {self.width}")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError(f"Camera height must be a positive integer, got {self.height}")
        if not isinstance(self.fps, int) or self.fps <= 0 or self.fps > 60:
            raise ValueError(f"Camera fps must be an integer between 1 and 60, got {self.fps}")
        if not isinstance(self.buffer_size, int) or self.buffer_size <= 0 or self.buffer_size > 5:
            raise ValueError(f"Camera buffer_size must be an integer between 1 and 5, got {self.buffer_size}")
        if self.frame_format not in {"bgr", "rgb"}:
            raise ValueError(f"Camera frame_format must be 'bgr' or 'rgb', got {self.frame_format}")
        if not isinstance(self.read_timeout, (int, float)) or self.read_timeout <= 0:
            raise ValueError(f"Camera read_timeout must be positive, got {self.read_timeout}")


class CameraService:
    """Camera service providing a low-latency newest-frame buffer.

    Usage:
    - create instance
    - call `start()` to begin capture thread
    - use `get_latest(copy=False)` to obtain newest frame reference
    - call `stop()` to shutdown and release resources
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: Optional[float] = None
        self._latest_frame_id: int = 0
        self._opened = False

    def _open_capture(self) -> bool:
        logger.info(f"Opening camera device {self.config.device}")
        cap = cv2.VideoCapture(self.config.device)
        if not cap.isOpened():
            logger.warning("cv2.VideoCapture failed to open")
            return False
        # Apply best-effort settings; drivers may ignore unsupported values
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.height))
        cap.set(cv2.CAP_PROP_FPS, int(self.config.fps))
        try:
            # buffer size hint for v4l2 backends
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.config.buffer_size))
        except Exception:
            pass
        with self._capture_lock:
            self._capture = cap
            self._opened = True
        return True

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout)
        with self._capture_lock:
            if self._capture:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None
        self._opened = False

    def _capture_loop(self) -> None:
        backoff = 0.5
        last_open_attempt = 0.0
        while not self._stop_event.is_set():
            with self._capture_lock:
                cap = self._capture
                opened = self._opened
            if not cap or not opened:
                now = time.time()
                if now - last_open_attempt < backoff:
                    time.sleep(0.05)
                    continue
                last_open_attempt = now
                if not self._open_capture():
                    logger.warning("Camera open failed, retrying")
                    time.sleep(backoff)
                    backoff = min(5.0, backoff * 1.5)
                    continue
                backoff = 0.5
                with self._capture_lock:
                    cap = self._capture
            if cap is None:
                continue
            try:
                # read is usually blocking and efficient in native code
                success, frame = cap.read()
                if not success or frame is None:
                    logger.warning("Camera read failure, attempting reconnect")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    with self._capture_lock:
                        self._capture = None
                        self._opened = False
                    time.sleep(0.1)
                    continue

                # Store frame reference to single-slot buffer; avoid copies
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_ts = time.time()
                    self._latest_frame_id += 1

            except Exception as exc:  # defensive: do not let thread die
                logger.exception("Camera capture loop exception: %s", exc)
                with self._capture_lock:
                    if self._capture:
                        try:
                            self._capture.release()
                        except Exception:
                            pass
                        self._capture = None
                        self._opened = False
                time.sleep(0.5)

        # cleanup when stopping
        with self._capture_lock:
            if self._capture:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None
        logger.info("Camera capture thread stopped")

    def is_available(self) -> bool:
        # quick non-blocking probe: try to open and read a single frame
        probe_cap = cv2.VideoCapture(self.config.device)
        try:
            if not probe_cap.isOpened():
                return False
            # try a quick read with timeout guard
            t0 = time.time()
            while time.time() - t0 < self.config.read_timeout:
                ok, _ = probe_cap.read()
                if ok:
                    return True
            return False
        finally:
            try:
                probe_cap.release()
            except Exception:
                pass

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._opened

    def get_last_frame_age(self) -> Optional[float]:
        with self._frame_lock:
            if self._latest_ts is None:
                return None
            return time.time() - self._latest_ts

    def wait_for_first_frame(self, timeout: float = 5.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self.get_latest(copy=False) is not None:
                return True
            if self._stop_event.is_set():
                return False
            time.sleep(0.05)
        return False

    def get_latest(self, copy: bool = False) -> Optional[Tuple[np.ndarray, float]]:
        """Return (frame, timestamp) of the newest frame or None.

        If `copy` is True a deep copy of the frame is returned. For lowest
        latency and memory use, keep `copy=False` and treat the returned
        numpy array as read-only.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame
            timestamp = float(self._latest_ts)
        if self.config.frame_format == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if copy:
                return (frame.copy(), timestamp)
            return (frame, timestamp)
        if copy:
            return (frame.copy(), timestamp)
        return (frame, timestamp)

    def get_latest_metadata(self, copy: bool = False) -> Optional[tuple[ np.ndarray, float, int]]:
        """Return (frame, timestamp, frame_id) for the newest frame or None."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame
            timestamp = float(self._latest_ts)
            frame_id = self._latest_frame_id
        if self.config.frame_format == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if copy:
                return (frame.copy(), timestamp, frame_id)
            return (frame, timestamp, frame_id)
        if copy:
            return (frame.copy(), timestamp, frame_id)
        return (frame, timestamp, frame_id)

    def test_mode(self, run_seconds: float = 5.0, save_debug_frame: Optional[Path] = None) -> None:
        """Run a lightweight capture test printing FPS and optionally saving a frame.

        This method runs independently (blocking) and is safe to use without
        the model service. It helps validate camera health and basic timing.
        """
        logger.info("Starting camera test mode")
        self.start()
        try:
            start = time.time()
            last_report = start
            report_interval = 1.0
            frames = 0
            frames_since_report = 0
            saved = False
            while time.time() - start < run_seconds:
                res = self.get_latest(copy=True)
                if res is None:
                    time.sleep(0.01)
                    continue
                frame, ts = res
                frames += 1
                frames_since_report += 1
                if save_debug_frame and not saved:
                    try:
                        cv2.imwrite(str(save_debug_frame), frame)
                        saved = True
                        logger.info("Saved debug frame to %s", save_debug_frame)
                    except Exception:
                        logger.exception("Failed to save debug frame")
                now = time.time()
                if now - last_report >= report_interval:
                    fps = frames_since_report / max(1e-6, now - last_report)
                    logger.info("Camera test FPS=%.2f total_frames=%d", fps, frames)
                    last_report = now
                    frames_since_report = 0
                time.sleep(0.001)

            elapsed = max(1e-6, time.time() - start)
            logger.info("Camera test complete: frames=%d elapsed=%.2fs fps=%.2f", frames, elapsed, frames / elapsed)
        finally:
            self.stop()

    def __enter__(self) -> "CameraService":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


if __name__ == "__main__":
    # simple smoke test when run directly
    logging.basicConfig(level=logging.INFO)
    cam = CameraService()
    try:
        cam.test_mode(run_seconds=4.0, save_debug_frame=Path("debug_frame.jpg"))
    finally:
        cam.stop()
