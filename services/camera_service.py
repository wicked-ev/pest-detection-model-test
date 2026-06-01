"""Camera management service optimized for Raspberry Pi 3 B+.

Responsibilities:
- Initialize and validate camera configuration
- Provide a low-latency capture loop that always exposes the latest frame
- Lightweight test mode for health checks and optional debug frame saving
- Graceful shutdown, reconnect logic, and safety protections

Design notes:
- Uses pluggable camera backend architecture (OpenCV, V4L2, etc.)
- Backends are tried in preference order until one succeeds
- Backend selection is cached to avoid startup delays
- A single-frame slot (latest frame) is used to avoid queue buildup and copies.
- A short-grace reconnect loop prevents dead capture streams from stalling.
"""

from __future__ import annotations

import logging
import select
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import configs
from .camera_backends import (
    BaseCameraBackend,
    CameraBackendError,
    CameraBackendFactory,
    CameraBackendConfig,
)

logger = logging.getLogger(__name__)


# Legacy camera configuration (kept for API compatibility)
@dataclass
class CameraConfig:
    """Camera service configuration (backward compatible wrapper).
    
    This class is maintained for API compatibility with existing code.
    Internally, it uses the CameraBackendConfig from the new architecture.
    """
    device: int = configs.CAMERA_DEVICE
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 2
    frame_format: str = "rgb"
    read_timeout: float = 3.0
    backend_preference: Optional[Tuple[str, ...]] = None
    cache_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        """Validate camera configuration."""
        if not isinstance(self.device, int) or self.device < 0:
            raise ValueError(f"Camera device must be non-negative, got {self.device}")
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError(f"Camera width must be positive, got {self.width}")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError(f"Camera height must be positive, got {self.height}")
        if not isinstance(self.fps, int) or self.fps <= 0 or self.fps > 60:
            raise ValueError(f"Camera fps must be between 1-60, got {self.fps}")
        if not isinstance(self.buffer_size, int) or self.buffer_size <= 0 or self.buffer_size > 5:
            raise ValueError(f"Camera buffer_size must be between 1-5, got {self.buffer_size}")
        if self.frame_format not in {"rgb"}:
            raise ValueError(f"Camera frame_format must be 'rgb', got {self.frame_format}")
        if not isinstance(self.read_timeout, (int, float)) or self.read_timeout <= 0:
            raise ValueError(f"Camera read_timeout must be positive, got {self.read_timeout}")

    def to_backend_config(self) -> CameraBackendConfig:
        """Convert to backend configuration object."""
        preference = self.backend_preference or ("opencv", "v4l2")
        return CameraBackendConfig(
            device=self.device,
            width=self.width,
            height=self.height,
            fps=self.fps,
            buffer_count=self.buffer_size,
            backend_preference=preference,
        )


class CameraService:
    """Camera service providing a low-latency newest-frame buffer using pluggable backends.
    
    The service uses backend abstraction to support multiple camera implementations:
    - OpenCV (preferred when available)
    - V4L2 (Linux-native)
    
    Backends are tried in preference order until one succeeds and is cached for fast
    subsequent startup.

    Usage:
    - create instance with optional CameraConfig
    - call `start()` to begin capture thread
    - use `get_latest(copy=False)` to obtain newest frame reference
    - call `stop()` to shutdown and release resources
    """

    def __init__(self, config: Optional[CameraConfig] = None, cache_dir: Optional[Path] = None):
        """Initialize camera service.
        
        Args:
            config: Camera configuration (uses defaults if None)
            cache_dir: Directory for caching backend selection
        """
        self.config = config or CameraConfig()
        self._cache_dir = Path(cache_dir) if cache_dir is not None else Path(configs.OUTPUT_DIR)
        self._backend_cache_path = self._cache_dir / ".camera_backend_cache.json"
        
        self._backend: Optional[BaseCameraBackend] = None
        self._backend_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: Optional[float] = None
        self._latest_frame_id: int = 0
        self._opened = False

    def _open_backend(self) -> bool:
        """Select and open a camera backend with fallback strategy."""
        logger.info("Opening camera backend (device=%d, %dx%d)",
                   self.config.device, self.config.width, self.config.height)
        
        backend_config = self.config.to_backend_config()
        backend = CameraBackendFactory.select_best_backend(
            backend_config,
            cache_path=self._backend_cache_path,
        )
        
        if backend is None:
            logger.error("No camera backend could be selected")
            return False
        
        try:
            with self._backend_lock:
                self._backend = backend
                self._opened = True
            logger.info("Camera backend opened: %s", backend.name)
            return True
        except Exception as e:
            logger.exception("Backend opening failed: %s", e)
            if backend:
                backend.close()
            return False

    def start(self) -> None:
        """Start the camera capture thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the camera capture thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout)
        with self._backend_lock:
            if self._backend:
                try:
                    self._backend.close()
                except Exception:
                    pass
                self._backend = None
        self._opened = False

    def _capture_loop(self) -> None:
        """Main capture loop running in background thread."""
        backoff = 0.5
        last_open_attempt = 0.0
        
        while not self._stop_event.is_set():
            with self._backend_lock:
                backend = self._backend
                opened = self._opened
            
            # Check if we need to open backend
            if not backend or not opened:
                now = time.time()
                if now - last_open_attempt < backoff:
                    time.sleep(0.05)
                    continue
                
                last_open_attempt = now
                if not self._open_backend():
                    logger.warning("Camera backend open failed, retrying in %.1fs", backoff)
                    time.sleep(backoff)
                    backoff = min(5.0, backoff * 1.5)
                    continue
                
                backoff = 0.5
                with self._backend_lock:
                    backend = self._backend
            
            if backend is None:
                continue
            
            try:
                # Try to read frame with timeout support
                try:
                    fd = backend.fileno()
                    ready, _, _ = select.select([fd], [], [], self.config.read_timeout)
                    if not ready:
                        logger.warning("Camera read timeout, reconnecting backend")
                        with self._backend_lock:
                            if self._backend:
                                self._backend.close()
                                self._backend = None
                                self._opened = False
                        continue
                except CameraBackendError:
                    # Backend doesn't support fileno (e.g., OpenCV)
                    # Just try to read directly
                    pass
                
                frame = backend.read()
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_ts = time.time()
                    self._latest_frame_id += 1
                    
            except CameraBackendError as e:
                logger.warning("Camera backend error: %s, reconnecting", e)
                with self._backend_lock:
                    if self._backend:
                        try:
                            self._backend.close()
                        except Exception:
                            pass
                        self._backend = None
                        self._opened = False
                time.sleep(0.5)
            except Exception as exc:
                logger.exception("Camera capture loop exception: %s", exc)
                with self._backend_lock:
                    if self._backend:
                        try:
                            self._backend.close()
                        except Exception:
                            pass
                        self._backend = None
                        self._opened = False
                time.sleep(0.5)
        
        # Cleanup on exit
        with self._backend_lock:
            if self._backend:
                try:
                    self._backend.close()
                except Exception:
                    pass
                self._backend = None
        logger.info("Camera capture thread stopped")

    def is_available(self) -> bool:
        """Check if any camera backend is available."""
        backend_config = self.config.to_backend_config()
        backends = CameraBackendFactory.create_backends(backend_config)
        return len(backends) > 0

    def is_running(self) -> bool:
        """Check if capture thread is running."""
        return self._thread is not None and self._thread.is_alive() and self._opened

    def get_last_frame_age(self) -> Optional[float]:
        """Get age of last captured frame in seconds."""
        with self._frame_lock:
            if self._latest_ts is None:
                return None
            return time.time() - self._latest_ts

    def wait_for_first_frame(self, timeout: float = 5.0) -> bool:
        """Wait for first frame to be captured (up to timeout seconds)."""
        start = time.time()
        while time.time() - start < timeout:
            if self.get_latest(copy=False) is not None:
                return True
            if self._stop_event.is_set():
                return False
            time.sleep(0.05)
        return False

    def get_latest(self, copy: bool = False) -> Optional[Tuple[np.ndarray, float]]:
        """Get the latest captured frame and timestamp.
        
        Args:
            copy: If True, return a deep copy of the frame
            
        Returns:
            Tuple of (frame, timestamp) or None if no frame available.
            Frame is always RGB uint8 format.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame
            timestamp = float(self._latest_ts) if self._latest_ts is not None else 0.0
        if copy:
            return (frame.copy(), timestamp)
        return (frame, timestamp)

    def get_latest_metadata(self, copy: bool = False) -> Optional[Tuple[np.ndarray, float, int]]:
        """Get the latest frame with frame ID.
        
        Args:
            copy: If True, return a deep copy of the frame
            
        Returns:
            Tuple of (frame, timestamp, frame_id) or None if no frame available.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame
            timestamp = float(self._latest_ts) if self._latest_ts is not None else 0.0
            frame_id = self._latest_frame_id
        if copy:
            return (frame.copy(), timestamp, frame_id)
        return (frame, timestamp, frame_id)

    def test_mode(self, run_seconds: float = 5.0, save_debug_frame: Optional[Path] = None) -> None:
        """Run a lightweight capture test printing FPS stats.
        
        Args:
            run_seconds: How long to run the test
            save_debug_frame: Optional path to save a debug frame image
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
                        try:
                            from PIL import Image
                            img = Image.fromarray(frame.astype(np.uint8), mode='RGB')
                            img.save(str(save_debug_frame))
                            saved = True
                            logger.info("Saved debug frame to %s (via PIL)", save_debug_frame)
                        except ImportError:
                            np.save(str(save_debug_frame), frame)
                            saved = True
                            logger.info("Saved debug frame to %s (numpy format)", save_debug_frame)
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
            logger.info("Camera test complete: frames=%d elapsed=%.2fs fps=%.2f",
                        frames, elapsed, frames / elapsed)
        finally:
            self.stop()

    def __enter__(self) -> "CameraService":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cam = CameraService()
    try:
        cam.test_mode(run_seconds=4.0, save_debug_frame=Path("debug_frame.jpg"))
    finally:
        cam.stop()