"""Camera management service optimized for Raspberry Pi 3 B+.

Responsibilities:
- Initialize and validate camera configuration
- Provide a low-latency capture loop that always exposes the latest frame
- Lightweight test mode for health checks and optional debug frame saving
- Graceful shutdown, reconnect logic, and safety protections

Design notes:
- A single-frame slot (latest frame) is used to avoid queue buildup and copies.
- A short-grace reconnect loop prevents dead capture streams from stalling.
- V4L2 capture is implemented using raw ioctl calls (fcntl + mmap + numpy only).
  No external packages (v4l2capture, cv2, picamera2) are required.
"""

from __future__ import annotations

import ctypes
import fcntl
import logging
import mmap
import os
import select
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import configs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# V4L2 ioctl constants and structures
# ---------------------------------------------------------------------------

# ioctl request codes
_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2

def _IOC(dir_, type_, nr, size):
    return (dir_ << 30) | (size << 16) | (ord(type_) << 8) | nr

def _IOWR(type_, nr, size):
    return _IOC(_IOC_READ | _IOC_WRITE, type_, nr, size)

def _IOW(type_, nr, size):
    return _IOC(_IOC_WRITE, type_, nr, size)

def _IOR(type_, nr, size):
    return _IOC(_IOC_READ, type_, nr, size)

# fourcc helpers
def _fourcc(a, b, c, d):
    return ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24)

V4L2_PIX_FMT_RGB24 = _fourcc('R', 'G', 'B', '3')
V4L2_PIX_FMT_YUYV  = _fourcc('Y', 'U', 'Y', 'V')

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP             = 1
V4L2_FIELD_ANY               = 0

# struct sizes (ARM 32-bit / armhf)
_FMT_SIZE     = 204   # struct v4l2_format
_REQBUF_SIZE  = 24    # struct v4l2_requestbuffers
_BUF_SIZE     = 88    # struct v4l2_buffer

# ioctl numbers
VIDIOC_S_FMT       = _IOWR('V', 5,  _FMT_SIZE)
VIDIOC_REQBUFS     = _IOWR('V', 8,  _REQBUF_SIZE)
VIDIOC_QUERYBUF    = _IOWR('V', 9,  _BUF_SIZE)
VIDIOC_QBUF        = _IOWR('V', 15, _BUF_SIZE)
VIDIOC_DQBUF       = _IOWR('V', 17, _BUF_SIZE)
VIDIOC_STREAMON    = _IOW ('V', 18, 4)
VIDIOC_STREAMOFF   = _IOW ('V', 19, 4)


def _ioctl(fd, request, buf):
    """Wrapper around fcntl.ioctl that raises OSError on failure."""
    return fcntl.ioctl(fd, request, buf, True)


def _set_format(fd, width, height, pixfmt):
    """Send VIDIOC_S_FMT; returns (actual_width, actual_height, actual_pixfmt)."""
    # struct v4l2_format layout for VIDEO_CAPTURE on ARM 32-bit:
    # u32 type (4), then v4l2_pix_format starting at offset 4:
    #   u32 width, u32 height, u32 pixelformat, u32 field,
    #   u32 bytesperline, u32 sizeimage, v4l2_colorspace colorspace,
    #   u32 priv, u32 flags, ...  pad to 200 bytes
    buf = bytearray(_FMT_SIZE)
    struct.pack_into('IIIII', buf, 0,
                     V4L2_BUF_TYPE_VIDEO_CAPTURE,
                     width, height, pixfmt, V4L2_FIELD_ANY)
    _ioctl(fd, VIDIOC_S_FMT, buf)
    _, aw, ah, apf = struct.unpack_from('IIII', buf, 0)
    return aw, ah, apf


def _request_buffers(fd, count):
    """Send VIDIOC_REQBUFS; returns granted buffer count."""
    buf = bytearray(_REQBUF_SIZE)
    struct.pack_into('III', buf, 0, count, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP)
    _ioctl(fd, VIDIOC_REQBUFS, buf)
    granted, = struct.unpack_from('I', buf, 0)
    return granted


def _query_buffer(fd, index):
    """Send VIDIOC_QUERYBUF; returns (offset, length)."""
    buf = bytearray(_BUF_SIZE)
    struct.pack_into('III', buf, 0, index, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP)
    _ioctl(fd, VIDIOC_QUERYBUF, buf)
    # length at offset 24, m.offset at offset 32 (union, first member)
    length, = struct.unpack_from('I', buf, 24)
    offset, = struct.unpack_from('I', buf, 32)
    return offset, length


def _queue_buffer(fd, index):
    buf = bytearray(_BUF_SIZE)
    struct.pack_into('III', buf, 0, index, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP)
    _ioctl(fd, VIDIOC_QBUF, buf)


def _dequeue_buffer(fd):
    """Send VIDIOC_DQBUF; returns buffer index."""
    buf = bytearray(_BUF_SIZE)
    struct.pack_into('II', buf, 0, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
    _ioctl(fd, VIDIOC_DQBUF, buf)
    index, = struct.unpack_from('I', buf, 0)
    return index


def _stream_on(fd):
    buf = struct.pack('I', V4L2_BUF_TYPE_VIDEO_CAPTURE)
    _ioctl(fd, VIDIOC_STREAMON, bytearray(buf))


def _stream_off(fd):
    buf = struct.pack('I', V4L2_BUF_TYPE_VIDEO_CAPTURE)
    try:
        _ioctl(fd, VIDIOC_STREAMOFF, bytearray(buf))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    device: int = configs.CAMERA_DEVICE
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 2        # number of mmap buffers (min 2 recommended)
    frame_format: str = "rgb"
    read_timeout: float = 3.0

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
        if self.frame_format not in {"rgb"}:
            raise ValueError(f"Camera frame_format must be 'rgb', got {self.frame_format}")
        if not isinstance(self.read_timeout, (int, float)) or self.read_timeout <= 0:
            raise ValueError(f"Camera read_timeout must be positive, got {self.read_timeout}")


# ---------------------------------------------------------------------------
# Raw V4L2 capture (no external packages)
# ---------------------------------------------------------------------------

class V4L2Capture:
    """Low-level V4L2 capture using only fcntl + mmap + numpy."""

    def __init__(self, device: int, width: int, height: int, fps: int, buffer_count: int) -> None:
        self.device_path = f"/dev/video{device}"
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_count = max(2, buffer_count)
        self._fd: Optional[int] = None
        self._mmaps: list[mmap.mmap] = []
        self._pixfmt: Optional[int] = None

    def open(self) -> None:
        self._fd = os.open(self.device_path, os.O_RDWR | os.O_NONBLOCK)
        try:
            self._setup()
        except Exception:
            os.close(self._fd)
            self._fd = None
            raise

    def _setup(self) -> None:
        fd = self._fd

        # Try RGB24 first, fall back to YUYV
        for pixfmt in (V4L2_PIX_FMT_RGB24, V4L2_PIX_FMT_YUYV):
            try:
                aw, ah, apf = _set_format(fd, self.width, self.height, pixfmt)
                self._pixfmt = apf
                self.width = aw
                self.height = ah
                logger.debug("Camera format set: %dx%d pixfmt=0x%x", aw, ah, apf)
                break
            except OSError:
                continue
        else:
            raise RuntimeError("Could not set RGB24 or YUYV format on camera")

        # Request mmap buffers
        granted = _request_buffers(fd, self.buffer_count)
        if granted < 1:
            raise RuntimeError(f"Kernel granted 0 buffers (asked for {self.buffer_count})")

        # Map each buffer into Python
        self._mmaps = []
        for i in range(granted):
            offset, length = _query_buffer(fd, i)
            mm = mmap.mmap(fd, length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)
            self._mmaps.append(mm)
            _queue_buffer(fd, i)

        _stream_on(fd)

    def fileno(self) -> int:
        if self._fd is None:
            raise RuntimeError("V4L2 capture is not open")
        return self._fd

    def read(self) -> np.ndarray:
        if self._fd is None:
            raise RuntimeError("V4L2 capture is not open")

        idx = _dequeue_buffer(self._fd)
        mm = self._mmaps[idx]
        mm.seek(0)

        if self._pixfmt == V4L2_PIX_FMT_RGB24:
            raw = mm.read(self.height * self.width * 3)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        else:
            raw = mm.read(self.height * self.width * 2)
            frame = self._decode_yuyv(raw)

        _queue_buffer(self._fd, idx)
        return frame

    def close(self) -> None:
        if self._fd is None:
            return
        _stream_off(self._fd)
        for mm in self._mmaps:
            try:
                mm.close()
            except Exception:
                pass
        self._mmaps = []
        try:
            os.close(self._fd)
        except Exception:
            pass
        self._fd = None

    # ------------------------------------------------------------------
    # YUYV → RGB conversion (pure numpy, no cv2)
    # ------------------------------------------------------------------

    def _decode_yuyv(self, raw: bytes) -> np.ndarray:
        yuyv = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width // 2, 4))
        y0 = yuyv[:, :, 0].astype(np.int16)
        u  = yuyv[:, :, 1].astype(np.int16) - 128
        y1 = yuyv[:, :, 2].astype(np.int16)
        v  = yuyv[:, :, 3].astype(np.int16) - 128

        rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        rgb[:, 0::2, :] = self._yuv_to_rgb(y0, u, v)
        rgb[:, 1::2, :] = self._yuv_to_rgb(y1, u, v)
        return rgb

    @staticmethod
    def _yuv_to_rgb(y, u, v) -> np.ndarray:
        r = y + (1.3707 * v)
        g = y - (0.3376 * u) - (0.6980 * v)
        b = y + (1.7324 * u)
        return np.clip(np.stack((r, g, b), axis=2), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# CameraService  (unchanged public API)
# ---------------------------------------------------------------------------

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
        self._capture: Optional[V4L2Capture] = None
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
        capture = V4L2Capture(
            device=int(self.config.device),
            width=int(self.config.width),
            height=int(self.config.height),
            fps=int(self.config.fps),
            buffer_count=max(2, int(self.config.buffer_size)),
        )
        try:
            capture.open()
            with self._capture_lock:
                self._capture = capture
                self._opened = True
            return True
        except Exception:
            logger.exception("V4L2 capture failed to open")
            capture.close()
            return False

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
                    self._capture.close()
                except Exception:
                    pass
                self._capture = None
        self._opened = False

    def _capture_loop(self) -> None:
        backoff = 0.5
        last_open_attempt = 0.0
        while not self._stop_event.is_set():
            with self._capture_lock:
                capture = self._capture
                opened = self._opened
            if not capture or not opened:
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
                    capture = self._capture
            if capture is None:
                continue
            try:
                # Wait for frame ready (non-blocking fd)
                ready, _, _ = select.select([capture.fileno()], [], [], self.config.read_timeout)
                if not ready:
                    logger.warning("Camera read timeout, reconnecting")
                    with self._capture_lock:
                        if self._capture:
                            self._capture.close()
                            self._capture = None
                            self._opened = False
                    continue

                frame = capture.read()
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_ts = time.time()
                    self._latest_frame_id += 1

            except Exception as exc:
                logger.exception("Camera capture loop exception: %s", exc)
                with self._capture_lock:
                    if self._capture:
                        try:
                            self._capture.close()
                        except Exception:
                            pass
                        self._capture = None
                        self._opened = False
                time.sleep(0.5)

        with self._capture_lock:
            if self._capture:
                try:
                    self._capture.close()
                except Exception:
                    pass
                self._capture = None
        logger.info("Camera capture thread stopped")

    def is_available(self) -> bool:
        capture = V4L2Capture(
            device=int(self.config.device),
            width=int(self.config.width),
            height=int(self.config.height),
            fps=int(self.config.fps),
            buffer_count=2,
        )
        try:
            capture.open()
            ready, _, _ = select.select([capture.fileno()], [], [], self.config.read_timeout)
            if not ready:
                return False
            frame = capture.read()
            return frame is not None
        except Exception:
            logger.debug("Camera availability check failed", exc_info=True)
            return False
        finally:
            capture.close()

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

        Frames are always in RGB format.
        If `copy` is True a deep copy of the frame is returned.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame
            timestamp = float(self._latest_ts) if self._latest_ts is not None else 0.0
        if copy:
            return (frame.copy(), timestamp)
        return (frame, timestamp)

    def get_latest_metadata(self, copy: bool = False) -> Optional[tuple[np.ndarray, float, int]]:
        """Return (frame, timestamp, frame_id) for the newest frame or None."""
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
        """Run a lightweight capture test printing FPS and optionally saving a frame."""
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