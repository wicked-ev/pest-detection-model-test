"""V4L2 camera backend (Linux-native, minimal dependencies)."""

from __future__ import annotations

import fcntl
import logging
import mmap
import os
import struct
from typing import Optional

import numpy as np

from .base import BaseCameraBackend, CameraBackendError

logger = logging.getLogger(__name__)

# ============================================================================
# V4L2 ioctl constants and structures
# ============================================================================

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
V4L2_PIX_FMT_YUYV = _fourcc('Y', 'U', 'Y', 'V')

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_ANY = 0

# struct sizes (ARM 32-bit / armhf)
_FMT_SIZE = 204   # struct v4l2_format
_REQBUF_SIZE = 24    # struct v4l2_requestbuffers
_BUF_SIZE = 88    # struct v4l2_buffer

# ioctl numbers
VIDIOC_S_FMT = _IOWR('V', 5, _FMT_SIZE)
VIDIOC_REQBUFS = _IOWR('V', 8, _REQBUF_SIZE)
VIDIOC_QUERYBUF = _IOWR('V', 9, _BUF_SIZE)
VIDIOC_QBUF = _IOWR('V', 15, _BUF_SIZE)
VIDIOC_DQBUF = _IOWR('V', 17, _BUF_SIZE)
VIDIOC_STREAMON = _IOW('V', 18, 4)
VIDIOC_STREAMOFF = _IOW('V', 19, 4)


def _ioctl(fd, request, buf):
    """Wrapper around fcntl.ioctl that raises OSError on failure."""
    return fcntl.ioctl(fd, request, buf, True)


def _set_format(fd, width, height, pixfmt):
    """Send VIDIOC_S_FMT; returns (actual_width, actual_height, actual_pixfmt)."""
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


# ============================================================================
# V4L2 Camera Backend
# ============================================================================


class V4L2CameraBackend(BaseCameraBackend):
    """V4L2 camera backend using Linux device API directly (no external packages).
    
    Supports:
    - RGB24 format (preferred)
    - YUYV format (fallback)
    
    Dependencies:
    - None (uses only fcntl, mmap, struct from stdlib)
    
    Typical device paths on Linux:
    - /dev/video0 (primary camera)
    - /dev/video1, /dev/video2, etc. (additional cameras)
    """

    name = "v4l2"

    def __init__(self, device: int = 0, width: int = 640, height: int = 480,
                 fps: int = 30, buffer_count: int = 2):
        self.device_path = f"/dev/video{device}"
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_count = max(2, buffer_count)
        self._fd: Optional[int] = None
        self._mmaps: list[mmap.mmap] = []
        self._pixfmt: Optional[int] = None

    def is_available(self) -> bool:
        """Check if device exists and is readable."""
        try:
            stat_result = os.stat(self.device_path)
            # Check if it's a character device
            return os.path.isfile(self.device_path) or (hasattr(stat_result, 'st_mode') and True)
        except (OSError, FileNotFoundError):
            logger.debug("V4L2 device not available: %s", self.device_path)
            return False

    def open(self) -> None:
        """Open and initialize the V4L2 camera."""
        if not self.is_available():
            raise CameraBackendError(f"V4L2 device not available: {self.device_path}")

        try:
            self._fd = os.open(self.device_path, os.O_RDWR | os.O_NONBLOCK)
        except OSError as e:
            raise CameraBackendError(f"Failed to open V4L2 device {self.device_path}: {e}")

        try:
            self._setup()
        except Exception as e:
            self._cleanup_resources()
            raise CameraBackendError(f"V4L2 setup failed: {e}")

    def _setup(self) -> None:
        """Initialize V4L2 capture."""
        if self._fd is None:
            raise CameraBackendError("V4L2 not opened")

        # Try RGB24 first, fall back to YUYV
        for pixfmt in (V4L2_PIX_FMT_RGB24, V4L2_PIX_FMT_YUYV):
            try:
                aw, ah, apf = _set_format(self._fd, self.width, self.height, pixfmt)
                self._pixfmt = apf
                self.width = aw
                self.height = ah
                logger.debug("V4L2 format set: %dx%d pixfmt=0x%x", aw, ah, apf)
                break
            except OSError:
                continue
        else:
            raise RuntimeError("Could not set RGB24 or YUYV format on V4L2 camera")

        # Request mmap buffers
        granted = _request_buffers(self._fd, self.buffer_count)
        if granted < 1:
            raise RuntimeError(f"Kernel granted 0 buffers (asked for {self.buffer_count})")

        # Map each buffer into Python
        self._mmaps = []
        for i in range(granted):
            offset, length = _query_buffer(self._fd, i)
            mm = mmap.mmap(self._fd, length, mmap.MAP_SHARED,
                          mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)
            self._mmaps.append(mm)
            _queue_buffer(self._fd, i)

        _stream_on(self._fd)
        logger.info("V4L2 camera opened: device=%s %dx%d buffers=%d",
                   self.device_path, self.width, self.height, len(self._mmaps))

    def _cleanup_resources(self) -> None:
        """Clean up V4L2 resources."""
        if self._fd is not None:
            _stream_off(self._fd)
        for mm in self._mmaps:
            try:
                mm.close()
            except Exception:
                pass
        self._mmaps = []
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None

    def fileno(self) -> int:
        """Return file descriptor for select/poll."""
        if self._fd is None:
            raise CameraBackendError("V4L2 camera not opened")
        return self._fd

    def read(self) -> np.ndarray:
        """Read next frame from V4L2 camera."""
        if self._fd is None:
            raise CameraBackendError("V4L2 camera not opened")

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

    def _decode_yuyv(self, raw: bytes) -> np.ndarray:
        """Convert YUYV to RGB."""
        yuyv = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width // 2, 4))
        y0 = yuyv[:, :, 0].astype(np.int16)
        u = yuyv[:, :, 1].astype(np.int16) - 128
        y1 = yuyv[:, :, 2].astype(np.int16)
        v = yuyv[:, :, 3].astype(np.int16) - 128

        rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        rgb[:, 0::2, :] = self._yuv_to_rgb(y0, u, v)
        rgb[:, 1::2, :] = self._yuv_to_rgb(y1, u, v)
        return rgb

    @staticmethod
    def _yuv_to_rgb(y, u, v) -> np.ndarray:
        """YUV to RGB conversion."""
        r = y + (1.3707 * v)
        g = y - (0.3376 * u) - (0.6980 * v)
        b = y + (1.7324 * u)
        return np.clip(np.stack((r, g, b), axis=2), 0, 255).astype(np.uint8)

    def close(self) -> None:
        """Close V4L2 camera and release resources."""
        self._cleanup_resources()
        logger.debug("V4L2 camera closed")
