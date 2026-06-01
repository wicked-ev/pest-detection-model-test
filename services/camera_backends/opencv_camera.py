"""OpenCV camera backend."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base import BaseCameraBackend, CameraBackendError

logger = logging.getLogger(__name__)


class OpenCVCameraBackend(BaseCameraBackend):
    """OpenCV camera backend using cv2.VideoCapture.
    
    Preferred backend when available (works across all platforms).
    
    Dependencies:
    - opencv-python or opencv-contrib-python
    
    Advantages:
    - Works on Linux, macOS, Windows
    - Handles various camera formats and codecs
    - More robust on some hardware
    
    Disadvantages:
    - May not work in certain virtual environments
    - Larger footprint than V4L2
    """

    name = "opencv"

    def __init__(self, device: int = 0, width: int = 640, height: int = 480,
                 fps: int = 30, buffer_count: int = 1):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_count = buffer_count
        self._cap: Optional[object] = None  # cv2.VideoCapture object

    def is_available(self) -> bool:
        """Check if OpenCV is installed."""
        try:
            import cv2
            # Try to create and immediately close a capture to verify
            cap = cv2.VideoCapture(self.device)
            if cap is not None and cap.isOpened():
                cap.release()
                return True
            return False
        except ImportError:
            logger.debug("OpenCV (cv2) not installed")
            return False
        except Exception as e:
            logger.debug("OpenCV availability check failed: %s", e)
            return False

    def open(self) -> None:
        """Open camera using OpenCV."""
        try:
            import cv2
        except ImportError:
            raise CameraBackendError("OpenCV (cv2) not installed")

        try:
            self._cap = cv2.VideoCapture(self.device)
            if self._cap is None or not self._cap.isOpened():
                raise CameraBackendError(f"Failed to open camera device {self.device}")

            # Set resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Set FPS
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Set buffer size (reduce buffering lag)
            # Not all backends support this, so ignore errors
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_count)
            except Exception:
                pass

            # Try to read a frame to verify the camera works
            ret, frame = self._cap.read()
            if not ret:
                self._cap.release()
                self._cap = None
                raise CameraBackendError(f"Failed to read frame from camera device {self.device}")

            # Verify or read the actual resolution
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info("OpenCV camera opened: device=%d %dx%d",
                       self.device, actual_width, actual_height)

        except Exception as e:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = None
            raise CameraBackendError(f"OpenCV camera open failed: {e}")

    def fileno(self) -> int:
        """Not supported for OpenCV backend."""
        raise CameraBackendError("fileno() not supported for OpenCV backend")

    def read(self) -> np.ndarray:
        """Read next frame from camera using OpenCV."""
        if self._cap is None:
            raise CameraBackendError("OpenCV camera not opened")

        ret, frame = self._cap.read()
        if not ret:
            raise CameraBackendError("Failed to read frame from OpenCV camera")

        # Convert BGR to RGB (OpenCV uses BGR by default)
        try:
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise CameraBackendError(f"Failed to convert frame format: {e}")

        return rgb_frame

    def close(self) -> None:
        """Close OpenCV camera."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        logger.debug("OpenCV camera closed")
