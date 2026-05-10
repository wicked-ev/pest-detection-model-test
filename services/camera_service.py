"""
Camera management service for Raspberry Pi camera hardware.

Provides a simple probe mechanism to verify that the Pi camera is
available and responsive before entering READY state.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import configs

logger = logging.getLogger(__name__)


class CameraService:
    """Simple camera probe service."""

    def __init__(self, device: int = configs.CAMERA_DEVICE, timeout_seconds: float = 3.0):
        self.device = device
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        """Return True if the configured camera can be opened."""
        logger.info(f"Checking camera availability on device {self.device}")
        capture = cv2.VideoCapture(self.device)
        try:
            if not capture.isOpened():
                logger.warning("Camera device failed to open")
                return False
            # Attempt a single read to confirm capture
            success, _ = capture.read()
            if not success:
                logger.warning("Camera opened but failed to read frame")
                return False
            logger.info("Camera probe succeeded")
            return True
        except Exception as exc:
            logger.error(f"Camera probe exception: {exc}")
            return False
        finally:
            capture.release()
