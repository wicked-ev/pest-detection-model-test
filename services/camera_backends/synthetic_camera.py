"""Synthetic camera backend: draws moving test frames, for running without a camera.

Select it with CAMERA_BACKENDS=synthetic (e.g. to test the robot software on a PC).
It is never chosen automatically.
"""
from __future__ import annotations

import time

import numpy as np

from .base import BaseCameraBackend, CameraBackendError


class SyntheticCameraBackend(BaseCameraBackend):
    """Green background with a yellow square moving left to right, at the configured fps."""

    name = "synthetic"

    def __init__(self, device: int = 0, width: int = 640, height: int = 480, fps: int = 30, buffer_count: int = 2):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._background: np.ndarray | None = None
        self._frame_index = 0
        self._next_frame_time = 0.0

    def is_available(self) -> bool:
        return True

    def open(self) -> None:
        self._background = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._background[:] = (40, 90, 40)
        self._frame_index = 0
        self._next_frame_time = time.monotonic()

    def read(self) -> np.ndarray:
        if self._background is None:
            raise CameraBackendError("Synthetic camera not opened")
        # Pace frames like a real camera, so the capture loop does not spin.
        delay = self._next_frame_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_frame_time = max(self._next_frame_time + 1.0 / self.fps, time.monotonic())

        frame = self._background.copy()
        size = max(16, self.height // 6)
        x = (self._frame_index * 8) % max(1, self.width - size)
        y = (self.height - size) // 2
        frame[y:y + size, x:x + size] = (210, 190, 60)
        self._frame_index += 1
        return frame

    def close(self) -> None:
        self._background = None

    def fileno(self) -> int:
        raise CameraBackendError("fileno() not supported for synthetic backend")
