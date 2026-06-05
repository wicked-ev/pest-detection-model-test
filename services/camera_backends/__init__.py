"""Camera backend abstraction layer.

This module provides a pluggable camera backend architecture supporting:
- Picamera2 (preferred for Raspberry Pi 4+, Pi 5)
- OpenCV (cross-platform fallback)
- V4L2 (Linux-native, minimal dependencies)

Backends are tried in preference order until one loads successfully.
Once a backend is selected, it is cached to avoid repeated failures.
"""

from .base import BaseCameraBackend, CameraBackendError
from .factory import CameraBackendFactory, CameraBackendConfig

__all__ = [
    "BaseCameraBackend",
    "CameraBackendError",
    "CameraBackendFactory",
    "CameraBackendConfig",
]
