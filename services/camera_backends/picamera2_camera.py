"""Picamera2 camera backend for Raspberry Pi.

Dependencies:
- picamera2 (Python 3.11+, Raspberry Pi OS with libcamera)

Advantages:
- Official Raspberry Pi camera library
- Better performance on Raspberry Pi 4+ and Pi 5
- Full hardware acceleration support
- Supports both legacy and new camera stack

Disadvantages:
- Raspberry Pi specific (requires libcamera)
- May not work on older Pi models or older OS versions
"""

from __future__ import annotations

import logging
import configs
from typing import Optional

import numpy as np

from .base import BaseCameraBackend, CameraBackendError

logger = logging.getLogger(__name__)


class Picamera2Backend(BaseCameraBackend):
    """Picamera2 camera backend for Raspberry Pi using libcamera.
    
    Uses the new Picamera2 library which is the official camera library
    for Raspberry Pi with libcamera support.
    
    Dependencies:
    - picamera2
    - numpy
    """

    name = "picamera2"

    def __init__(self, device: int = 0, width: int = 640, height: int = 480,
                 fps: int = 30, buffer_count: int = 2):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_count = buffer_count
        self._camera: Optional[object] = None  # Picamera2 object
        self._stream: Optional[object] = None  # Output stream

    def is_available(self) -> bool:
        """Check if picamera2 is installed and available."""
        try:
            import picamera2
            logger.debug("Picamera2 library is available")
            return True
        except ImportError:
            logger.debug("Picamera2 library not installed")
            return False
        except Exception as e:
            logger.debug("Picamera2 availability check failed: %s", e)
            return False

    def open(self) -> None:
        """Open camera using Picamera2."""
        try:
            from picamera2 import Picamera2
            import libcamera
            logger.debug("Picamera2 and libcamera libraries imported successfully")
        except ImportError as e:
            logger.error("Failed to import Picamera2 or libcamera: %s", e)
            raise CameraBackendError(f"Picamera2 or libcamera not installed: {e}")

        try:
            logger.debug("Creating Picamera2 instance for camera device %d", self.device)
            # Create camera instance
            self._camera = Picamera2(camera_num=self.device)
            logger.debug("Picamera2 instance created")
            
            # Create camera configuration
            logger.debug("Creating preview configuration: %dx%d RGB888 format", 
                       self.width, self.height)
            config = self._camera.create_preview_configuration(
                main={"format": "RGB888", "size": (self.width, self.height)},
                raw=None,
            )
            logger.debug("Configuration created")
            
            # Set framerate
            logger.debug("Setting framerate to %d fps", self.fps)
            config["controls"] = {"FrameRate": self.fps}
            
            # Apply configuration
            logger.debug("Applying camera configuration")
            self._camera.configure(config)
            logger.debug("Configuration applied")
            
            # Start camera
            logger.debug("Starting camera capture")
            self._camera.start()
            logger.debug("Camera started")
            
            logger.info("Picamera2 camera opened successfully: device=%d %dx%d fps=%d",
                       self.device, self.width, self.height, self.fps)
            
        except Exception as e:
            logger.error("Picamera2 camera open failed: %s", e, exc_info=True)
            if self._camera is not None:
                try:
                    self._camera.stop()
                except Exception as cleanup_error:
                    logger.debug("Error while stopping camera during cleanup: %s", cleanup_error)
                try:
                    self._camera.close()
                except Exception as cleanup_error:
                    logger.debug("Error while closing camera during cleanup: %s", cleanup_error)
            self._camera = None
            raise CameraBackendError(f"Picamera2 camera open failed: {e}")

    def read(self) -> np.ndarray:
        """Read next frame from camera using Picamera2."""
        if self._camera is None:
            raise CameraBackendError("Picamera2 camera not opened")

        try:
            # Capture frame
            if configs.STREAMING_LOGS_ENABLED:
                logger.debug("Capturing array from Picamera2")
            array = self._camera.capture_array()
            if configs.STREAMING_LOGS_ENABLED:
                logger.debug("Array captured, type=%s shape=%s", type(array), 
                           getattr(array, 'shape', 'N/A'))
            
            if array is None:
                logger.error("Picamera2 returned None array")
                raise CameraBackendError("Failed to capture frame from Picamera2 (None returned)")
            
            # Ensure frame is in RGB format and correct shape
            if len(array.shape) != 3 or array.shape[2] != 3:
                logger.error("Unexpected frame shape from Picamera2: %s (expected (H, W, 3))", 
                           array.shape)
                raise CameraBackendError(
                    f"Unexpected frame shape from Picamera2: {array.shape}"
                )
            
            if configs.STREAMING_LOGS_ENABLED:
                logger.debug("Frame validated: shape=%s dtype=%s", array.shape, array.dtype)
            return array
            
        except CameraBackendError:
            raise
        except Exception as e:
            logger.error("Failed to read frame from Picamera2: %s", e, exc_info=True)
            raise CameraBackendError(f"Failed to read frame from Picamera2: {e}")

    def close(self) -> None:
        """Close Picamera2 camera."""
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            try:
                self._camera.close()
            except Exception:
                pass
            self._camera = None
        logger.debug("Picamera2 camera closed")

    def fileno(self) -> int:
        """Not supported for Picamera2 backend."""
        raise CameraBackendError("fileno() not supported for Picamera2 backend")
