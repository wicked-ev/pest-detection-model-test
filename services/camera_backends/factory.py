"""Camera backend factory with fallback strategy."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .base import BaseCameraBackend, CameraBackendError
from .opencv_camera import OpenCVCameraBackend
from .v4l2_camera import V4L2CameraBackend

logger = logging.getLogger(__name__)


@dataclass
class CameraBackendConfig:
    """Configuration for camera backends."""
    device: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_count: int = 2
    backend_preference: Optional[Sequence[str]] = None


class CameraBackendFactory:
    """Factory for creating and managing camera backends with fallback strategy."""

    BACKENDS = {
        "opencv": OpenCVCameraBackend,
        "v4l2": V4L2CameraBackend,
    }

    @classmethod
    def create_backend(cls, name: str, config: CameraBackendConfig) -> Optional[BaseCameraBackend]:
        """Create a single camera backend by name.
        
        Returns None if backend is unknown or unavailable.
        """
        backend_cls = cls.BACKENDS.get(name.lower())
        if backend_cls is None:
            logger.warning("Unknown camera backend: %s", name)
            return None

        try:
            backend = backend_cls(
                device=config.device,
                width=config.width,
                height=config.height,
                fps=config.fps,
                buffer_count=config.buffer_count,
            )
            if not backend.is_available():
                logger.debug("Camera backend '%s' is not available", name)
                return None
            return backend
        except Exception as e:
            logger.debug("Failed to create camera backend '%s': %s", name, e)
            return None

    @classmethod
    def create_backends(cls, config: CameraBackendConfig) -> List[BaseCameraBackend]:
        """Create a list of available camera backends in preference order.
        
        Skips unavailable backends and returns only those that can be instantiated.
        """
        preference = config.backend_preference or ("opencv", "v4l2")
        backends: List[BaseCameraBackend] = []

        for name in preference:
            backend = cls.create_backend(name.strip().lower(), config)
            if backend is not None:
                backends.append(backend)
                logger.debug("Camera backend '%s' available", backend.name)

        if not backends:
            logger.warning("No camera backends available from preference: %s", preference)

        return backends

    @classmethod
    def select_best_backend(cls, config: CameraBackendConfig,
                           cache_path: Optional[Path] = None) -> Optional[BaseCameraBackend]:
        """Select the best available camera backend with caching support.
        
        Selection strategy:
        1. Check cache for previously successful backend
        2. If cached backend works, use it
        3. Otherwise try each backend in preference order
        4. Return first one that loads successfully
        5. Cache the successful selection
        
        Args:
            config: Camera configuration
            cache_path: Optional path to cache backend selection
            
        Returns:
            Selected backend instance, or None if no backends available
        """
        # Try cached backend first
        if cache_path and cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                    cached_name = data.get("camera_backend")
                    if cached_name:
                        logger.debug("Trying cached camera backend: %s", cached_name)
                        backend = cls.create_backend(cached_name, config)
                        if backend is not None:
                            try:
                                backend.open()
                                logger.info("Using cached camera backend: %s", backend.name)
                                return backend
                            except Exception as e:
                                logger.warning("Cached backend failed, trying others: %s", e)
                                backend.close()
            except Exception as e:
                logger.debug("Failed to load cached backend: %s", e)

        # Try each available backend
        backends = cls.create_backends(config)
        for backend in backends:
            try:
                logger.debug("Attempting camera backend: %s", backend.name)
                backend.open()
                logger.info("Successfully selected camera backend: %s", backend.name)

                # Cache the selection
                if cache_path:
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_path, 'w') as f:
                            json.dump({"camera_backend": backend.name}, f)
                        logger.debug("Cached camera backend selection: %s", backend.name)
                    except Exception as e:
                        logger.debug("Failed to cache backend selection: %s", e)

                return backend
            except Exception as e:
                logger.warning("Camera backend '%s' failed: %s", backend.name, e)
                try:
                    backend.close()
                except Exception:
                    pass
                continue

        logger.error("No camera backend could be selected from: %s", 
                    config.backend_preference or ("opencv", "v4l2"))
        return None
