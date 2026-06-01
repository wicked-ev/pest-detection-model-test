"""Base camera backend abstraction."""

from __future__ import annotations

import abc
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CameraBackendError(RuntimeError):
    """Exception raised by camera backends."""
    pass


class BaseCameraBackend(abc.ABC):
    """Abstract base for camera backends.
    
    All backends must:
    1. Return frames in RGB format (H x W x 3)
    2. Be thread-safe for single-threaded access patterns
    3. Support graceful open/close lifecycle
    4. Provide a simple read() interface
    """

    name = "base"

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this backend's dependencies are installed and the hardware is accessible.
        
        Should return quickly without opening resources.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def open(self) -> None:
        """Open and initialize the camera resource.
        
        Raises CameraBackendError if initialization fails.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def read(self) -> np.ndarray:
        """Read and return the next frame.
        
        Returns:
            Frame as numpy array with shape (H, W, 3) and dtype uint8 in RGB format.
            
        Raises CameraBackendError if read fails.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Close and release the camera resource.
        
        Should be safe to call multiple times and when open() was never called.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fileno(self) -> int:
        """Return OS file descriptor for select/poll operations.
        
        Returns -1 if not applicable to this backend.
        Raises CameraBackendError if not supported.
        """
        raise NotImplementedError
