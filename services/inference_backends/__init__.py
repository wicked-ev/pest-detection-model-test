"""Inference backend abstraction and factory.

This module provides a pluggable inference backend architecture supporting:
- TensorFlow Lite (preferred for Raspberry Pi)
- ONNX Runtime (alternative local inference)
- Remote Inference (final fallback for edge cases)

Backends are tried in preference order until one loads successfully.
Once a backend is selected, it is cached to avoid repeated failures.
"""

from .base import BaseInferenceBackend, InferenceBackendError, Detection
from .factory import InferenceBackendFactory, InferenceBackendConfig

__all__ = [
    "BaseInferenceBackend",
    "InferenceBackendError",
    "Detection",
    "InferenceBackendFactory",
    "InferenceBackendConfig",
]
