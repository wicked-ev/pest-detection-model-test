"""Inference backend factory with fallback strategy."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .base import BaseInferenceBackend, InferenceBackendError, Detection
from .tflite import TFLiteBackend
from .onnx import ONNXBackend

logger = logging.getLogger(__name__)


@dataclass
class InferenceBackendConfig:
    """Configuration for inference backends."""
    model_path: Path | str
    onnx_model_path: Path | str
    annotations_path: Path | str
    confidence_threshold: float = 0.5
    num_threads: int = 1
    backend_preference: Optional[Sequence[str]] = None


class InferenceBackendFactory:
    """Factory for creating and managing inference backends with fallback strategy."""

    BACKENDS = {
        "tflite": TFLiteBackend,
        "onnx": ONNXBackend,
    }

    @classmethod
    def create_backend(cls, name: str, config: InferenceBackendConfig) -> Optional[BaseInferenceBackend]:
        """Create a single inference backend by name.
        
        Returns None if backend is unknown or unavailable.
        """
        backend_cls = cls.BACKENDS.get(name.lower())
        if backend_cls is None:
            logger.warning("Unknown inference backend: %s", name)
            return None

        try:
            # Use appropriate model path for backend
            if name.lower() == "onnx":
                model_path = config.onnx_model_path
            else:
                model_path = config.model_path

            backend = backend_cls(
                model_path=model_path,
                annotations_path=config.annotations_path,
                confidence_threshold=config.confidence_threshold,
                num_threads=config.num_threads,
            )

            if not backend.is_available():
                logger.debug("Inference backend '%s' is not available", name)
                return None

            return backend
        except Exception as e:
            logger.debug("Failed to create inference backend '%s': %s", name, e)
            return None

    @classmethod
    def create_backends(cls, config: InferenceBackendConfig) -> List[BaseInferenceBackend]:
        """Create a list of available inference backends in preference order.
        
        Skips unavailable backends and returns only those that can be instantiated.
        """
        preference = config.backend_preference or ("tflite", "onnx")
        backends: List[BaseInferenceBackend] = []

        for name in preference:
            backend = cls.create_backend(name.strip().lower(), config)
            if backend is not None:
                backends.append(backend)
                logger.debug("Inference backend '%s' available", backend.name)

        if not backends:
            logger.warning("No inference backends available from preference: %s", preference)

        return backends

    @classmethod
    def select_best_backend(cls, config: InferenceBackendConfig,
                           cache_path: Optional[Path] = None) -> Optional[BaseInferenceBackend]:
        """Select the best available inference backend with caching support.
        
        Selection strategy:
        1. Check cache for previously successful backend
        2. If cached backend works, use it
        3. Otherwise try each backend in preference order
        4. Return first one that loads successfully
        5. Cache the successful selection
        
        Args:
            config: Inference backend configuration
            cache_path: Optional path to cache backend selection
            
        Returns:
            Selected backend instance, or None if no backends available
        """
        # Try cached backend first
        if cache_path and cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                    cached_name = data.get("inference_backend")
                    if cached_name:
                        logger.debug("Trying cached inference backend: %s", cached_name)
                        backend = cls.create_backend(cached_name, config)
                        if backend is not None:
                            try:
                                if backend.load_model():
                                    logger.info("Using cached inference backend: %s", backend.name)
                                    return backend
                            except Exception as e:
                                logger.warning("Cached backend failed to load, trying others: %s", e)
            except Exception as e:
                logger.debug("Failed to load cached backend: %s", e)

        # Try each available backend
        backends = cls.create_backends(config)
        for backend in backends:
            try:
                logger.debug("Attempting inference backend: %s", backend.name)
                if backend.load_model():
                    logger.info("Successfully selected inference backend: %s", backend.name)

                    # Cache the selection
                    if cache_path:
                        try:
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(cache_path, 'w') as f:
                                json.dump({"inference_backend": backend.name}, f)
                            logger.debug("Cached inference backend selection: %s", backend.name)
                        except Exception as e:
                            logger.debug("Failed to cache backend selection: %s", e)

                    return backend
            except Exception as e:
                logger.warning("Inference backend '%s' failed: %s", backend.name, e)
                continue

        logger.error("No inference backend could be selected from: %s",
                    config.backend_preference or ("tflite", "onnx"))
        return None
