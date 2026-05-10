"""
AI model validation service.

Provides a lightweight abstraction that can be expanded into full
model loading and validation without coupling startup flow to the
inference engine directly.
"""

import logging
from pathlib import Path
from typing import Optional

import configs

logger = logging.getLogger(__name__)


class ModelService:
    """AI model validation and loader abstraction."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path is not None else configs.MODEL_PATH

    def exists(self) -> bool:
        """Return True if the model file exists."""
        exists = self.model_path.exists()
        if exists:
            logger.info(f"AI model file found at {self.model_path}")
        else:
            logger.warning(f"AI model file missing at {self.model_path}")
        return exists

    def verify_load(self) -> bool:
        """Placeholder for model loading verification."""
        if not self.exists():
            return False
        logger.info("Model verification placeholder succeeded")
        return True
