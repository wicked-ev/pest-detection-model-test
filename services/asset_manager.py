"""Asset manager to ensure model assets are present and valid.

Responsibilities:
- Check for presence of model and annotations
- Download missing assets using DownloadService
- Validate files (size > 0, JSON parse, ONNX load via onnxruntime)
- Retry and cleanup on failure
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .download_service import DownloadService, DownloadError
import configs

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except Exception:
    ort = None


class AssetManager:
    def __init__(self, download_service: Optional[DownloadService] = None):
        self.download_service = download_service or DownloadService()
        self.model_path: Path = Path(configs.MODEL_PATH)
        self.annotations_path: Path = Path(configs.MODEL_PATH).with_name("_annotations.coco.json")
        self.model_url = configs.MODEL_DOWNLOAD_URL
        self.annotations_url = configs.ANNOTATIONS_DOWNLOAD_URL

    def _validate_json(self, path: Path) -> bool:
        try:
            text = path.read_text(encoding="utf-8")
            json.loads(text)
            return True
        except Exception:
            logger.exception("Annotations JSON validation failed: %s", path)
            return False

    def _validate_onnx(self, path: Path) -> bool:
        if ort is None:
            logger.warning("onnxruntime not available; skipping ONNX validation")
            return True
        try:
            # try to create a lightweight session
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            _ = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            return True
        except Exception:
            logger.exception("ONNX validation failed for %s", path)
            return False

    def _is_valid_file(self, path: Path, kind: str) -> bool:
        if not path.exists():
            logger.debug("Asset missing: %s", path)
            return False
        try:
            if path.stat().st_size == 0:
                logger.warning("Asset %s is empty: %s", kind, path)
                return False
        except Exception:
            logger.exception("Failed to stat path %s", path)
            return False

        if kind == "annotations":
            return self._validate_json(path)
        if kind == "model":
            return self._validate_onnx(path)
        return True

    def ensure_assets(self) -> bool:
        """Ensure model and annotation files exist and are valid. Returns True when assets are ready."""
        # Model
        if not self._is_valid_file(self.model_path, "model"):
            logger.info("Model missing or invalid, downloading from %s", self.model_url)
            try:
                self.download_service.download(self.model_url, self.model_path)
            except DownloadError as exc:
                logger.error("Model download failed: %s", exc)
                return False
            if not self._is_valid_file(self.model_path, "model"):
                logger.error("Model validation failed after download")
                try:
                    self.model_path.unlink()
                except Exception:
                    logger.exception("Failed to remove invalid model file")
                return False

        # Annotations
        if not self._is_valid_file(self.annotations_path, "annotations"):
            logger.info("Annotations missing or invalid, downloading from %s", self.annotations_url)
            try:
                self.download_service.download(self.annotations_url, self.annotations_path)
            except DownloadError as exc:
                logger.error("Annotations download failed: %s", exc)
                return False
            if not self._is_valid_file(self.annotations_path, "annotations"):
                logger.error("Annotations validation failed after download")
                try:
                    self.annotations_path.unlink()
                except Exception:
                    logger.exception("Failed to remove invalid annotations file")
                return False

        logger.info("All assets ready: model=%s annotations=%s", self.model_path, self.annotations_path)
        return True
