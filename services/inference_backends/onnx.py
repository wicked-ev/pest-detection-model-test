"""ONNX Runtime inference backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import BaseInferenceBackend, InferenceBackendError, Detection

logger = logging.getLogger(__name__)

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
except Exception:
    ort = None


class ONNXBackend(BaseInferenceBackend):
    """ONNX Runtime inference backend.
    
    Alternative backend supporting ONNX model format.
    Works across platforms where ONNX Runtime is available.
    
    Dependencies:
    - onnxruntime
    
    Advantages:
    - Wide model format support
    - Works on multiple platforms
    - Good performance on CPUs
    
    Disadvantages:
    - May not be available on all environments
    - Not always the best choice for resource-constrained devices
    """

    name = "onnx"

    def __init__(
        self,
        model_path: Path | str,
        annotations_path: Path | str,
        confidence_threshold: float = 0.5,
        num_threads: int = 1,
    ):
        super().__init__(str(model_path), str(annotations_path), confidence_threshold, num_threads)
        self.model_path = Path(model_path)
        self.annotations_path = Path(annotations_path)
        self._session: Optional[Any] = None
        self._input_name: Optional[str] = None
        self._input_dtype = np.float32

    def is_available(self) -> bool:
        """Check if ONNX Runtime is installed."""
        return ort is not None

    def load_model(self) -> bool:
        """Load ONNX model."""
        if not self.is_available():
            logger.warning("ONNX Runtime is unavailable")
            return False

        if not self.model_path.exists():
            logger.error("ONNX model file missing: %s", self.model_path)
            return False

        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, self._num_threads)
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(self.model_path), so, providers=["CPUExecutionProvider"]
            )
            inputs = self._session.get_inputs()
            if not inputs:
                logger.error("ONNX model has no inputs")
                return False

            inp = inputs[0]
            self._input_name = inp.name
            raw_shape = [dim if isinstance(dim, int) else None for dim in inp.shape]
            self._input_shape = self._infer_input_shape(raw_shape)
            self._input_dtype = np.float32
            self._load_annotations(str(self.annotations_path))
            logger.info("ONNX model loaded: input=%s name=%s",
                       self._input_shape, self._input_name)
            return True
        except Exception:
            logger.exception("Failed to create ONNX Runtime session")
            return False

    def _infer_input_shape(self, raw_shape: List[Optional[int]]) -> Tuple[int, int]:
        """Infer input shape from model metadata."""
        if len(raw_shape) == 4:
            _, c0, c1, c2 = raw_shape
            if c0 == 3 and c1 and c2:
                return (c1, c2)
            if c2 == 3 and c0 and c1:
                return (c0, c1)
        if len(raw_shape) == 3:
            c0, c1, c2 = raw_shape
            if c0 == 3 and c1 and c2:
                return (c1, c2)
            if c2 == 3 and c0 and c1:
                return (c0, c1)
        # Fallback to default inference size
        import configs
        return (configs.INFERENCE_HEIGHT, configs.INFERENCE_WIDTH)

    def _prepare_input(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for ONNX inference."""
        if self._session is None or self._input_name is None or self._input_shape is None:
            raise InferenceBackendError("ONNX backend is not loaded")

        target_h, target_w = self._input_shape
        image = frame
        if image.shape[:2] != (target_h, target_w):
            pil_img = Image.fromarray(image.astype(np.uint8), mode='RGB')
            pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
            image = np.array(pil_img)

        # Normalize to float32
        image = image.astype(np.float32) / 255.0

        # Convert to NCHW format with batch dimension
        tensor = np.transpose(image, (2, 0, 1))[None, :, :, :]
        return np.ascontiguousarray(tensor, dtype=np.float32)

    def run(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a frame."""
        if self._session is None or self._input_name is None:
            raise InferenceBackendError("ONNX backend is not loaded")

        inp = self._prepare_input(frame)
        try:
            raw_outputs = self._session.run(None, {self._input_name: inp})
            outputs = [np.asarray(o) for o in raw_outputs]
            return self._parse_outputs(outputs, frame.shape[:2])
        except Exception:
            logger.exception("ONNX inference failed")
            return []
