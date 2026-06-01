"""TensorFlow Lite inference backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import BaseInferenceBackend, InferenceBackendError, Detection

logger = logging.getLogger(__name__)

# Try to import TFLite runtime
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except Exception:
    TFLiteInterpreter = None
    try:
        from tensorflow.lite import Interpreter as TFLiteInterpreter
    except Exception:
        TFLiteInterpreter = None


class TFLiteBackend(BaseInferenceBackend):
    """TensorFlow Lite inference backend optimized for Raspberry Pi.
    
    Preferred backend for edge devices with limited resources.
    
    Dependencies:
    - tflite-runtime (preferred) OR tensorflow
    
    Advantages:
    - Minimal memory footprint
    - Fast inference on mobile/embedded devices
    - Supports quantized models
    
    Disadvantages:
    - Limited model format support compared to ONNX
    - Some exported models may have unsupported operations
    """

    name = "tflite"

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
        self._interpreter: Optional[Any] = None
        self._input_index: Optional[int] = None
        self._output_details: List[Dict[str, Any]] = []
        self._input_dtype = np.float32
        self._input_layout = "NHWC"
        self._input_buffer: Optional[np.ndarray] = None

    def is_available(self) -> bool:
        """Check if TensorFlow Lite runtime is installed."""
        return TFLiteInterpreter is not None

    def load_model(self) -> bool:
        """Load TensorFlow Lite model."""
        if not self.is_available():
            logger.warning("TensorFlow Lite runtime is unavailable")
            return False

        if not self.model_path.exists():
            logger.error("TFLite model file missing: %s", self.model_path)
            return False

        try:
            self._interpreter = TFLiteInterpreter(
                model_path=str(self.model_path), num_threads=self._num_threads
            )
            self._interpreter.allocate_tensors()
            input_details = self._interpreter.get_input_details()
            output_details = self._interpreter.get_output_details()
            if not input_details or not output_details:
                logger.error("TFLite model does not expose valid input/output tensors")
                return False

            input_detail = input_details[0]
            shape = input_detail.get("shape", [])
            dtype = input_detail.get("dtype", np.float32)
            self._input_dtype = np.dtype(dtype)
            self._input_index = int(input_detail["index"])

            if len(shape) == 4:
                _, height, width, channels = shape
                if channels == 3:
                    self._input_layout = "NHWC"
                    self._input_shape = (int(height), int(width))
                else:
                    self._input_layout = "NCHW"
                    self._input_shape = (int(shape[2]), int(shape[3]))
            else:
                # Use default inference size from config if shape is unclear
                import configs
                self._input_shape = (configs.INFERENCE_HEIGHT, configs.INFERENCE_WIDTH)

            self._output_details = output_details
            self._load_annotations(str(self.annotations_path))
            logger.info("TFLite model loaded: input=%s dtype=%s layout=%s",
                       self._input_shape, self._input_dtype, self._input_layout)
            return True
        except Exception:
            logger.exception("Failed to create TensorFlow Lite interpreter")
            return False

    def _prepare_input(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for TFLite inference."""
        if self._input_shape is None:
            raise InferenceBackendError("TFLite input shape is unknown")

        target_h, target_w = self._input_shape
        image = frame
        if image.shape[:2] != (target_h, target_w):
            pil_img = Image.fromarray(image.astype(np.uint8), mode='RGB')
            pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
            image = np.array(pil_img)

        # Normalize or convert to target dtype
        if self._input_dtype == np.float32:
            image = image.astype(np.float32) / 255.0
        elif self._input_dtype == np.uint8:
            image = image.astype(np.uint8)
        else:
            image = image.astype(self._input_dtype)

        # Convert to NCHW if needed
        if self._input_layout == "NCHW":
            image = np.transpose(image, (2, 0, 1))

        tensor_shape = (
            1, target_h, target_w, 3,
        ) if self._input_layout == "NHWC" else (1, 3, target_h, target_w)

        # Reuse buffer if possible
        if self._input_buffer is None or self._input_buffer.shape != tensor_shape or \
           self._input_buffer.dtype != self._input_dtype:
            self._input_buffer = np.empty(tensor_shape, dtype=self._input_dtype)

        self._input_buffer[0] = image
        return self._input_buffer

    def run(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a frame."""
        if self._interpreter is None or self._input_index is None:
            raise InferenceBackendError("TFLite backend is not loaded")

        input_tensor = self._prepare_input(frame)
        self._interpreter.set_tensor(self._input_index, input_tensor)
        try:
            self._interpreter.invoke()
            outputs = [
                np.asarray(self._interpreter.get_tensor(output["index"]))
                for output in self._output_details
            ]
            return self._parse_outputs(outputs, frame.shape[:2])
        except Exception:
            logger.exception("TFLite inference failed")
            return []
