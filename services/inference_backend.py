"""Inference backend abstraction for TensorFlow Lite and ONNX Runtime."""

from __future__ import annotations

import abc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import configs

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - handled at runtime
    ort = None

try:
    import tflite_runtime.interpreter as tflite_runtime
    TFLiteInterpreter = tflite_runtime.Interpreter
except Exception:
    TFLiteInterpreter = None
    try:
        from tensorflow.lite import Interpreter as TFLiteInterpreter
    except Exception:
        TFLiteInterpreter = None

Detection = Dict[str, object]


class InferenceBackendError(RuntimeError):
    pass


class BaseInferenceBackend(abc.ABC):
    """Abstract inference backend interface."""

    name = "base"

    def __init__(
        self,
        model_path: Path,
        annotations_path: Path,
        confidence_threshold: float = 0.5,
        num_threads: int = 1,
    ):
        self.model_path = Path(model_path)
        self.annotations_path = Path(annotations_path)
        self.confidence_threshold = float(confidence_threshold)
        self._class_names: List[str] = []
        self._input_shape: Optional[Tuple[int, int]] = None
        self._input_dtype = np.float32
        self._input_layout = "NHWC"
        self._num_threads = int(num_threads)
        self._input_buffer: Optional[np.ndarray] = None

    def _load_annotations(self) -> None:
        if not self.annotations_path.exists():
            logger.warning("Annotations file missing: %s", self.annotations_path)
            return
        try:
            data = json.loads(self.annotations_path.read_text(encoding="utf-8"))
            categories = data.get("categories", [])
            cats = sorted(categories, key=lambda c: int(c.get("id", 0)))
            self._class_names = [c.get("name", "") for c in cats]
            logger.info("Loaded %d class names from annotations", len(self._class_names))
        except Exception:
            logger.exception("Failed to parse annotations JSON")

    def _normalize_box(
        self, box: Sequence[float], frame_shape: Tuple[int, int]
    ) -> Tuple[float, float, float, float]:
        h, w = frame_shape
        x1, y1, x2, y2 = box
        if x1 <= 1.01 and y1 <= 1.01 and x2 <= 1.01 and y2 <= 1.01:
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        return float(x1), float(y1), float(x2), float(y2)

    def _parse_outputs(
        self, outputs: Sequence[np.ndarray], frame_shape: Tuple[int, int]
    ) -> List[Detection]:
        detections: List[Detection] = []
        h, w = frame_shape

        try:
            if len(outputs) == 4:
                boxes, classes, scores, counts = outputs
                num_detections = int(np.asarray(counts).flatten()[0])
                classes = np.asarray(classes).reshape(-1)
                scores = np.asarray(scores).reshape(-1)
                boxes = np.asarray(boxes)

                for i in range(min(num_detections, boxes.shape[0])):
                    score = float(scores[i])
                    if score < self.confidence_threshold:
                        continue
                    label = int(classes[i]) if classes is not None else 0
                    name = self._class_names[label] if 0 <= label < len(self._class_names) else str(label)
                    raw_box = boxes[i].tolist()
                    if len(raw_box) == 4:
                        x1, y1, x2, y2 = raw_box
                    else:
                        continue
                    x1, y1, x2, y2 = self._normalize_box((x1, y1, x2, y2), frame_shape)
                    detections.append({
                        "class": name,
                        "confidence": score,
                        "bbox": (x1, y1, x2, y2),
                    })
                return detections

            if len(outputs) >= 3:
                boxes = outputs[0]
                scores = outputs[1]
                labels = outputs[2]
                for i in range(boxes.shape[0]):
                    score = float(scores[i])
                    if score < self.confidence_threshold:
                        continue
                    lab = int(labels[i]) if labels is not None else 0
                    name = self._class_names[lab] if 0 <= lab < len(self._class_names) else str(lab)
                    x1, y1, x2, y2 = boxes[i].tolist()
                    x1, y1, x2, y2 = self._normalize_box((x1, y1, x2, y2), frame_shape)
                    detections.append({
                        "class": name,
                        "confidence": score,
                        "bbox": (x1, y1, x2, y2),
                    })
                return detections

            if len(outputs) == 1:
                out = outputs[0]
                if out.ndim == 2 and out.shape[1] >= 6:
                    for row in out:
                        score = float(row[4])
                        if score < self.confidence_threshold:
                            continue
                        lab = int(row[5])
                        name = self._class_names[lab] if 0 <= lab < len(self._class_names) else str(lab)
                        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                        x1, y1, x2, y2 = self._normalize_box((x1, y1, x2, y2), frame_shape)
                        detections.append({
                            "class": name,
                            "confidence": score,
                            "bbox": (x1, y1, x2, y2),
                        })
                    return detections
        except Exception:
            logger.exception("Failed to parse model outputs")

        return detections

    @abc.abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class TFLiteBackend(BaseInferenceBackend):
    name = "tflite"

    def __init__(
        self,
        model_path: Path,
        annotations_path: Path,
        confidence_threshold: float = 0.5,
        num_threads: int = 1,
    ):
        super().__init__(model_path, annotations_path, confidence_threshold, num_threads)
        self._interpreter: Optional[Any] = None
        self._input_index: Optional[int] = None
        self._output_details: List[Dict[str, Any]] = []

    def is_available(self) -> bool:
        return TFLiteInterpreter is not None

    def load_model(self) -> bool:
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
                self._input_shape = (configs.INFERENCE_HEIGHT, configs.INFERENCE_WIDTH)

            self._output_details = output_details
            self._load_annotations()
            logger.info("TFLite model loaded: input=%s dtype=%s layout=%s", self._input_shape, self._input_dtype, self._input_layout)
            return True
        except Exception:
            logger.exception("Failed to create TensorFlow Lite interpreter")
            return False

    def _prepare_input(self, frame: np.ndarray) -> np.ndarray:
        if self._input_shape is None:
            raise InferenceBackendError("TFLite input shape is unknown")

        target_h, target_w = self._input_shape
        image = frame
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._input_dtype == np.float32:
            image = image.astype(np.float32) / 255.0
        elif self._input_dtype == np.uint8:
            image = image.astype(np.uint8)
        else:
            image = image.astype(self._input_dtype)

        if self._input_layout == "NCHW":
            image = np.transpose(image, (2, 0, 1))

        tensor_shape = (
            1,
            target_h,
            target_w,
            3,
        ) if self._input_layout == "NHWC" else (1, 3, target_h, target_w)

        if self._input_buffer is None or self._input_buffer.shape != tensor_shape or self._input_buffer.dtype != self._input_dtype:
            self._input_buffer = np.empty(tensor_shape, dtype=self._input_dtype)

        self._input_buffer[0] = image
        return self._input_buffer

    def run(self, frame: np.ndarray) -> List[Detection]:
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


class ONNXBackend(BaseInferenceBackend):
    name = "onnx"

    def __init__(
        self,
        model_path: Path,
        annotations_path: Path,
        confidence_threshold: float = 0.5,
        num_threads: int = 1,
    ):
        super().__init__(model_path, annotations_path, confidence_threshold, num_threads)
        self._session: Optional[Any] = None
        self._input_name: Optional[str] = None

    def is_available(self) -> bool:
        return ort is not None

    def load_model(self) -> bool:
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
            self._session = ort.InferenceSession(str(self.model_path), so, providers=["CPUExecutionProvider"])
            inputs = self._session.get_inputs()
            if not inputs:
                logger.error("ONNX model has no inputs")
                return False

            inp = inputs[0]
            self._input_name = inp.name
            raw_shape = [dim if isinstance(dim, int) else None for dim in inp.shape]
            self._input_shape = self._infer_input_shape(raw_shape)
            self._input_dtype = np.float32
            self._load_annotations()
            logger.info("ONNX model loaded: input=%s name=%s", self._input_shape, self._input_name)
            return True
        except Exception:
            logger.exception("Failed to create ONNX Runtime session")
            return False

    def _infer_input_shape(self, raw_shape: List[Optional[int]]) -> Tuple[int, int]:
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
        return (configs.INFERENCE_HEIGHT, configs.INFERENCE_WIDTH)

    def _prepare_input(self, frame: np.ndarray) -> np.ndarray:
        if self._session is None or self._input_name is None or self._input_shape is None:
            raise InferenceBackendError("ONNX backend is not loaded")

        target_h, target_w = self._input_shape
        image = frame
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        tensor = np.transpose(image, (2, 0, 1))[None, :, :, :]
        return np.ascontiguousarray(tensor, dtype=np.float32)

    def run(self, frame: np.ndarray) -> List[Detection]:
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


class BackendFactory:
    BACKENDS = {
        "tflite": TFLiteBackend,
        "onnx": ONNXBackend,
    }

    @classmethod
    def create_backend(cls, name: str, **kwargs: Any) -> Optional[BaseInferenceBackend]:
        backend_cls = cls.BACKENDS.get(name.lower())
        if backend_cls is None:
            logger.warning("Unknown inference backend requested: %s", name)
            return None
        return backend_cls(**kwargs)

    @classmethod
    def create_backends(
        cls,
        preference: Sequence[str],
        model_path: Path,
        onnx_model_path: Path,
        annotations_path: Path,
        confidence_threshold: float = 0.5,
        num_threads: int = 1,
    ) -> List[BaseInferenceBackend]:
        backends: List[BaseInferenceBackend] = []
        for name in preference:
            if name == "tflite":
                backend = cls.create_backend(
                    name,
                    model_path=model_path,
                    annotations_path=annotations_path,
                    confidence_threshold=confidence_threshold,
                    num_threads=num_threads,
                )
            elif name == "onnx":
                backend = cls.create_backend(
                    name,
                    model_path=onnx_model_path,
                    annotations_path=annotations_path,
                    confidence_threshold=confidence_threshold,
                    num_threads=num_threads,
                )
            else:
                backend = cls.create_backend(
                    name,
                    model_path=model_path,
                    annotations_path=annotations_path,
                    confidence_threshold=confidence_threshold,
                    num_threads=num_threads,
                )
            if backend is not None:
                backends.append(backend)
        return backends
