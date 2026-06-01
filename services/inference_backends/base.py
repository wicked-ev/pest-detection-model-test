"""Base inference backend abstraction."""

from __future__ import annotations

import abc
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias for detection results
Detection = Dict[str, object]


class InferenceBackendError(RuntimeError):
    """Exception raised by inference backends."""
    pass


class BaseInferenceBackend(abc.ABC):
    """Abstract base for inference backends.
    
    All backends must:
    1. Support loading a model from a file path
    2. Accept frames in RGB format (H x W x 3, uint8)
    3. Return detections in standard format
    4. Be thread-safe for single-threaded inference patterns
    """

    name = "base"

    def __init__(
        self,
        model_path: str,
        annotations_path: str,
        confidence_threshold: float = 0.5,
        num_threads: int = 1,
    ):
        self.model_path = model_path
        self.annotations_path = annotations_path
        self.confidence_threshold = float(confidence_threshold)
        self._class_names: List[str] = []
        self._input_shape: Optional[Tuple[int, int]] = None
        self._num_threads = int(num_threads)

    def _load_annotations(self, path: str) -> None:
        """Load COCO-format class annotations."""
        from pathlib import Path
        import json

        anno_path = Path(path)
        if not anno_path.exists():
            logger.warning("Annotations file missing: %s", anno_path)
            return

        try:
            data = json.loads(anno_path.read_text(encoding="utf-8"))
            categories = data.get("categories", [])
            cats = sorted(categories, key=lambda c: int(c.get("id", 0)))
            self._class_names = [c.get("name", "") for c in cats]
            logger.info("Loaded %d class names from annotations", len(self._class_names))
        except Exception:
            logger.exception("Failed to parse annotations JSON")

    def _normalize_box(
        self, box: Sequence[float], frame_shape: Tuple[int, int]
    ) -> Tuple[float, float, float, float]:
        """Normalize bounding box to frame coordinates."""
        h, w = frame_shape
        x1, y1, x2, y2 = box
        # If coordinates are normalized (0-1), scale to frame size
        if x1 <= 1.01 and y1 <= 1.01 and x2 <= 1.01 and y2 <= 1.01:
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        return float(x1), float(y1), float(x2), float(y2)

    def _parse_outputs(
        self, outputs: Sequence[np.ndarray], frame_shape: Tuple[int, int]
    ) -> List[Detection]:
        """Parse model outputs into standard detection format.
        
        Handles multiple output formats:
        - 4 outputs: [boxes, classes, scores, counts]
        - 3+ outputs: [boxes, scores, labels, ...]
        - 1 output: [N, 6+] where each row is [x1, y1, x2, y2, score, class, ...]
        """
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
                        x1, y1, x2, y2 = raw_box[0], raw_box[1], raw_box[2], raw_box[3]
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
                        x1, y1, x2, y2, score, label = row[:6]
                        score = float(score)
                        if score < self.confidence_threshold:
                            continue
                        label = int(label)
                        name = self._class_names[label] if 0 <= label < len(self._class_names) else str(label)
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
        """Check if backend dependencies are installed."""
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self) -> bool:
        """Load and initialize the model.
        
        Returns True if successful, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a frame.
        
        Args:
            frame: Input frame as numpy array (H, W, 3), uint8, RGB format
            
        Returns:
            List of detections, each with 'class', 'confidence', 'bbox', and timing info
        """
        raise NotImplementedError
