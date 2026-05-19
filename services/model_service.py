"""Model service using ONNX Runtime with a small event system.

This implementation is conservative for Raspberry Pi 3 B+:
- small worker threads
- reuses buffers where possible
- non-blocking event dispatch using a thread pool
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - handled at runtime
    ort = None

import configs

logger = logging.getLogger(__name__)


Detection = Dict[str, object]
Listener = Callable[[List[Detection]], None]


class ModelService:
    """ONNX-based model service with streaming inference and event callbacks.

    It supports loading class names from a COCO-style annotations JSON and
    runs inference in a dedicated thread while pulling latest frames from
    the camera service.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        annotations_path: Optional[Path] = None,
        confidence_threshold: float = 0.3,
        intra_op_threads: int = 1,
    ):
        self.model_path = Path(model_path) if model_path is not None else Path(configs.MODEL_PATH)
        self.annotations_path = Path(annotations_path) if annotations_path is not None else Path(configs.MODEL_PATH).with_name("_annotations.coco.json")
        self.confidence_threshold = float(confidence_threshold)
        self._session: Optional[ort.InferenceSession] = None if ort else None
        self._input_name: Optional[str] = None
        self._input_shape: Optional[Tuple[int, int]] = None  # (H, W)
        self._class_names: List[str] = []
        self._listeners: List[Listener] = []
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._inference_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._intra_op_threads = int(intra_op_threads)
        self._last_inference_ts: Optional[float] = None
        self._last_frame_id: Optional[int] = None
        self._streaming_active = False

    def _load_annotations(self) -> None:
        if not self.annotations_path.exists():
            logger.warning("Annotations file missing: %s", self.annotations_path)
            return
        try:
            data = json.loads(self.annotations_path.read_text(encoding="utf-8"))
            categories = data.get("categories", [])
            # categories are list of {id, name, supercategory}
            # build a mapping from id->name sorted by id
            cats = sorted(categories, key=lambda c: int(c.get("id", 0)))
            self._class_names = [c.get("name", "") for c in cats]
            logger.info("Loaded %d class names from annotations", len(self._class_names))
        except Exception:
            logger.exception("Failed to parse annotations JSON")

    def load_model(self) -> bool:
        if ort is None:
            logger.error("onnxruntime is not installed")
            return False
        if not self.model_path.exists():
            logger.error("Model file missing: %s", self.model_path)
            return False

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, self._intra_op_threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self._session = ort.InferenceSession(str(self.model_path), so, providers=["CPUExecutionProvider"])
            inputs = self._session.get_inputs()
            if not inputs:
                logger.error("ONNX model has no inputs")
                return False
            inp = inputs[0]
            self._input_name = inp.name
            raw_shape = list(inp.shape)
            shape = [dim if isinstance(dim, int) else None for dim in raw_shape]
            # shape often (batch, channels, H, W) or (batch, H, W, channels)
            if len(shape) == 4:
                batch, c0, c1, c2 = shape
                if c0 == 3 and c1 and c2:
                    self._input_shape = (c1, c2)
                elif c2 == 3 and c1 and c0:
                    self._input_shape = (c0, c1)
                else:
                    self._input_shape = (configs.INFERENCE_HEIGHT, configs.INFERENCE_WIDTH)
            elif len(shape) == 3:
                c0, c1, c2 = shape
                if c0 == 3 and c1 and c2:
                    self._input_shape = (c1, c2)
                elif c2 == 3 and c0 and c1:
                    self._input_shape = (c0, c1)
                else:
                    self._input_shape = (configs.INFERENCE_HEIGHT, configs.INFERENCE_WIDTH)
            else:
                self._input_shape = (configs.INFERENCE_HEIGHT, configs.INFERENCE_WIDTH)
            if self._input_shape is None:
                self._input_shape = (configs.INFERENCE_HEIGHT, configs.INFERENCE_WIDTH)

            logger.info("ONNX model loaded: input=%s shape=%s", self._input_name, self._input_shape)
            self._load_annotations()
            return True
        except Exception:
            logger.exception("Failed to create ONNX Runtime session")
            return False

    def verify_load(self) -> bool:
        """Verify the model can load successfully without raising exceptions."""
        return self.load_model()

    def add_listener(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def _reset_executor(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            logger.exception("Failed to shutdown executor")
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _dispatch(self, detections: List[Detection]) -> None:
        # non-blocking dispatch of all detections from a single frame
        if not detections:
            return
        for l in list(self._listeners):
            try:
                self._executor.submit(l, detections)
            except Exception:
                logger.exception("Listener submit failed")

    def is_streaming(self) -> bool:
        return self._inference_thread is not None and self._inference_thread.is_alive() and self._streaming_active

    def get_last_inference_age(self) -> Optional[float]:
        if self._last_inference_ts is None:
            return None
        return time.time() - self._last_inference_ts

    def restart_streaming(self, camera_service, throttle_fps: Optional[float] = None) -> None:
        self.stop_streaming()
        self.start_streaming(camera_service, throttle_fps)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # convert BGR->RGB, resize and normalize to [0,1]
        h, w = frame.shape[:2]
        target_h, target_w = self._input_shape
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (h, w) != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        arr = img.astype(np.float32) / 255.0
        # produce NCHW
        # Many ONNX models expect [N,C,H,W]
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        return arr

    def _parse_outputs(self, outputs: Sequence[np.ndarray], frame_shape: Tuple[int, int]) -> List[Detection]:
        """Attempt to parse common detection output formats.

        Returns list of dicts: class, score, bbox(x1,y1,x2,y2) in pixels.
        """
        detections: List[Detection] = []
        h, w = frame_shape
        try:
            if len(outputs) >= 3:
                # common ordering: boxes, scores, labels
                boxes = outputs[0]
                scores = outputs[1]
                labels = outputs[2]
                # ensure shapes align
                for i in range(boxes.shape[0]):
                    score = float(scores[i])
                    if score < self.confidence_threshold:
                        continue
                    lab = int(labels[i]) if labels is not None else 0
                    name = self._class_names[lab] if 0 <= lab < len(self._class_names) else str(lab)
                    x1, y1, x2, y2 = boxes[i]
                    # boxes may be normalized 0-1
                    if x1 <= 1.01 and y1 <= 1.01 and x2 <= 1.01 and y2 <= 1.01:
                        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
                    detections.append({
                        "class": name,
                        "confidence": score,
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    })
                return detections

            # fallback: single tensor containing [N,6] -> x1,y1,x2,y2,score,label
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
                        if x1 <= 1.01 and y1 <= 1.01 and x2 <= 1.01 and y2 <= 1.01:
                            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
                        detections.append({
                            "class": name,
                            "confidence": score,
                            "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        })
                    return detections
        except Exception:
            logger.exception("Failed to parse model outputs")
        return detections

    def run_inference_on_image(self, frame: np.ndarray) -> List[Detection]:
        if self._session is None or self._input_name is None or self._input_shape is None:
            raise RuntimeError("Model not loaded")
        inp = self._preprocess(frame)
        # reuse tensor memory by ensuring contiguous
        inp = np.ascontiguousarray(inp, dtype=np.float32)
        try:
            start = time.time()
            raw_outs = self._session.run(None, {self._input_name: inp})
            elapsed = time.time() - start
            logger.debug("Inference time: %.3f s", elapsed)
            # convert outputs to numpy arrays if not already
            outs = [np.asarray(o) for o in raw_outs]
            dets = self._parse_outputs(outs, frame.shape[:2])
            # attach timing metadata
            for d in dets:
                d["inference_time_s"] = elapsed
                d["timestamp"] = datetime.utcnow().isoformat() + "Z"
            return dets
        except Exception:
            logger.exception("Inference failed")
            return []

    def test_mode(self, sample_image: Optional[Path] = None) -> None:
        """Quick test: load model and run inference on `sample_image` (or camera).

        Prints detected classes, confidences and timing.
        """
        if not self.load_model():
            logger.error("Model load failed in test mode")
            return
        img = None
        if sample_image and sample_image.exists():
            img = cv2.imread(str(sample_image))
        if img is None:
            # fallback to single camera capture
            from services.camera_service import CameraService, CameraConfig

            cam = CameraService(CameraConfig())
            cam.start()
            t0 = time.time()
            while time.time() - t0 < 3.0:
                got = cam.get_latest(copy=True)
                if got:
                    img, _ = got
                    break
                time.sleep(0.05)
            cam.stop()
        if img is None:
            logger.error("No image available for test")
            return
        dets = self.run_inference_on_image(img)
        logger.info("Test detected %d objects", len(dets))
        for d in dets:
            logger.info("%s %.3f bbox=%s time=%.3fs", d.get("class"), d.get("confidence"), d.get("bbox"), d.get("inference_time_s"))

    def start_streaming(self, camera_service, throttle_fps: Optional[float] = None) -> None:
        """Start continuous inference reading from `camera_service`.

        The inference loop always pulls the newest frame and skips frames
        when busy to minimize latency.
        """
        if self._inference_thread and self._inference_thread.is_alive():
            return
        if self._session is None:
            if not self.load_model():
                raise RuntimeError("Model load failed")
        self._stop_event.clear()
        self._streaming_active = True
        def loop():
            min_frame_interval = 1.0 / throttle_fps if throttle_fps else 0.0
            last_infer = 0.0
            last_frame_id: Optional[int] = None
            self._last_frame_id = None
            self._last_inference_ts = time.time()
            while not self._stop_event.is_set():
                res = None
                try:
                    res = camera_service.get_latest_metadata(copy=False)
                except AttributeError:
                    res = camera_service.get_latest(copy=False)
                if res is None:
                    time.sleep(0.005)
                    continue
                if len(res) == 3:
                    frame, ts, frame_id = res
                else:
                    frame, ts = res
                    frame_id = None
                if frame_id is not None and last_frame_id is not None and frame_id == last_frame_id:
                    time.sleep(0.005)
                    continue
                # throttle by desired fps
                if min_frame_interval and (time.time() - last_infer) < min_frame_interval:
                    time.sleep(0.001)
                    continue
                last_frame_id = frame_id
                last_infer = time.time()
                self._last_inference_ts = last_infer
                dets = self.run_inference_on_image(frame)
                self._dispatch(dets)
            logger.info("Inference streaming stopped")

        self._inference_thread = threading.Thread(target=loop, name="inference-stream", daemon=True)
        self._inference_thread.start()

    def stop_streaming(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._inference_thread:
            self._inference_thread.join(timeout)
        self._streaming_active = False
        self._reset_executor()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    svc = ModelService()
    svc.test_mode()
