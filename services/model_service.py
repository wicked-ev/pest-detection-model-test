"""Model service with backend abstraction and streaming inference.

This service selects the best available backend based on configuration and
runs inference on a dedicated thread while dispatching detection callbacks.

The service uses an enhanced fallback architecture that:
- Tries inference backends in preference order
- Caches successful backend selection to avoid repeated startup penalties
- Supports multiple backend types (TFLite, ONNX, Remote)
- Provides clean error handling and recovery
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

import configs
from .inference_backends import (
    BaseInferenceBackend,
    Detection,
    InferenceBackendFactory,
    InferenceBackendConfig,
)

logger = logging.getLogger(__name__)
Listener = Callable[[List[Detection]], None]


class ModelService:
    """Unified model service that selects and runs an inference backend.
    
    Features:
    - Automatic backend selection with fallback strategy
    - Backend selection caching to avoid startup delays
    - Streaming inference on dedicated thread
    - Detection event dispatching to listeners
    - Graceful shutdown and error recovery
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        onnx_model_path: Optional[Path] = None,
        annotations_path: Optional[Path] = None,
        confidence_threshold: Optional[float] = None,
        num_threads: int = 1,
        backend_preference: Optional[Sequence[str]] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the model service.
        
        Args:
            model_path: Path to TFLite model (defaults from configs)
            onnx_model_path: Path to ONNX model (defaults from configs)
            annotations_path: Path to COCO annotations JSON
            confidence_threshold: Detection confidence threshold
            num_threads: Number of inference threads
            backend_preference: Backend selection order (e.g., ["tflite", "onnx"])
            cache_dir: Directory for caching backend selection
        """
        self.model_path = Path(model_path) if model_path is not None else Path(configs.MODEL_PATH)
        self.onnx_model_path = Path(onnx_model_path) if onnx_model_path is not None else Path(configs.ONNX_MODEL_PATH)
        self.annotations_path = Path(annotations_path) if annotations_path is not None else self.model_path.with_name("_annotations.coco.json")
        self.confidence_threshold = float(confidence_threshold) if confidence_threshold is not None else float(configs.MODEL_CONFIDENCE_THRESHOLD)
        self.backend_preference = (
            [item.strip().lower() for item in backend_preference]
            if backend_preference is not None
            else configs.MODEL_BACKEND_PREFERENCE
        )
        
        # Setup cache directory for backend selection
        self._cache_dir = Path(cache_dir) if cache_dir is not None else Path(configs.OUTPUT_DIR)
        self._backend_cache_path = self._cache_dir / ".backend_cache.json"
        
        # Create backend configuration
        self._backend_config = InferenceBackendConfig(
            model_path=self.model_path,
            onnx_model_path=self.onnx_model_path,
            annotations_path=self.annotations_path,
            confidence_threshold=self.confidence_threshold,
            num_threads=num_threads,
            backend_preference=self.backend_preference,
        )
        
        self._backend: Optional[BaseInferenceBackend] = None
        self._listeners: List[Listener] = []
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._inference_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_inference_ts: Optional[float] = None
        self._last_frame_id: Optional[int] = None
        self._streaming_active = False

    def get_backend_name(self) -> Optional[str]:
        """Get the name of the currently selected backend."""
        return self._backend.name if self._backend is not None else None

    def load_model(self) -> bool:
        """Load the inference model using backend selection with fallback strategy.
        
        Uses factory to:
        1. Check cache for previously successful backend
        2. Try each backend in preference order until one succeeds
        3. Cache the successful selection for next startup
        
        Returns True if a backend was successfully loaded, False otherwise.
        """
        self._backend = InferenceBackendFactory.select_best_backend(
            self._backend_config,
            cache_path=self._backend_cache_path,
        )
        return self._backend is not None

    def verify_load(self) -> bool:
        """Verify model is loaded. Alias for load_model() for compatibility."""
        return self.load_model()

    def add_listener(self, fn: Listener) -> None:
        """Add a detection event listener."""
        self._listeners.append(fn)

    def _reset_executor(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            logger.exception("Failed to shutdown executor")
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _dispatch(self, detections: List[Detection]) -> None:
        if not detections:
            return
        for listener in list(self._listeners):
            try:
                self._executor.submit(listener, detections)
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

    def _ensure_backend(self) -> BaseInferenceBackend:
        if self._backend is None:
            raise RuntimeError("Inference backend not loaded")
        return self._backend

    def run_inference_on_image(self, frame: np.ndarray) -> List[Detection]:
        backend = self._ensure_backend()
        start = time.time()
        detections = backend.run(frame)
        elapsed = time.time() - start
        for detection in detections:
            detection["inference_time_s"] = elapsed
            detection["timestamp"] = datetime.utcnow().isoformat() + "Z"
        return detections

    def test_mode(self, sample_image: Optional[Path] = None) -> None:
        if not self.load_model():
            logger.error("Model load failed in test mode")
            return

        img: Optional[np.ndarray] = None
        if sample_image and sample_image.exists():
            try:
                pil_img = Image.open(str(sample_image)).convert('RGB')
                img = np.array(pil_img)
            except Exception:
                logger.exception("Failed to load image from file: %s", sample_image)

        if img is None:
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
        logger.info("Test detected %d objects using backend %s", len(dets), self.get_backend_name())
        for d in dets:
            logger.info(
                "%s %.3f bbox=%s time=%.3fs",
                d.get("class"),
                d.get("confidence"),
                d.get("bbox"),
                d.get("inference_time_s"),
            )

    def start_streaming(self, camera_service, throttle_fps: Optional[float] = None) -> None:
        if self._inference_thread and self._inference_thread.is_alive():
            return
        if self._backend is None and not self.load_model():
            raise RuntimeError("Model load failed")

        self._stop_event.clear()
        self._streaming_active = True

        def loop() -> None:
            min_frame_interval = 1.0 / throttle_fps if throttle_fps else 0.0
            last_infer = 0.0
            last_frame_id: Optional[int] = None
            frame_counter = 0
            self._last_frame_id = None
            self._last_inference_ts = time.time()
            while not self._stop_event.is_set():
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

                last_frame_id = frame_id
                frame_counter += 1
                if configs.FRAME_SKIP and frame_counter % configs.FRAME_SKIP != 0:
                    continue

                if min_frame_interval and (time.time() - last_infer) < min_frame_interval:
                    time.sleep(0.001)
                    continue

                last_infer = time.time()
                self._last_inference_ts = last_infer
                try:
                    dets = self.run_inference_on_image(frame)
                except Exception:
                    logger.exception("Inference loop encountered an error")
                    dets = []
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
