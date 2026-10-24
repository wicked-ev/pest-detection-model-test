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

import io

import numpy as np
from typing import Any

import configs
from .inference_backends import (
    BaseInferenceBackend,
    Detection,
    InferenceBackendFactory,
    InferenceBackendConfig,
)

logger = logging.getLogger(__name__)
Listener = Callable[[List[Detection]], None]


def encode_jpeg(frame: np.ndarray, quality: Optional[int] = None) -> bytes:
    """Encode an RGB uint8 frame as JPEG; returns b"" if it cannot be encoded."""
    quality = int(quality if quality is not None else getattr(configs, "JPEG_QUALITY", 75))
    try:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(frame, mode="RGB").save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()
    except ImportError:
        pass
    except Exception:
        logger.exception("JPEG encoding failed")
        return b""
    try:
        import cv2

        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return jpeg.tobytes() if ok else b""
    except Exception:
        logger.exception("JPEG encoding failed (neither Pillow nor OpenCV worked)")
        return b""


def _unpack_frame(res: Any) -> Tuple[Any, float, Optional[int]]:
    """Normalize camera service results into (frame, timestamp, frame_id)."""
    if isinstance(res, tuple):
        if len(res) == 3:
            return res[0], float(res[1]), res[2]
        if len(res) == 2:
            return res[0], float(res[1]), None
    if isinstance(res, np.ndarray):
        return res, time.time(), None
    raise TypeError(f"Unsupported camera result type: {type(res)!r}")


def _latest_frame(camera_service) -> Any:
    try:
        return camera_service.get_latest_metadata(copy=False)
    except AttributeError:
        return camera_service.get_latest(copy=False)


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
        num_threads: Optional[int] = None,
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
            # The Pi 3B+ has 4 cores; a single thread would leave three idle.
            num_threads=num_threads if num_threads is not None else configs.MODEL_NUM_THREADS,
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
        # Remote streaming (fallback) state
        self._remote_thread: Optional[threading.Thread] = None
        self._remote_stop_event = threading.Event()
        self._remote_streaming_active = False
        self._last_remote_activity_ts: Optional[float] = None
        self._remote_network_service: Optional[Any] = None
        # Set in monitor mode: local detections are also sent to the server through it.
        self._monitor_network_service: Optional[Any] = None

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

    def has_runtime(self) -> bool:
        """True if at least one inference runtime (tflite, onnx) is installed. Loads nothing."""
        return bool(InferenceBackendFactory.create_backends(self._backend_config))

    def is_model_loaded(self) -> bool:
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
        local_ok = self._inference_thread is not None and self._inference_thread.is_alive() and self._streaming_active
        remote_ok = self._remote_thread is not None and self._remote_thread.is_alive() and self._remote_streaming_active
        return local_ok or remote_ok

    def is_remote_streaming(self) -> bool:
        return self._remote_thread is not None and self._remote_thread.is_alive() and self._remote_streaming_active

    def is_local_streaming(self) -> bool:
        return self._inference_thread is not None and self._inference_thread.is_alive() and self._streaming_active

    def get_stream_mode(self) -> Optional[str]:
        if self.is_local_streaming():
            return "local"
        if self.is_remote_streaming():
            return "remote"
        return None

    def get_last_inference_age(self) -> Optional[float]:
        if self._last_inference_ts is None:
            return None
        return time.time() - self._last_inference_ts

    def get_last_remote_activity_age(self) -> Optional[float]:
        if self._remote_network_service is not None:
            get_age = getattr(self._remote_network_service, "get_last_stream_activity_age", None)
            if callable(get_age):
                age = get_age()
                if isinstance(age, (int, float)):
                    return float(age)
                return None

        if self._last_remote_activity_ts is None:
            return None
        return time.time() - self._last_remote_activity_ts

    def restart_streaming(self, camera_service, throttle_fps: Optional[float] = None) -> None:
        self.stop_streaming()
        self.start_streaming(camera_service, throttle_fps)

    def start_remote_streaming(self, camera_service, network_service: Any, throttle_fps: Optional[float] = None) -> None:
        """Server mode: send frames to the server, which runs detection and returns results."""
        self.start_frame_streaming(camera_service, network_service, fps=throttle_fps, mode="server")

    def start_frame_streaming(
        self,
        camera_service,
        network_service: Any,
        fps: Optional[float] = None,
        mode: str = "server",
    ) -> None:
        """Send JPEG frames to the server at up to `fps` frames per second.

        mode "server": the server detects pests in these frames (fallback when the robot
        cannot run the model). mode "monitor": the robot detects locally and these frames
        are the live view; local detections are sent along by the inference loop.
        The connection reconnects by itself; frames are only encoded while it is up.
        """
        if self._remote_thread and self._remote_thread.is_alive():
            return
        fps = float(fps if fps else getattr(configs, "STREAM_FPS", 5.0))

        self._remote_stop_event.clear()
        self._remote_streaming_active = True
        self._remote_network_service = network_service
        self._monitor_network_service = network_service if mode == "monitor" else None
        network_service.start_stream_channel(robot_id=getattr(configs, "ROBOT_ID", None), mode=mode)

        stop = self._remote_stop_event
        interval = 1.0 / fps

        def loop() -> None:
            logger.info("Frame streaming started (%s mode, %.1f fps)", mode, fps)
            next_due = 0.0
            last_frame_id: Optional[int] = None
            sent = encode_failures = 0
            while not stop.is_set():
                now = time.monotonic()
                if now < next_due:
                    stop.wait(next_due - now)
                    continue
                # Encoding costs CPU on a Pi 3; skip it while nobody can receive the frames.
                if not network_service.is_stream_connected():
                    stop.wait(0.5)
                    continue
                try:
                    res = _latest_frame(camera_service)
                except Exception:
                    logger.exception("Frame streaming could not read the camera")
                    stop.wait(0.5)
                    continue
                if res is None:
                    stop.wait(0.05)
                    continue
                frame, _, frame_id = _unpack_frame(res)
                if frame_id is not None and frame_id == last_frame_id:
                    stop.wait(0.01)  # no new frame yet
                    continue
                last_frame_id = frame_id
                next_due = now + interval

                frame_bytes = encode_jpeg(frame)
                if not frame_bytes:
                    encode_failures += 1
                    continue
                network_service.update_stream_frame(frame_bytes)
                self._last_remote_activity_ts = time.time()
                sent += 1
            logger.info("Frame streaming stopped (sent=%d encode_failures=%d)", sent, encode_failures)

        self._remote_thread = threading.Thread(target=loop, name="frame-stream", daemon=True)
        self._remote_thread.start()

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
                try:
                    from PIL import Image
                    pil_img = Image.open(str(sample_image)).convert('RGB')
                    img = np.array(pil_img)
                except Exception:
                    # PIL not available or failed; fall back to numpy load where possible
                    pil_img = None
                    img = None
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
                    self._stop_event.wait(0.05)
                    continue

                if len(res) == 3:
                    frame, ts, frame_id = res
                else:
                    frame, ts = res
                    frame_id = None

                if frame_id is not None and last_frame_id is not None and frame_id == last_frame_id:
                    self._stop_event.wait(0.01)  # no new frame yet
                    continue

                last_frame_id = frame_id
                frame_counter += 1
                if configs.FRAME_SKIP and frame_counter % configs.FRAME_SKIP != 0:
                    continue

                wait = min_frame_interval - (time.time() - last_infer)
                if min_frame_interval and wait > 0:
                    self._stop_event.wait(wait)
                    continue

                last_infer = time.time()
                self._last_inference_ts = last_infer
                try:
                    dets = self.run_inference_on_image(frame)
                except Exception:
                    logger.exception("Inference loop encountered an error")
                    dets = []
                self._dispatch(dets)
                self._send_monitor_detections(dets, frame)

            logger.info("Inference streaming stopped")

        self._inference_thread = threading.Thread(target=loop, name="inference-stream", daemon=True)
        self._inference_thread.start()

    def _send_monitor_detections(self, detections: List[Detection], frame: np.ndarray) -> None:
        """Monitor mode: send local detections and their frame to the server."""
        network = self._monitor_network_service
        if not detections or network is None or not network.is_stream_connected():
            return
        frame_bytes = encode_jpeg(frame)
        if frame_bytes:
            height, width = frame.shape[:2]
            network.send_stream_detections(detections, (width, height), frame_bytes)

    def stop_streaming(self, timeout: float = 2.0) -> None:
        # Stop local inference streaming
        try:
            self._stop_event.set()
            if self._inference_thread:
                self._inference_thread.join(timeout)
        finally:
            self._streaming_active = False
            self._reset_executor()

        # Stop remote streaming if active
        try:
            self._remote_stop_event.set()
            if self._remote_thread:
                self._remote_thread.join(timeout)
            if self._remote_network_service is not None:
                try:
                    self._remote_network_service.stop_stream_channel(timeout=timeout)
                except Exception:
                    logger.exception("Failed to stop websocket stream channel")
        finally:
            self._remote_streaming_active = False
            self._remote_network_service = None
            self._monitor_network_service = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    svc = ModelService()
    svc.test_mode()
