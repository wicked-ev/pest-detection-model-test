"""Network connection service for the robot client.

This module preserves the local provisioning socket behavior while adding the
websocket control and stream channels used by the server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import socket
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from socket import socket as socket_obj
from typing import Any, Callable, Dict, Optional
from urllib.parse import quote_plus

import requests
try:
    import websockets  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - dependency is provided at runtime
    websockets = None  # type: ignore[assignment]

import configs

logger = logging.getLogger(__name__)


class NetworkService:
    """Abstraction for local provisioning sockets and remote websocket links."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        server_connection_attempts: int = 25,
        connection_retry_delay: float = 2.0,
        heartbeat_interval: float = 8.0,
    ):
        self.server_url = server_url or configs.SERVER_URL
        self.server_host = configs.SERVER_HOST
        self.server_port = configs.SERVER_PORT
        self.robot_id = getattr(configs, "ROBOT_ID", socket.gethostname())
        self.server_connection_attempts = server_connection_attempts
        self.connection_retry_delay = connection_retry_delay
        self.heartbeat_interval = heartbeat_interval

        self.client: Optional[socket_obj] = None
        self._server_socket: Optional[socket_obj] = None
        self._is_connected = False
        self._is_server_on = False

        self._control_stop_event = threading.Event()
        self._control_shutdown_event: Optional[threading.Event] = None
        self._control_thread: Optional[threading.Thread] = None
        self._control_connected = False
        self._control_connected_event = threading.Event()
        self._control_inbox: "queue.Queue[str]" = queue.Queue()
        self._control_outbox: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._control_state_provider: Optional[Callable[[], str]] = None
        self._control_robot_id = self.robot_id
        self._control_heartbeat_interval = heartbeat_interval

        self._stream_stop_event = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_connected = False
        self._stream_connected_event = threading.Event()
        self._stream_robot_id = self.robot_id
        self._stream_model_name: Optional[str] = None
        # "server": the server runs detection on our frames and sends results back.
        # "monitor": we run detection ourselves and send frames plus results for monitoring.
        self._stream_mode = "server"
        # Monitor mode: (detections message, the frame it belongs to), sent before live frames.
        # Bounded, so a slow link drops the oldest results instead of growing memory.
        self._stream_detections: "deque[tuple[str, bytes]]" = deque(maxlen=8)
        self._stream_detections_lock = threading.Lock()
        # Set by a session once it connects; the retry loop uses it to reset its backoff.
        self._control_session_connected = False
        self._stream_session_connected = False
        self._stream_detection_callback: Optional[Callable[[list], None]] = None
        self._stream_latest_frame_lock = threading.Lock()
        self._stream_latest_frame_bytes: Optional[bytes] = None
        self._stream_latest_frame_seq = 0
        self._stream_last_sent_seq = 0
        self._stream_frame_update_ts: Optional[float] = None
        self._stream_last_sent_ts: Optional[float] = None
        self._callback_executor = ThreadPoolExecutor(max_workers=2)

    def _http_base_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    def _ws_base_url(self) -> str:
        return f"ws://{self.server_host}:{self.server_port}"

    def _ws_connect(self, uri: str) -> Any:
        """Open a server websocket, sending the shared robot key as a header when configured."""
        options: Dict[str, Any] = {"open_timeout": 10.0, "close_timeout": 5.0, "ping_interval": None}
        robot_key = getattr(configs, "ROBOT_KEY", "")
        if robot_key:
            headers = {"X-Robot-Key": robot_key}
            # websockets >= 14 names this argument additional_headers; older releases use extra_headers.
            major = int(str(getattr(websockets, "__version__", "14")).split(".")[0])
            options["additional_headers" if major >= 14 else "extra_headers"] = headers
        return websockets.connect(uri, **options)

    def _state_value(self) -> str:
        if self._control_state_provider is None:
            return "unknown"
        try:
            value = self._control_state_provider()
        except Exception:
            logger.exception("State provider failed")
            return "unknown"
        return str(value)

    def _select_model_name(self) -> Optional[str]:
        if self._stream_model_name:
            return self._stream_model_name

        if websockets is None:
            logger.warning("websockets package is not available; stream model discovery will be skipped")
            return None

        try:
            response = requests.get(f"{self._http_base_url()}/api/models", timeout=5.0)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Could not discover server models from /api/models: %s", exc)
            return None

        if isinstance(payload, list):
            candidates: list[Any] = payload
        elif isinstance(payload, dict):
            candidates = []
            for key in ("models", "available_models", "items", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    candidates = value
                    break
            if not candidates:
                candidates = [payload]
        else:
            candidates = [payload]

        for item in candidates:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                for key in ("name", "model", "id", "slug", "format"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

        return None

    def wait_for_client(self, host: Optional[str] = None, port: Optional[int] = None, stop_event: Optional[threading.Event] = None) -> bool:
        logger.info("Waiting for a client connection on the local socket server")

        host = host or self.server_host
        port = port or self.server_port

        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((host, port))
            self._server_socket.listen(1)
            # Set short timeout so we can poll for shutdown
            self._server_socket.settimeout(1.0)

            while True:
                if stop_event is not None and stop_event.is_set():
                    logger.info("wait_for_client cancelled by stop event")
                    try:
                        self._server_socket.close()
                    except Exception:
                        pass
                    self._is_connected = False
                    return False

                try:
                    client, addr = self._server_socket.accept()
                    logger.info("Client connected with address %s", addr)
                    self.client = client
                    self._is_connected = True
                    self._is_server_on = True
                    # remove timeout on accepted socket
                    try:
                        self.client.settimeout(None)
                    except Exception:
                        pass
                    return True
                except socket.timeout:
                    continue
        except OSError as exc:
            logger.error("Error accepting client connection on local server %s:%s: %s", host, port, exc)
            self._is_connected = False
            return False

    def has_network_connection(self, host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> bool:
        """Check if internet/network access exists."""
        try:
            socket.setdefaulttimeout(timeout)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((host, port))
            return True
        except OSError:
            return False

    def set_stream_detection_callback(self, callback: Optional[Callable[[list], None]]) -> None:
        self._stream_detection_callback = callback

    def connect_to_server(
        self,
        stop_event: Optional[threading.Event] = None,
        robot_id: Optional[str] = None,
        state_provider: Optional[Callable[[], str]] = None,
        wait_timeout: Optional[float] = None,
    ) -> bool:
        """Start the control connection and wait up to wait_timeout for it.

        The connection runs in a background thread that keeps reconnecting until
        disconnect() is called, so a server that is down at startup is picked up later.
        Returns whether it is connected now.
        """
        if not (self._control_thread and self._control_thread.is_alive()):
            self._control_stop_event.clear()
            self._control_shutdown_event = stop_event
            self._control_connected_event.clear()
            self._control_robot_id = robot_id or self.robot_id
            self._control_state_provider = state_provider
            self._control_thread = threading.Thread(target=self._control_worker, name="control-ws", daemon=True)
            self._control_thread.start()

        if wait_timeout is None:
            wait_timeout = getattr(configs, "SERVER_CONNECT_WAIT", 15.0)
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            if self._control_connected_event.wait(timeout=0.1):
                return True
            if stop_event is not None and stop_event.is_set():
                return False
        logger.warning(
            "Server %s not reachable yet; continuing and retrying in the background",
            self._ws_base_url(),
        )
        return False

    def start_stream_channel(
        self,
        model_name: Optional[str] = None,
        robot_id: Optional[str] = None,
        mode: str = "server",
        wait_timeout: float = 0.0,
    ) -> bool:
        """Start sending frames to the server; it keeps reconnecting in the background.

        mode "server": the server detects pests in our frames and returns the results.
        mode "monitor": we detect locally; frames and send_stream_detections() results
        are sent so the server can show and record them.
        Returns whether the stream is connected within wait_timeout (0 = don't wait).
        """
        if self._stream_thread and self._stream_thread.is_alive():
            if self._stream_mode == mode:
                return self._stream_connected
            self.stop_stream_channel()

        self._stream_stop_event = threading.Event()
        self._stream_connected_event.clear()
        self._stream_robot_id = robot_id or self.robot_id
        self._stream_model_name = model_name
        self._stream_mode = mode
        self._stream_latest_frame_seq = 0
        self._stream_last_sent_seq = 0
        self._stream_latest_frame_bytes = None
        self._stream_frame_update_ts = None
        self._stream_last_sent_ts = None
        with self._stream_detections_lock:
            self._stream_detections.clear()

        self._stream_thread = threading.Thread(target=self._stream_worker, name="stream-ws", daemon=True)
        self._stream_thread.start()
        return self._stream_connected_event.wait(timeout=wait_timeout) if wait_timeout > 0 else False

    def get_stream_mode(self) -> Optional[str]:
        if self._stream_thread and self._stream_thread.is_alive():
            return self._stream_mode
        return None

    def send_stream_detections(self, detections: list, image_size: tuple, frame_bytes: bytes) -> None:
        """Monitor mode: queue our own detections together with the frame they came from."""
        if not frame_bytes:
            return
        message = json.dumps({
            "type": "detections",
            "image_size": [int(image_size[0]), int(image_size[1])],
            "detections": [
                {
                    "class": str(d.get("class")),
                    "confidence": float(d.get("confidence", 0.0)),
                    "bbox": [float(v) for v in d.get("bbox", ())],
                }
                for d in detections
            ],
        })
        with self._stream_detections_lock:
            self._stream_detections.append((message, frame_bytes))

    def stop_stream_channel(self, timeout: float = 2.0) -> None:
        self._stream_stop_event.set()
        if self._stream_thread:
            self._stream_thread.join(timeout)
        self._stream_thread = None
        self._stream_connected = False
        self._stream_connected_event.clear()

    def update_stream_frame(self, frame_bytes: bytes) -> None:
        if not frame_bytes:
            return
        with self._stream_latest_frame_lock:
            self._stream_latest_frame_bytes = bytes(frame_bytes)
            self._stream_latest_frame_seq += 1
            self._stream_frame_update_ts = time.time()

    def _control_worker(self) -> None:
        base_delay = max(1.0, float(self.connection_retry_delay))
        backoff = base_delay
        while not self._control_should_stop():
            self._control_session_connected = False
            try:
                asyncio.run(self._run_control_session())
            except Exception as exc:
                self._log_session_error("Control", exc)

            self._control_connected = False
            self._control_connected_event.clear()
            self._is_connected = self.client is not None
            if self._control_should_stop():
                break

            # After a working session reconnect quickly; while the server stays down, back off.
            backoff = base_delay if self._control_session_connected else min(30.0, backoff * 2.0)
            self._control_stop_event.wait(backoff)

    def _control_should_stop(self) -> bool:
        return self._control_stop_event.is_set() or (
            self._control_shutdown_event is not None and self._control_shutdown_event.is_set()
        )

    async def _run_control_session(self) -> None:
        if websockets is None:
            raise RuntimeError("websockets package is not installed")

        uri = f"{self._ws_base_url()}/ws/control/{quote_plus(self._control_robot_id)}"
        logger.info("Connecting control websocket: %s", uri)

        async with self._ws_connect(uri) as websocket:
            self._control_connected = True
            self._is_connected = True
            self._control_connected_event.set()
            self._control_session_connected = True
            logger.info("Control websocket connected")

            recv_task = asyncio.create_task(self._control_receive_loop(websocket))
            send_task = asyncio.create_task(self._control_send_loop(websocket))
            heartbeat_task = asyncio.create_task(self._control_heartbeat_loop(websocket))

            done, pending = await asyncio.wait(
                {recv_task, send_task, heartbeat_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            for task in done:
                task.result()

    async def _control_receive_loop(self, websocket: Any) -> None:
        while not self._control_should_stop():
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if message is None:
                continue

            if isinstance(message, bytes):
                try:
                    message = message.decode("utf-8")
                except UnicodeDecodeError:
                    logger.warning("Ignoring non-text control websocket message")
                    continue

            self._control_inbox.put(str(message))

    async def _control_send_loop(self, websocket: Any) -> None:
        while not self._control_should_stop():
            try:
                payload = self._control_outbox.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue

            try:
                await websocket.send(json.dumps(payload))
            except Exception:
                logger.exception("Failed to send control websocket message")
                raise

    async def _control_heartbeat_loop(self, websocket: Any) -> None:
        while not self._control_should_stop():
            await asyncio.sleep(self._control_heartbeat_interval)
            if self._control_should_stop():
                break

            heartbeat = {
                "type": "heartbeat",
                "state": self._state_value(),
                "timestamp": time.time(),
            }
            try:
                logger.debug("Sending heartbeat for %s", self._control_robot_id)
                await websocket.send(json.dumps(heartbeat))
            except Exception:
                logger.exception("Failed to send control websocket heartbeat")
                raise

    def _stream_worker(self) -> None:
        base_delay = max(1.0, float(self.connection_retry_delay))
        backoff = base_delay
        while not self._stream_stop_event.is_set():
            self._stream_session_connected = False
            try:
                asyncio.run(self._run_stream_session())
            except Exception as exc:
                self._log_session_error("Stream", exc)

            self._stream_connected = False
            self._stream_connected_event.clear()
            if self._stream_stop_event.is_set():
                break

            backoff = base_delay if self._stream_session_connected else min(30.0, backoff * 2.0)
            self._stream_stop_event.wait(backoff)

    @staticmethod
    def _log_session_error(kind: str, exc: Exception) -> None:
        text = str(exc)
        if "403" in text:
            logger.error(
                "%s connection rejected by the server (HTTP 403): "
                "ROBOT_KEY in .env must match the server's ROBOT_KEY",
                kind,
            )
        else:
            logger.warning("%s websocket ended: %s", kind, text or type(exc).__name__)

    async def _run_stream_session(self) -> None:
        if websockets is None:
            raise RuntimeError("websockets package is not installed")

        monitor = self._stream_mode == "monitor"
        # In monitor mode the server runs no model, so there is no model to pick.
        model_name = None if monitor else self._select_model_name()
        if monitor:
            query = "?mode=monitor"
        else:
            query = f"?model={quote_plus(model_name)}" if model_name else ""
        uri = f"{self._ws_base_url()}/ws/stream/{quote_plus(self._stream_robot_id)}{query}"
        logger.info("Connecting stream websocket (%s mode): %s", self._stream_mode, uri)

        async with self._ws_connect(uri) as websocket:
            self._stream_connected = True
            self._is_connected = True
            self._stream_connected_event.set()
            self._stream_session_connected = True
            logger.info("Stream websocket connected")

            if not monitor and not model_name:
                handshake_model = self._select_model_name()
                if handshake_model:
                    await websocket.send(json.dumps({"model": handshake_model}))
                    logger.info("Sent stream handshake with model=%s", handshake_model)

            recv_task = asyncio.create_task(self._stream_receive_loop(websocket))
            send_task = asyncio.create_task(self._stream_send_loop(websocket))

            done, pending = await asyncio.wait(
                {recv_task, send_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            for task in done:
                task.result()

    async def _stream_receive_loop(self, websocket: Any) -> None:
        while not self._stream_stop_event.is_set():
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if message is None:
                continue

            if isinstance(message, bytes):
                try:
                    message = message.decode("utf-8")
                except UnicodeDecodeError:
                    logger.warning("Ignoring non-text stream websocket message")
                    continue

            try:
                payload = json.loads(str(message))
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON stream websocket message")
                continue

            detections = payload.get("detections") if isinstance(payload, dict) else None
            if detections is None:
                detections = payload

            if isinstance(detections, dict):
                nested_detections = detections.get("detections")
                if isinstance(nested_detections, list):
                    detections = nested_detections
                else:
                    detections = []
            elif not isinstance(detections, list):
                detections = []

            callback = self._stream_detection_callback
            if callback is None:
                continue

            try:
                self._callback_executor.submit(callback, detections)
                self._stream_last_sent_ts = time.time()
            except Exception:
                logger.exception("Failed to dispatch stream detection callback")

    async def _stream_send_loop(self, websocket: Any) -> None:
        while not self._stream_stop_event.is_set():
            # Monitor mode: our detections go first, each followed by its own frame,
            # so the server pairs them correctly (one websocket keeps messages in order).
            with self._stream_detections_lock:
                pending = self._stream_detections.popleft() if self._stream_detections else None
            if pending is not None:
                message, detection_frame = pending
                await websocket.send(message)
                await websocket.send(detection_frame)
                self._stream_last_sent_ts = time.time()
                continue

            with self._stream_latest_frame_lock:
                frame_seq = self._stream_latest_frame_seq
                frame_bytes = self._stream_latest_frame_bytes

            if frame_bytes is None or frame_seq == self._stream_last_sent_seq:
                await asyncio.sleep(0.03)
                continue

            try:
                await websocket.send(frame_bytes)
                self._stream_last_sent_seq = frame_seq
                self._stream_last_sent_ts = time.time()
            except Exception:
                logger.exception("Failed to send stream frame")
                raise

    def disconnect(self) -> bool:
        """Disconnect from the server gracefully."""
        success = True

        self._control_stop_event.set()
        self._control_shutdown_event = None
        self.stop_stream_channel()

        if self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None
        self._control_connected = False
        self._control_connected_event.clear()

        if self.client is not None:
            try:
                self.client.close()
                logger.info("Disconnected from control server")
            except OSError as exc:
                logger.error("Error disconnecting from control server: %s", exc)
                success = False
            finally:
                self.client = None

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError as exc:
                logger.error("Error closing local server socket: %s", exc)
                success = False
            finally:
                self._server_socket = None

        self._is_connected = False
        return success

    def is_remote_model_loaded(self) -> bool:
        # TODO: unfinished; the empty body was a syntax error that stopped this module from importing.
        raise NotImplementedError

    def is_connected(self) -> bool:
        """Return True when connected to any active server transport."""
        return self._is_connected or self.client is not None or self._control_connected or self._stream_connected

    def is_control_connected(self) -> bool:
        return self._control_connected

    def is_stream_connected(self) -> bool:
        return self._stream_connected

    def get_last_stream_activity_age(self) -> Optional[float]:
        if self._stream_last_sent_ts is None:
            return None
        return time.time() - self._stream_last_sent_ts

    def send_message(self, payload: Dict[str, Any]) -> bool:
        """Send a JSON message to the active control transport."""
        if self._control_thread and self._control_thread.is_alive():
            self._control_outbox.put(payload)
            return True

        if not self.is_connected() or self.client is None:
            logger.warning("Cannot send telemetry: server is disconnected")
            return False

        logger.debug("Sending telemetry payload over local socket: %s", payload)

        try:
            message = json.dumps(payload)
            self.client.sendall(message.encode("utf-8"))
            return True
        except OSError as exc:
            logger.error("Error sending message: %s", exc)
            return False

    def receive_message(self) -> Optional[str]:
        return self.receive_message_with_timeout()

    def receive_message_with_timeout(self, timeout: float = 10.0) -> Optional[str]:
        """Receive a queued control message or a provisioning socket payload."""
        if self._control_thread and self._control_thread.is_alive():
            try:
                return self._control_inbox.get(timeout=timeout)
            except queue.Empty:
                return None

        if not self.is_connected() or self.client is None:
            logger.warning("Cannot receive telemetry: server is disconnected")
            return None

        logger.debug("Receiving message with timeout=%s...", timeout)

        end_time = time.time() + timeout
        buffer = b""

        while time.time() < end_time:
            try:
                remaining = max(0.1, end_time - time.time())
                try:
                    self.client.settimeout(remaining)
                except OSError:
                    pass

                chunk = self.client.recv(4096)
                if not chunk:
                    logger.warning("No data received (client closed connection)")
                    break
                buffer += chunk

                try:
                    text = buffer.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                try:
                    json.loads(text)
                    return text
                except json.JSONDecodeError:
                    continue

            except socket.timeout:
                continue
            except OSError as exc:
                logger.error("Error receiving data: %s", exc)
                return None

        if buffer:
            try:
                return buffer.decode("utf-8")
            except UnicodeDecodeError:
                logger.error("Failed to decode received bytes")
                return None

        logger.debug("No data received before timeout")
        return None
