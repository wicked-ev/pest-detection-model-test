"""Frame streaming to the server and monitor-mode detections, with fake camera and network."""

import threading
import time

import numpy as np
import pytest

from services.model_service import ModelService, encode_jpeg


class FakeCamera:
    """Produces a new frame id every call, like a running camera."""

    def __init__(self):
        self.frame = np.full((48, 64, 3), 120, dtype=np.uint8)
        self.frame_id = 0
        self.lock = threading.Lock()

    def get_latest_metadata(self, copy=False):
        with self.lock:
            self.frame_id += 1
            return self.frame, time.time(), self.frame_id


class FakeNetwork:
    def __init__(self, connected=True):
        self.connected = connected
        self.frames = []
        self.detections = []
        self.channel_mode = None

    def start_stream_channel(self, robot_id=None, mode="server"):
        self.channel_mode = mode
        return False

    def stop_stream_channel(self, timeout=2.0):
        pass

    def is_stream_connected(self):
        return self.connected

    def update_stream_frame(self, frame_bytes):
        self.frames.append(frame_bytes)

    def send_stream_detections(self, detections, image_size, frame_bytes):
        self.detections.append((detections, image_size, frame_bytes))


@pytest.fixture
def service():
    svc = ModelService(num_threads=1)
    yield svc
    svc.stop_streaming()


def test_jpeg_encoding_produces_a_jpeg():
    data = encode_jpeg(np.zeros((8, 8, 3), dtype=np.uint8))
    assert data[:2] == b"\xff\xd8"


def test_frames_are_sent_at_the_requested_rate(service):
    network = FakeNetwork()
    service.start_frame_streaming(FakeCamera(), network, fps=10, mode="server")
    time.sleep(1.0)
    service.stop_streaming()
    assert network.channel_mode == "server"
    assert 6 <= len(network.frames) <= 13  # ~10 per second
    assert network.frames[0][:2] == b"\xff\xd8"


def test_no_frames_are_encoded_while_the_server_is_unreachable(service):
    network = FakeNetwork(connected=False)
    service.start_frame_streaming(FakeCamera(), network, fps=20)
    time.sleep(0.5)
    assert network.frames == []
    network.connected = True  # server comes back
    time.sleep(0.8)
    assert network.frames


def test_monitor_mode_sends_local_detections_with_their_frame(service):
    network = FakeNetwork()
    service.start_frame_streaming(FakeCamera(), network, fps=5, mode="monitor")
    assert network.channel_mode == "monitor"
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    detections = [{"class": "whitefly", "confidence": 0.9, "bbox": (1, 2, 3, 4)}]

    service._send_monitor_detections(detections, frame)
    service._send_monitor_detections([], frame)  # nothing found: nothing extra sent

    [(sent, size, jpeg)] = network.detections
    assert sent == detections and size == (64, 48) and jpeg[:2] == b"\xff\xd8"


def test_server_mode_does_not_send_local_detections(service):
    network = FakeNetwork()
    service.start_frame_streaming(FakeCamera(), network, fps=5, mode="server")
    service._send_monitor_detections([{"class": "x", "confidence": 1.0, "bbox": (0, 0, 1, 1)}], np.zeros((4, 4, 3), np.uint8))
    assert network.detections == []


def test_model_uses_all_cores_by_default():
    import configs

    assert ModelService()._backend_config.num_threads == configs.MODEL_NUM_THREADS


def test_runtime_check_reflects_installed_runtimes(monkeypatch):
    from services.inference_backends import factory

    svc = ModelService(num_threads=1)
    monkeypatch.setattr(factory.InferenceBackendFactory, "create_backends", classmethod(lambda cls, config: []))
    assert svc.has_runtime() is False
    monkeypatch.setattr(factory.InferenceBackendFactory, "create_backends", classmethod(lambda cls, config: ["tflite"]))
    assert svc.has_runtime() is True


def test_no_model_download_without_a_runtime(monkeypatch):
    """On a Pi with no runtime, auto mode goes to the server without fetching the model."""
    import app as robot_app

    robot = robot_app.RobotApplication.__new__(robot_app.RobotApplication)
    downloads = []
    robot.model_service = ModelService(num_threads=1)
    robot.asset_manager = type("Assets", (), {"ensure_assets": lambda self, stop_event=None: downloads.append(1) or True})()
    robot._shutdown_requested = threading.Event()
    monkeypatch.setattr(robot.model_service, "has_runtime", lambda: False)
    assert robot._start_local_detection() is False
    assert downloads == []
