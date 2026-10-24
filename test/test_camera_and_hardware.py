"""Camera colour order, the synthetic camera, backend selection, and missing pyserial."""

import json
import time

import numpy as np

import hardware.arduino as arduino
from services.camera_backends.factory import CameraBackendConfig, CameraBackendFactory
from services.camera_backends.picamera2_camera import Picamera2Backend
from services.camera_backends.synthetic_camera import SyntheticCameraBackend


class FakePicamera2:
    def __init__(self, frame):
        self.frame = frame

    def capture_array(self):
        return self.frame


def test_picamera2_frames_are_converted_to_rgb():
    # Picamera2's "RGB888" format delivers pixels in B, G, R order.
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue channel first
    backend = Picamera2Backend()
    backend._camera = FakePicamera2(bgr)
    frame = backend.read()
    assert frame[0, 0].tolist() == [0, 0, 255]  # R, G, B: blue is now last
    assert frame.flags["C_CONTIGUOUS"]


def test_synthetic_camera_gives_moving_frames_at_its_frame_rate():
    camera = SyntheticCameraBackend(width=160, height=120, fps=20)
    camera.open()
    started = time.monotonic()
    frames = [camera.read() for _ in range(5)]
    elapsed = time.monotonic() - started
    assert frames[0].shape == (120, 160, 3) and frames[0].dtype == np.uint8
    assert not np.array_equal(frames[0], frames[4])  # the square moves
    assert elapsed >= 0.15  # paced at ~20 fps instead of spinning


def test_cached_camera_backend_is_ignored_when_not_in_the_preference_list(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"camera_backend": "opencv"}))
    config = CameraBackendConfig(width=64, height=48, fps=10, backend_preference=("synthetic",))
    backend = CameraBackendFactory.select_best_backend(config, cache_path=cache)
    assert backend.name == "synthetic"
    backend.close()


def test_arduino_without_pyserial_fails_cleanly(monkeypatch):
    monkeypatch.setattr(arduino, "serial", None)
    monkeypatch.setattr(arduino, "_SERIAL_IMPORT_ERROR", ImportError("pyserial missing"))
    connection = arduino.ArduinoConnection(port="/dev/null")
    assert connection.connect() is False
