"""Settings read from .env / the environment."""

import importlib
import socket

import pytest

import configs


@pytest.fixture
def reload_configs(monkeypatch):
    def reload(**env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return importlib.reload(configs)

    yield reload
    monkeypatch.undo()
    importlib.reload(configs)  # back to the real settings for other tests


def test_empty_robot_id_falls_back_to_the_hostname(reload_configs):
    # An empty ID put "/ws/control/" in the URL, which the server rejects with 403.
    assert reload_configs(ROBOT_ID="").ROBOT_ID == socket.gethostname()
    assert reload_configs(ROBOT_ID="   ").ROBOT_ID == socket.gethostname()


def test_robot_id_is_used_when_set(reload_configs):
    assert reload_configs(ROBOT_ID=" pest-robot-1 ").ROBOT_ID == "pest-robot-1"


def test_invalid_numbers_fall_back_to_defaults(reload_configs):
    settings = reload_configs(STREAM_FPS="fast", CAMERA_FPS="", DETECTION_MODE="sometimes")
    assert settings.STREAM_FPS == 5.0
    assert settings.CAMERA_FPS == 15
    assert settings.DETECTION_MODE == "auto"
