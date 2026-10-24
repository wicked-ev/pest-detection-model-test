"""The interactive hardware test tool, driven with scripted answers and a fake Arduino."""

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "hardware_test.py"


class FakeArduino:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def connect(self):
        return True

    def set_motor(self, motor_id, speed, direction):
        self.calls.append(("motor", motor_id, speed, direction))
        return True

    def set_pump(self, speed, direction):
        self.calls.append(("pump", speed, direction))
        return True

    def stop_all(self):
        self.calls.append(("stop_all",))
        return True

    def disconnect(self):
        self.calls.append(("disconnect",))


@pytest.fixture
def tool(monkeypatch):
    spec = importlib.util.spec_from_file_location("hardware_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    instances = []

    def make(*args, **kwargs):
        instances.append(FakeArduino())
        return instances[-1]

    monkeypatch.setattr(module, "ArduinoConnection", make)
    return module, instances


def run(tool, monkeypatch, capsys, answers):
    module, instances = tool
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    code = module.main()
    return code, capsys.readouterr().out, instances[0]


def test_wiring_and_pump_direction_are_worked_out(tool, monkeypatch, capsys):
    answers = [
        "y",
        "", "fl", "y",   # motor 0: front left, direction 1 = forward
        "", "fr", "n",   # motor 1: front right, direction 1 = backward (mirrored)
        "", "rl", "y",   # motor 2
        "", "rr", "n",   # motor 3
        "", "n",         # pump with direction 1: no spray
        "", "y",         # pump with direction 0: sprays
    ]
    code, out, arduino = run(tool, monkeypatch, capsys, answers)
    assert code == 0
    for line in ["LEFT_MOTORS=0,2", "RIGHT_MOTORS=1,3", "LEFT_FORWARD_DIRECTION=1",
                 "RIGHT_FORWARD_DIRECTION=0", "PUMP_DIRECTION=0"]:
        assert line in out
    # Every pulse is followed by a stop, and it ends stopped and disconnected.
    assert arduino.calls[-2:] == [("stop_all",), ("disconnect",)]
    assert all(arduino.calls[i + 1] == ("stop_all",) for i, c in enumerate(arduino.calls) if c[0] in ("motor", "pump"))


def test_inconsistent_wiring_is_flagged(tool, monkeypatch, capsys):
    answers = ["y", "", "fl", "y", "", "fr", "y", "", "rl", "n", "", "rr", "y", "", "y"]
    code, out, _ = run(tool, monkeypatch, capsys, answers)
    assert code == 2 and "swap the two wires" in out


def test_refuses_to_run_unless_the_wheels_are_lifted(tool, monkeypatch, capsys):
    code, out = run_without_arduino(tool, monkeypatch, capsys)
    assert code == 1 and "Lift the robot" in out


def run_without_arduino(tool, monkeypatch, capsys):
    module, instances = tool
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    code = module.main()
    assert instances == []  # never touched the motors
    return code, capsys.readouterr().out
