"""Manual driving, the automatic stop, spraying, motor wiring and Arduino command serialization."""

import threading
import time

import pytest

import configs
import hardware.arduino as arduino_module
from hardware.motors import MovementDirection
from services.drive_service import DriveController
from services.spray_service import SprayController


class FakeMotors:
    """Records what the robot would do to its motors and pump."""

    def __init__(self, pump_stop_fails=False):
        self.log = []
        self.pump_stop_fails = pump_stop_fails
        self.lock = threading.Lock()

    def move(self, direction, speed_scale=1.0):
        with self.lock:
            self.log.append(("move", direction.value, round(speed_scale, 2)))
        return True

    def set_pump(self, speed, direction):
        with self.lock:
            self.log.append(("pump", speed, direction))
        return not (speed == 0 and self.pump_stop_fails)

    def stop(self):
        with self.lock:
            self.log.append(("stop_all",))
        return True

    def events(self, kind):
        with self.lock:
            return [entry for entry in self.log if entry[0] == kind]


def wait_for(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.fixture
def motors():
    return FakeMotors()


@pytest.fixture
def drive(motors):
    controller = DriveController(lambda: motors, command_timeout=0.3)
    yield controller
    controller.close()


@pytest.fixture
def spray_settings(monkeypatch):
    monkeypatch.setattr(configs, "SPRAY_DURATION", 0.2)
    monkeypatch.setattr(configs, "SPRAY_COOLDOWN", 10.0)
    monkeypatch.setattr(configs, "AUTO_SPRAY_MIN_CONFIDENCE", 0.7)
    monkeypatch.setattr(configs, "PUMP_SPEED", 255)
    monkeypatch.setattr(configs, "PUMP_DIRECTION", 1)


def pest(confidence, label="whitefly"):
    return [{"class": label, "confidence": confidence, "bbox": (0, 0, 10, 10)}]


# --- Driving -----------------------------------------------------------------------------


def test_repeated_moves_are_sent_to_the_motors_once(drive, motors):
    for _ in range(5):
        assert drive.drive(MovementDirection.FORWARD, 0.5)
    assert motors.events("move") == [("move", "forward", 0.5)]
    assert drive.direction == "forward"


def test_changing_direction_or_speed_is_sent(drive, motors):
    drive.drive(MovementDirection.FORWARD, 1.0)
    drive.drive(MovementDirection.LEFT, 1.0)
    drive.drive(MovementDirection.LEFT, 0.4)
    assert [e[1:] for e in motors.events("move")] == [("forward", 1.0), ("left", 1.0), ("left", 0.4)]


def test_robot_stops_by_itself_when_drive_commands_stop_arriving(drive, motors):
    drive.drive(MovementDirection.FORWARD)
    assert wait_for(lambda: drive.direction is None, timeout=1.0)
    assert motors.events("move")[-1][1] == "stop"


def test_holding_a_key_keeps_the_robot_moving(drive, motors):
    for _ in range(6):  # 0.6 s of repeats, longer than the 0.3 s timeout
        drive.drive(MovementDirection.FORWARD)
        time.sleep(0.1)
    assert drive.direction == "forward"
    assert ("move", "stop", 1.0) not in motors.events("move")


def test_driving_without_motors_is_refused():
    controller = DriveController(lambda: None, command_timeout=0.3)
    try:
        assert controller.drive(MovementDirection.FORWARD) is False
    finally:
        controller.close()


def test_blocked_drive_refuses_moves_until_unblocked(drive, motors):
    drive.block("spraying")
    assert drive.drive(MovementDirection.FORWARD) is False
    drive.unblock()
    assert drive.drive(MovementDirection.FORWARD) is True


# --- Spraying ----------------------------------------------------------------------------


def make_spray(motors, drive, events=None):
    return SprayController(lambda: motors, drive, on_spray=(events.append if events is not None else None))


def test_auto_spray_is_off_until_switched_on(motors, drive, spray_settings):
    spray = make_spray(motors, drive)
    assert spray.on_detections(pest(0.95)) is False
    assert motors.events("pump") == []


def test_confident_detection_stops_wheels_then_sprays_a_burst(motors, drive, spray_settings):
    events = []
    spray = make_spray(motors, drive, events)
    spray.set_auto(True)
    drive.drive(MovementDirection.FORWARD)

    assert spray.on_detections(pest(0.9)) is True
    assert wait_for(lambda: events, timeout=2.0)

    log = motors.log
    stop_index = log.index(("move", "stop", 1.0))
    pump_on = log.index(("pump", 255, 1))
    pump_off = log.index(("pump", 0, 1))
    assert stop_index < pump_on < pump_off  # wheels stop before the pump starts
    assert events[0]["trigger"] == "auto" and events[0]["label"] == "whitefly"
    assert 0.15 <= events[0]["duration"] < 1.0
    assert drive.direction is None  # stays stopped afterwards


def test_low_confidence_does_not_spray(motors, drive, spray_settings):
    spray = make_spray(motors, drive)
    spray.set_auto(True)
    assert spray.on_detections(pest(0.5)) is False


def test_cooldown_prevents_back_to_back_auto_sprays(motors, drive, spray_settings):
    events = []
    spray = make_spray(motors, drive, events)
    spray.set_auto(True)
    assert spray.on_detections(pest(0.9))
    assert wait_for(lambda: events)
    assert spray.on_detections(pest(0.9)) is False  # within SPRAY_COOLDOWN
    # A manual spray is still allowed during the cooldown.
    assert spray.spray(0.1, trigger="manual")
    assert wait_for(lambda: len(events) == 2)


def test_driving_is_refused_while_spraying(motors, drive, spray_settings):
    spray = make_spray(motors, drive)
    spray.spray(0.5)
    assert wait_for(lambda: spray.spraying)
    assert drive.drive(MovementDirection.FORWARD) is False
    assert wait_for(lambda: not spray.spraying, timeout=2.0)
    assert drive.drive(MovementDirection.FORWARD) is True


def test_cancel_ends_a_spray_early(motors, drive, spray_settings):
    events = []
    spray = make_spray(motors, drive, events)
    spray.spray(3.0)
    assert wait_for(lambda: spray.spraying)
    spray.cancel()
    assert wait_for(lambda: events, timeout=1.0)
    assert events[0]["duration"] < 1.0


def test_spray_length_is_capped(motors, drive, spray_settings, monkeypatch):
    monkeypatch.setattr(configs, "SPRAY_MAX_DURATION", 0.2)
    events = []
    spray = make_spray(motors, drive, events)
    spray.spray(60)
    assert wait_for(lambda: events, timeout=2.0)
    assert events[0]["duration"] < 0.5


def test_failed_pump_stop_falls_back_to_stopping_everything(drive, spray_settings):
    motors = FakeMotors(pump_stop_fails=True)
    spray = SprayController(lambda: motors, drive)
    spray.spray(0.1)
    assert wait_for(lambda: motors.events("stop_all"), timeout=2.0)


def test_spraying_without_a_pump_is_refused(drive, spray_settings):
    assert SprayController(lambda: None, drive).spray(1.0) is False


# --- Motor wiring --------------------------------------------------------------------------


@pytest.fixture
def recording_arduino(monkeypatch):
    connection = arduino_module.ArduinoConnection(port="/dev/null")
    sent = []
    monkeypatch.setattr(connection, "set_motor", lambda motor_id, speed, direction: sent.append((motor_id, speed, direction)) or True)
    monkeypatch.setattr(connection, "stop_all", lambda: sent.append("STOP_ALL") or True)
    return connection, sent


def test_forward_uses_the_configured_wiring(recording_arduino, monkeypatch):
    connection, sent = recording_arduino
    monkeypatch.setattr(configs, "LEFT_MOTORS", (0, 2))
    monkeypatch.setattr(configs, "RIGHT_MOTORS", (1, 3))
    monkeypatch.setattr(configs, "LEFT_FORWARD_DIRECTION", 1)
    monkeypatch.setattr(configs, "RIGHT_FORWARD_DIRECTION", 0)  # right side mounted mirrored
    monkeypatch.setattr(configs, "DRIVE_SPEED", 200)
    assert connection.move("forward", 0.5)
    assert sent == [(0, 100, 1), (2, 100, 1), (1, 100, 0), (3, 100, 0)]


def test_turning_left_spins_the_sides_in_opposite_directions(recording_arduino, monkeypatch):
    connection, sent = recording_arduino
    monkeypatch.setattr(configs, "LEFT_MOTORS", (0,))
    monkeypatch.setattr(configs, "RIGHT_MOTORS", (1,))
    monkeypatch.setattr(configs, "LEFT_FORWARD_DIRECTION", 1)
    monkeypatch.setattr(configs, "RIGHT_FORWARD_DIRECTION", 1)
    monkeypatch.setattr(configs, "TURN_SPEED", 175)
    connection.move("left")
    assert sent == [(0, 175, 0), (1, 175, 1)]


def test_drive_stop_only_stops_the_wheels(recording_arduino):
    connection, sent = recording_arduino
    connection.move("stop")
    assert "STOP_ALL" not in sent  # STOP_ALL would also stop a spray
    assert all(speed == 0 for _, speed, _ in sent)


def test_arduino_commands_do_not_overlap(monkeypatch):
    """A drive command and a health-check ping must not take each other's reply."""
    connection = arduino_module.ArduinoConnection(port="/dev/null")
    connection._is_connected = True
    active, overlaps = [0], []

    def write(message):
        active[0] += 1
        overlaps.append(active[0])

    def wait(expected, timeout):
        time.sleep(0.02)
        active[0] -= 1
        return "OK"

    monkeypatch.setattr(connection, "_write_message", write)
    monkeypatch.setattr(connection, "_wait_for_response", wait)
    threads = [threading.Thread(target=connection.set_motor, args=(i % 4, 100, 1)) for i in range(8)]
    threads += [threading.Thread(target=connection.set_pump, args=(0, 1)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max(overlaps) == 1


# --- Simulated Arduino (ARDUINO_PORT=simulated) ------------------------------------------------


@pytest.fixture
def simulated_board():
    connection = arduino_module.ArduinoConnection(port="simulated")
    assert connection.connect()
    yield connection
    connection.disconnect()


def test_simulated_board_answers_like_the_firmware(simulated_board):
    from hardware.motors import MotorController

    motors = MotorController(simulated_board)
    assert motors.initialize()  # PING / CHECK / STOP_ALL / GET_STATUS all answered
    assert simulated_board.ping()


def test_drive_and_spray_through_the_simulated_board(simulated_board, spray_settings):
    from hardware.motors import MotorController

    motors = MotorController(simulated_board)
    assert motors.initialize()
    sim = simulated_board._serial
    drive = DriveController(lambda: motors, command_timeout=0.3)
    try:
        assert drive.drive(MovementDirection.FORWARD, 0.5)
        assert all(speed == 100 for speed, _ in sim.motors)
        assert wait_for(lambda: all(speed == 0 for speed, _ in sim.motors), timeout=1.5)  # automatic stop

        events = []
        spray = SprayController(lambda: motors, drive, on_spray=events.append)
        spray.spray(0.3)
        assert wait_for(lambda: sim.pump[0] == 255, timeout=1.0)
        assert wait_for(lambda: events, timeout=2.0)
        assert sim.pump[0] == 0
    finally:
        drive.close()
