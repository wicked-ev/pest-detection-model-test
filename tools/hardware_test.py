"""Interactive hardware test: find the motor wiring and pump direction.

Run on the Pi with the WHEELS OFF THE GROUND and the robot program stopped
(only one program can use the serial port):

    .venv/bin/python tools/hardware_test.py

Each step runs one motor (or the pump) briefly at low speed, asks what happened,
and stops everything. At the end it prints the .env lines to use.
Press Ctrl+C at any time: everything is stopped.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import configs  # noqa: E402
from hardware.arduino import ArduinoConnection  # noqa: E402

TEST_SPEED = 120  # PWM out of 255: enough to see, gentle on the gearbox
TEST_SECONDS = 1.0
WHEELS = {"fl": "front left", "fr": "front right", "rl": "rear left", "rr": "rear right"}


def ask(question: str, options: tuple) -> str:
    while True:
        answer = input(f"{question} [{'/'.join(options)}]: ").strip().lower()
        if answer in options:
            return answer
        print(f"  Please answer one of: {', '.join(options)}")


def pulse_motor(arduino: ArduinoConnection, motor_id: int) -> None:
    arduino.set_motor(motor_id, TEST_SPEED, 1)
    time.sleep(TEST_SECONDS)
    arduino.stop_all()


def pulse_pump(arduino: ArduinoConnection, direction: int) -> None:
    arduino.set_pump(configs.PUMP_SPEED, direction)
    time.sleep(0.5)
    arduino.stop_all()


def main() -> int:
    print(__doc__)
    if ask("Are the wheels off the ground and the robot program stopped?", ("y", "n")) != "y":
        print("Lift the robot so the wheels spin freely, stop app.py, then run this again.")
        return 1

    arduino = ArduinoConnection(
        port=configs.ARDUINO_PORT,
        baudrate=configs.ARDUINO_BAUDRATE,
        timeout=configs.ARDUINO_TIMEOUT,
        write_timeout=configs.ARDUINO_WRITE_TIMEOUT,
    )
    if not arduino.connect():
        print(f"Could not connect to the Arduino on {configs.ARDUINO_PORT} (see ARDUINO_PORT in .env).")
        return 1

    left, right = [], []
    left_dirs, right_dirs = set(), set()
    pump_direction = None
    try:
        arduino.stop_all()
        for motor_id in range(4):
            input(f"\nMotor {motor_id}: press Enter to run it for {TEST_SECONDS:.0f}s (direction value 1)...")
            pulse_motor(arduino, motor_id)
            wheel = ask("Which wheel turned? (none = nothing moved)", tuple(WHEELS) + ("none",))
            if wheel == "none":
                print("  Skipped. Check the wiring of this motor driver.")
                continue
            forward = ask("Did it turn the way that drives the robot FORWARD?", ("y", "n")) == "y"
            value_for_forward = 1 if forward else 0
            if wheel in ("fl", "rl"):
                left.append(motor_id)
                left_dirs.add(value_for_forward)
            else:
                right.append(motor_id)
                right_dirs.add(value_for_forward)

        for direction in (1, 0):
            input(f"\nPump: press Enter to run it for 0.5s (direction value {direction})...")
            pulse_pump(arduino, direction)
            if ask("Did it spray?", ("y", "n")) == "y":
                pump_direction = direction
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        arduino.stop_all()
        arduino.disconnect()
        print("All motors and the pump are stopped.")

    print("\n================ Results ================")
    problems = []
    if len(left_dirs) > 1 or len(right_dirs) > 1:
        problems.append("Motors on the same side need different direction values; swap the two wires of one of them.")
    if not left or not right:
        problems.append("Not every side has a working motor; check the wiring before driving.")
    if pump_direction is None:
        problems.append("The pump did not spray in either direction; check its wiring and the tank.")
    if left and right:
        print("Add these lines to .env:\n")
        print(f"LEFT_MOTORS={','.join(map(str, left))}")
        print(f"RIGHT_MOTORS={','.join(map(str, right))}")
        print(f"LEFT_FORWARD_DIRECTION={min(left_dirs) if left_dirs else 1}")
        print(f"RIGHT_FORWARD_DIRECTION={min(right_dirs) if right_dirs else 1}")
    if pump_direction is not None:
        print(f"PUMP_DIRECTION={pump_direction}")
    for problem in problems:
        print(f"\nWARNING: {problem}")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
