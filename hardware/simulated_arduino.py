"""A software stand-in for the Arduino motor board (set ARDUINO_PORT=simulated).

It speaks the same serial protocol as hardware/arduino_controller.ino and answers like
it, including the 6-second failsafe. Use it to run and test the robot software, or to
demo it, without the robot hardware. Motor and pump changes are logged.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

logger = logging.getLogger(__name__)

FAILSAFE_TIMEOUT = 6.0


class SimulatedSerial:
    """Minimal pyserial-like object: write() commands, read() the replies."""

    def __init__(self) -> None:
        self.is_open = True
        self.motors = [[0, 0] for _ in range(4)]  # [speed, direction]
        self.pump = [0, 0]
        self._replies: "queue.Queue[bytes]" = queue.Queue()
        self._pending = b""
        self._input = ""
        self._last_command = time.monotonic()
        self._lock = threading.Lock()
        self._failsafe = threading.Thread(target=self._watch_failsafe, name="sim-arduino", daemon=True)
        self._failsafe.start()

    # --- pyserial interface used by ArduinoConnection ---

    def write(self, data: bytes) -> int:
        self._input += data.decode("utf-8", errors="ignore")
        while "\n" in self._input:
            line, self._input = self._input.split("\n", 1)
            line = line.strip()
            if line:
                with self._lock:
                    self._last_command = time.monotonic()
                    self._reply(self._handle(line))
        return len(data)

    def flush(self) -> None:
        pass

    def read(self, size: int = 1) -> bytes:
        if not self._pending:
            try:
                self._pending = self._replies.get(timeout=0.1)
            except queue.Empty:
                return b""
        chunk, self._pending = self._pending[:size], self._pending[size:]
        return chunk

    def reset_input_buffer(self) -> None:
        self._pending = b""
        while not self._replies.empty():
            self._replies.get_nowait()

    def reset_output_buffer(self) -> None:
        self._input = ""

    def close(self) -> None:
        self.is_open = False

    # --- firmware behaviour ---

    def _reply(self, text: str) -> None:
        self._replies.put((text + "\n").encode())

    def _handle(self, line: str) -> str:
        parts = line.split()
        command = parts[0].upper()
        if command == "PING":
            return "PONG"
        if command == "CHECK":
            return "MOTORS_OK"
        if command in ("STOP", "STOP_ALL"):
            self._stop_everything("STOP_ALL")
            return "OK"
        if command == "GET_STATUS":
            values = [v for speed_dir in self.motors for v in speed_dir] + self.pump
            return "STATUS:" + ",".join(str(v) for v in values)
        if command == "SET_MOTOR":
            try:
                motor_id, speed, direction = int(parts[1]), int(parts[2]), int(parts[3])
            except (IndexError, ValueError):
                return "ERROR:INVALID_MOTOR_PARAMS"
            if not 0 <= motor_id <= 3:
                return "ERROR:INVALID_MOTOR_ID"
            if not (0 <= speed <= 255 and direction in (0, 1)):
                return "ERROR:INVALID_MOTOR_PARAMS"
            if self.motors[motor_id] != [speed, direction]:
                self.motors[motor_id] = [speed, direction]
                logger.info("[simulated Arduino] motor %d: speed %d, direction %d", motor_id, speed, direction)
            return "OK"
        if command == "SET_PUMP":
            try:
                speed, direction = int(parts[1]), int(parts[2])
            except (IndexError, ValueError):
                return "ERROR:INVALID_PUMP_PARAMS"
            if not (0 <= speed <= 255 and direction in (0, 1)):
                return "ERROR:INVALID_PUMP_PARAMS"
            self.pump = [speed, direction]
            logger.info("[simulated Arduino] pump %s", "ON" if speed else "off")
            return "OK"
        return "ERROR:UNKNOWN_COMMAND"

    def _stop_everything(self, reason: str) -> None:
        if any(speed for speed, _ in self.motors) or self.pump[0]:
            logger.info("[simulated Arduino] %s: all motors and pump stopped", reason)
        self.motors = [[0, d] for _, d in self.motors]
        self.pump = [0, self.pump[1]]

    def _watch_failsafe(self) -> None:
        while self.is_open:
            time.sleep(0.2)
            with self._lock:
                running = any(speed for speed, _ in self.motors) or self.pump[0]
                if running and time.monotonic() - self._last_command > FAILSAFE_TIMEOUT:
                    self._stop_everything("failsafe")
                    self._reply("ERROR:FAILSAFE_STOP")
