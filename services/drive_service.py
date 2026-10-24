"""Manual driving with a dead-man's switch.

The dashboard sends "move" repeatedly while a drive key is held. Each command keeps the
robot moving for DRIVE_COMMAND_TIMEOUT seconds; if none arrives in time (key released,
tab closed, Wi-Fi lost) the robot stops by itself. The Arduino's own 6 s failsafe does not
cover this, because the Pi's health-check pings keep resetting it.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import configs
from hardware.motors import MotorController, MovementDirection

logger = logging.getLogger(__name__)


class DriveController:
    def __init__(
        self,
        motors_provider: Callable[[], Optional[MotorController]],
        command_timeout: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._motors_provider = motors_provider
        self._timeout = command_timeout if command_timeout is not None else configs.DRIVE_COMMAND_TIMEOUT
        self._clock = clock
        self._lock = threading.RLock()
        self._direction: Optional[MovementDirection] = None  # None or STOP = standing still
        self._speed_scale = 1.0
        self._deadline = 0.0
        self._blocked_reason: Optional[str] = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="drive-deadman", daemon=True)
        self._thread.start()

    @property
    def direction(self) -> Optional[str]:
        direction = self._direction
        return direction.value if direction and direction != MovementDirection.STOP else None

    def has_motors(self) -> bool:
        return self._motors_provider() is not None

    def drive(self, direction: MovementDirection, speed_scale: float = 1.0) -> bool:
        """Move (or keep moving) for another DRIVE_COMMAND_TIMEOUT seconds."""
        if direction == MovementDirection.STOP:
            return self.stop("stop requested")
        motors = self._motors_provider()
        if motors is None:
            logger.warning("Ignoring move %s: motors not connected", direction.value)
            return False
        with self._lock:
            if self._blocked_reason:
                logger.info("Ignoring move %s: %s", direction.value, self._blocked_reason)
                return False
            self._deadline = self._clock() + self._timeout
            speed_scale = max(0.1, min(1.0, float(speed_scale)))
            if direction == self._direction and abs(speed_scale - self._speed_scale) < 0.01:
                return True  # already moving this way; the repeat only extends the deadline
            if motors.move(direction, speed_scale):
                self._direction, self._speed_scale = direction, speed_scale
                return True
            self._direction = None
            return False

    def stop(self, reason: str = "") -> bool:
        with self._lock:
            was_moving = self.direction is not None
            self._direction = MovementDirection.STOP
            motors = self._motors_provider()
            if motors is None:
                return True
            if was_moving and reason:
                logger.info("Stopping wheels: %s", reason)
            return motors.move(MovementDirection.STOP)

    def block(self, reason: str) -> None:
        """Stop and refuse drive commands until unblock() (e.g. while spraying)."""
        with self._lock:
            self._blocked_reason = reason
            self.stop(reason)

    def unblock(self) -> None:
        with self._lock:
            self._blocked_reason = None

    def close(self) -> None:
        self._stop_event.set()
        self.stop("shutting down")

    def _watch(self) -> None:
        while not self._stop_event.wait(0.05):
            with self._lock:
                expired = self.direction is not None and self._clock() > self._deadline
            if expired:
                self.stop(f"no drive command for {self._timeout:.1f}s")
