"""Sprayer control: manual sprays and automatic sprays when a pest is detected.

A spray stops the wheels, runs the pump for a short burst, then stops the pump and
leaves the robot standing (driving resumes only when someone drives it again).
Automatic spraying is off at startup and is switched on from the dashboard.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import configs
from hardware.motors import MotorController
from services.drive_service import DriveController

logger = logging.getLogger(__name__)

SprayListener = Callable[[dict], None]


class SprayController:
    def __init__(
        self,
        motors_provider: Callable[[], Optional[MotorController]],
        drive: DriveController,
        on_spray: Optional[SprayListener] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._motors_provider = motors_provider
        self._drive = drive
        self._on_spray = on_spray
        self._clock = clock
        self._lock = threading.Lock()
        self._auto_enabled = False
        self._spraying = False
        self._last_auto_spray: Optional[float] = None
        self._cancel = threading.Event()

    @property
    def auto_enabled(self) -> bool:
        return self._auto_enabled

    @property
    def spraying(self) -> bool:
        return self._spraying

    def set_auto(self, enabled: bool) -> None:
        self._auto_enabled = bool(enabled)
        logger.info("Automatic spraying %s", "ENABLED" if self._auto_enabled else "disabled")

    def on_detections(self, detections: list) -> bool:
        """Spray automatically if enabled, confident enough and out of cooldown."""
        if not self._auto_enabled or not detections:
            return False
        best = max(detections, key=lambda d: float(d.get("confidence", 0.0)))
        confidence = float(best.get("confidence", 0.0))
        if confidence < configs.AUTO_SPRAY_MIN_CONFIDENCE:
            return False
        with self._lock:
            if self._last_auto_spray is not None and self._clock() - self._last_auto_spray < configs.SPRAY_COOLDOWN:
                return False
        return self.spray(configs.SPRAY_DURATION, trigger="auto", label=str(best.get("class")), confidence=confidence)

    def spray(
        self,
        duration: float,
        trigger: str = "manual",
        label: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """Start one spray burst in the background. False if busy or no pump."""
        if self._motors_provider() is None:
            logger.warning("Cannot spray: motor board (pump) not connected")
            return False
        duration = max(0.1, min(float(duration), configs.SPRAY_MAX_DURATION))
        with self._lock:
            if self._spraying:
                return False
            self._spraying = True
            if trigger == "auto":
                self._last_auto_spray = self._clock()
        self._cancel.clear()
        threading.Thread(
            target=self._run,
            args=(duration, trigger, label, confidence),
            name="spray",
            daemon=True,
        ).start()
        return True

    def cancel(self) -> None:
        """Stop a spray in progress (halt, shutdown, watchdog)."""
        self._cancel.set()

    def _run(self, duration: float, trigger: str, label: Optional[str], confidence: Optional[float]) -> None:
        motors = self._motors_provider()
        started = self._clock()
        pumped = False
        try:
            self._drive.block("spraying")
            if motors is not None and motors.set_pump(configs.PUMP_SPEED, configs.PUMP_DIRECTION):
                pumped = True
                logger.info("Spraying for %.1fs (%s%s)", duration, trigger, f", {label}" if label else "")
                self._cancel.wait(duration)
            else:
                logger.error("Pump did not start")
        except Exception:
            logger.exception("Spray failed")
        finally:
            # Always try to stop the pump; fall back to stopping everything.
            stopped = False
            if motors is not None:
                try:
                    stopped = motors.set_pump(0, configs.PUMP_DIRECTION)
                except Exception:
                    logger.exception("Pump stop failed")
                if not stopped:
                    try:
                        motors.stop()
                    except Exception:
                        logger.exception("Emergency stop after pump failure also failed")
            self._drive.unblock()
            self._spraying = False

        if pumped and self._on_spray is not None:
            try:
                self._on_spray({
                    "type": "spray",
                    "trigger": trigger,
                    "label": label,
                    "confidence": confidence,
                    "duration": round(self._clock() - started, 2),
                })
            except Exception:
                logger.exception("Spray listener failed")
