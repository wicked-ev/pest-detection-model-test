from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

EmergencyCallback = Callable[[str], None]


class EmergencyStopService:
    """Global emergency stop service and notifier."""

    def __init__(self):
        self._event = threading.Event()
        self._reason: Optional[str] = None
        self._callbacks: List[EmergencyCallback] = []
        self._lock = threading.Lock()

    def engage(self, reason: str) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()
            logger.critical(f"EMERGENCY STOP engaged: {reason}")
            for callback in self._callbacks:
                try:
                    callback(reason)
                except Exception as exc:
                    logger.error(f"Emergency callback failed: {exc}")

    def clear(self) -> None:
        with self._lock:
            self._event.clear()
            self._reason = None
            logger.info("Emergency stop cleared")

    def is_engaged(self) -> bool:
        return self._event.is_set()

    def reason(self) -> Optional[str]:
        return self._reason

    def register_callback(self, callback: EmergencyCallback) -> None:
        with self._lock:
            self._callbacks.append(callback)
            logger.debug(f"Registered emergency callback: {callback.__name__}")
