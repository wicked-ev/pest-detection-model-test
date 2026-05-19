from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

WatchdogCheck = Callable[[], Tuple[bool, str]]
WatchdogRecovery = Callable[[], bool]
WatchdogFailureCallback = Callable[[str, str], None]


@dataclass
class WatchdogTarget:
    name: str
    check_fn: WatchdogCheck
    recovery_fn: Optional[WatchdogRecovery] = None
    critical: bool = True
    failure_count: int = 0
    last_message: str = ""


class WatchdogService:
    """Runtime watchdog for thread and service health."""

    def __init__(self, interval: float = 1.0, failure_threshold: int = 2):
        self._interval = interval
        self._failure_threshold = max(1, failure_threshold)
        self._targets: Dict[str, WatchdogTarget] = {}
        self._callbacks: List[WatchdogFailureCallback] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def register_target(
        self,
        name: str,
        check_fn: WatchdogCheck,
        recovery_fn: Optional[WatchdogRecovery] = None,
        critical: bool = True,
    ) -> None:
        with self._lock:
            self._targets[name] = WatchdogTarget(
                name=name,
                check_fn=check_fn,
                recovery_fn=recovery_fn,
                critical=critical,
            )
            logger.debug(f"Registered watchdog target: {name}")

    def register_failure_callback(self, callback: WatchdogFailureCallback) -> None:
        with self._lock:
            self._callbacks.append(callback)
            logger.debug(f"Registered watchdog failure callback: {callback.__name__}")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="WatchdogMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("Watchdog service started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        logger.info("Watchdog service stopped")

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                targets = list(self._targets.values())
            for target in targets:
                try:
                    ok, message = target.check_fn()
                except Exception as exc:
                    ok = False
                    message = f"Exception during watchdog check: {exc}"

                target.last_message = message
                if ok:
                    target.failure_count = 0
                    logger.debug(f"Watchdog OK: {target.name}")
                    continue

                target.failure_count += 1
                logger.warning(
                    f"Watchdog target failure [{target.name}] "
                    f"(count={target.failure_count}): {message}"
                )

                if target.recovery_fn:
                    try:
                        recovered = target.recovery_fn()
                        if recovered:
                            logger.info(f"Watchdog recovered target: {target.name}")
                            target.failure_count = 0
                            continue
                    except Exception as exc:
                        logger.warning(
                            f"Recovery for {target.name} failed: {exc}"
                        )

                if target.critical and target.failure_count >= self._failure_threshold:
                    for callback in self._callbacks:
                        try:
                            callback(target.name, message)
                        except Exception as exc:
                            logger.error(
                                f"Watchdog failure callback failed for {target.name}: {exc}"
                            )

            time.sleep(self._interval)
 