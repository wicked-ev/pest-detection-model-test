from __future__ import annotations

import logging
import threading
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)

CleanupCallback = Callable[[], None]


class LifecycleManager:
    """Manage startup resource registration and reverse-order cleanup."""

    def __init__(self):
        self._cleanup_stack: List[Tuple[str, CleanupCallback]] = []
        self._lock = threading.Lock()

    def register(self, name: str, cleanup_fn: CleanupCallback) -> None:
        """Register a cleanup callback for a resource initialized during startup."""
        with self._lock:
            self._cleanup_stack.append((name, cleanup_fn))
        logger.debug("Registered lifecycle cleanup callback: %s", name)

    def cleanup_all(self) -> None:
        """Execute all registered cleanup callbacks in reverse order."""
        with self._lock:
            cleanup_stack = list(reversed(self._cleanup_stack))
            self._cleanup_stack.clear()

        if not cleanup_stack:
            logger.debug("No lifecycle cleanup callbacks registered")
            return

        logger.info("Starting lifecycle cleanup for %d resources", len(cleanup_stack))
        for name, cleanup_fn in cleanup_stack:
            try:
                logger.info("Cleaning up resource: %s", name)
                cleanup_fn()
            except Exception as exc:
                logger.exception("Cleanup failed for %s: %s", name, exc)
        logger.info("Lifecycle cleanup complete")
