"""
Network connection service for remote control server.

Provides a clean abstraction over the WebSocket connection and telemetry
message flow without coupling the robot's startup logic to a specific
network library implementation.
"""

import logging
from typing import Any, Dict, Optional

import configs

logger = logging.getLogger(__name__)


class NetworkService:
    """Abstraction for remote server connectivity."""

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url or configs.SERVER_URL
        self._is_connected = False

    def connect_to_server(self) -> bool:
        """Attempt to connect to the remote control server."""
        logger.info(f"Connecting to control server at {self.server_url}")
        # TODO: Replace with actual WebSocket client implementation.
        self._is_connected = False
        return self._is_connected

    def disconnect(self) -> None:
        """Disconnect from the server gracefully."""
        if self._is_connected:
            logger.info("Disconnecting from control server")
        self._is_connected = False

    def is_connected(self) -> bool:
        """Return True when connected to the remote server."""
        return self._is_connected

    def send_message(self, payload: Dict[str, Any]) -> bool:
        """Send a telemetry message to the remote server."""
        if not self._is_connected:
            logger.warning("Cannot send telemetry: server is disconnected")
            return False
        logger.debug(f"Sending telemetry payload: {payload}")
        # TODO: Implement WebSocket send.
        return True
