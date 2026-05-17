"""
Network connection service for remote control server.

Provides a clean abstraction over the WebSocket connection and telemetry
message flow without coupling the robot's startup logic to a specific
network library implementation.
"""

import json
import logging
import socket
import time
from socket import socket as socket_obj
from typing import Any, Dict, Optional

import configs

logger = logging.getLogger(__name__)


class NetworkService:
    """Abstraction for remote server connectivity."""

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url or configs.SERVER_URL
        self.server_host = configs.SERVER_HOST
        self.server_port = configs.SERVER_PORT
        self.client: Optional[socket_obj] = None
        self._server_socket: Optional[socket_obj] = None
        self._is_connected = False
        self._is_server_on = False

    def wait_for_client(self, host: Optional[str] = None, port: Optional[int] = None) -> bool:
        logger.info("Waiting for a client connection on the local socket server")

        host = host or self.server_host
        port = port or self.server_port

        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((host, port))
            self._server_socket.listen(1)

            client, addr = self._server_socket.accept()
            logger.info(f"Client connected with address {addr}")
            self.client = client
            self._is_connected = True
            self._is_server_on = True
            return True
        except OSError as exc:
            logger.error(
                f"Error accepting client connection on local server {host}:{port}: {exc}"
            )
            self._is_connected = False
            return False
        
    def connect_to_server(self) -> bool:
        """Attempt to connect to the remote control server."""
        logger.info(f"Connecting to control server at {self.server_url}")

        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(10.0)
            client_socket.connect((self.server_host, self.server_port))

            self.client = client_socket
            self._is_connected = True
            logger.info(
                f"Connected to remote control server at {self.server_host}:{self.server_port}"
            )
            return True
        except (OSError, socket.error) as exc:
            logger.error(
                f"Error connecting to remote control server {self.server_host}:{self.server_port}: {exc}"
            )
            self._is_connected = False
            self.client = None
            return False


    def disconnect(self) -> bool:
        """Disconnect from the server gracefully."""
        success = True

        if self.client is not None:
            try:
                self.client.close()
                logger.info("Disconnected from control server")
            except OSError as exc:
                logger.error(f"Error disconnecting from control server: {exc}")
                success = False
            finally:
                self.client = None

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError as exc:
                logger.error(f"Error closing local server socket: {exc}")
                success = False
            finally:
                self._server_socket = None

        self._is_connected = False
        return success

    def is_connected(self) -> bool:
        """Return True when connected to the remote server."""
        return self._is_connected and self.client is not None

    def send_message(self, payload: Dict[str, Any]) -> bool:
        """Send a telemetry message to the remote server."""
        if not self.is_connected():
            logger.warning("Cannot send telemetry: server is disconnected")
            return False

        logger.debug(f"Sending telemetry payload: {payload}")
        assert self.client is not None

        try:
            message = json.dumps(payload)
            self.client.sendall(message.encode("utf-8"))
            return True
        except OSError as exc:
            logger.error(f"Error sending message: {exc}")
            return False


    def receive_message(self) -> Optional[str]:
        return self.receive_message_with_timeout()

    def receive_message_with_timeout(self, timeout: float = 10.0) -> Optional[str]:
        """Receive data from the connected client and return a UTF-8 string.

        This method will keep reading from the socket until either a valid
        JSON document can be parsed from the accumulated bytes, the timeout
        expires, or the client closes the connection. This makes receiving
        JSON payloads robust against partial TCP reads.
        """
        if not self.is_connected():
            logger.warning("Cannot receive telemetry: server is disconnected")
            return None

        logger.debug("Receiving message with timeout=%s...", timeout)
        assert self.client is not None

        end_time = time.time() + timeout
        buffer = b""

        while time.time() < end_time:
            try:
                remaining = max(0.1, end_time - time.time())
                try:
                    self.client.settimeout(remaining)
                except OSError:
                    # Some sockets may not support settimeout repeatedly; ignore
                    pass

                chunk = self.client.recv(4096)
                if not chunk:
                    logger.warning("No data received (client closed connection)")
                    break
                buffer += chunk

                try:
                    text = buffer.decode("utf-8")
                except UnicodeDecodeError:
                    # Wait for more bytes to complete multi-byte sequences
                    continue

                # Try to parse JSON to detect whether we've received a complete message
                try:
                    json.loads(text)
                    return text
                except json.JSONDecodeError:
                    # Not yet complete; keep reading
                    continue

            except socket.timeout:
                # no data during this short interval, loop until overall timeout
                continue
            except OSError as exc:
                logger.error(f"Error receiving data: {exc}")
                return None

        # Timeout or client closed. If we have any buffered bytes, attempt to decode
        if buffer:
            try:
                return buffer.decode("utf-8")
            except UnicodeDecodeError:
                logger.error("Failed to decode received bytes")
                return None

        logger.debug("No data received before timeout")
        return None
        