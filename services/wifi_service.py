"""
WiFi management and provisioning service.

Provides an architectural layer for managing saved WiFi credentials,
connecting to networks, and entering hotspot/provisioning mode.

This module is intentionally lightweight and platform-agnostic. It
captures the expected responsibilities while keeping implementation
separate from the OS-specific provisioning logic.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time
import configs
from services.network_service import NetworkService

logger = logging.getLogger(__name__)


@dataclass
class WiFiCredentials:
    ssid: str
    password: str


class WiFiManager:
    """Manages WiFi credentials, connection attempts, and hotspot mode."""

    def __init__(self, credential_path: Optional[Path] = None):
        self.credentials_path = (
            Path(credential_path) if credential_path is not None else configs.WIFI_CREDENTIALS_PATH
        )
        self.credentials_path.parent.mkdir(parents=True, exist_ok=True)

    def load_saved_credentials(self) -> Optional[list[WiFiCredentials]]:
        """Load saved WiFi credentials from persistent storage."""
        if not self.credentials_path.exists():
            logger.debug("No saved WiFi credentials found")
            return None

        wifiCredentials: list[WiFiCredentials] = [] 
        try:
            data = json.loads(self.credentials_path.read_text())

            if not isinstance(data, list):
                logger.warning("WiFi credentials file must contain a list")
                return []

            for item in data:
                ssid = item.get("ssid", "").strip()
                password = item.get("password", "").strip()

                if not ssid or not password:
                    logger.warning(f"Skipping invalid WiFi credential: {item}")
                    continue

                wifiCredentials.append(
                    WiFiCredentials(ssid=ssid, password=password)
                )

                return wifiCredentials

        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"Failed to read saved WiFi credentials: {exc}")
            return []

    def save_credentials(self, credentials: WiFiCredentials) -> bool:
        """Persist WiFi credentials for future startup attempts."""
        try:
            self.credentials_path.write_text(
                json.dumps({"ssid": credentials.ssid, "password": credentials.password}, indent=2)
            )
            logger.info(f"Saved WiFi credentials to {self.credentials_path}")
            return True
        except OSError as exc:
            logger.error(f"Failed to save WiFi credentials: {exc}")
            return False

    def validate_credentials(self, credentials: list[WiFiCredentials]) -> bool:
        """Validate credentials shape before using them."""
        #TODO: add support for empty passwords
        for credential in credentials:
            if not credential.ssid:
                logger.warning("WIFI SSID is empty")
                return False
            if not credential.password:
                logger.warning("WiFi password is empty")
                return False                        
        return True

    def connect_saved_networks(self) -> bool:
        """Attempt to connect to previously saved WiFi credentials."""
        credentials = self.load_saved_credentials()
        if credentials is None:
            logger.info("No WiFi credentials available for saved network connection")
            return False

        return self.connect(credentials)

    def connect(self, credentials: list[WiFiCredentials]) -> bool:
        """Attempt to connect to a WiFi network.
        """
        if not self.validate_credentials(credentials):
            return False

        for credential in credentials:
            logger.info(f"Attempting WiFi connection to SSID '{credential.ssid}'")
            try:
                subprocess.run([
                    "nmcli",
                    "device",
                    "wifi",
                    "connect",
                    credential.ssid,
                    credential.password       
                ], check=True)
                return True
            except subprocess.CalledProcessError:
                logger.error(f"Error connecting to SSID: ${credential.ssid}")
                return False
        return False

    def start_hotspot(self) -> bool:
        """Enable hotspot/access point mode for provisioning."""
        logger.info("Starting hotspot mode for WiFi provisioning")
        try:
            subprocess.run([
                "nmcli",
                "device",
                "wifi",
                "hotspot",
                "ifname",
                "wlan0",
                "ssid",
                "MyPiSetup",
                "password",
                "raspberry123"
            ])
            return True
        except subprocess.CalledProcessError:
            logger.error(f"Error opening hotspot with SSID: MyPiSetup")
            return False

    def stop_hotspot(self) -> bool:
        """Attempt to stop hotspot mode.

        The exact command used to stop a hotspot is platform-dependent. This
        implementation uses `nmcli` to disconnect the wireless device. It is
        a best-effort helper and may need adjustment for different systems.
        """
        logger.info("Stopping hotspot mode")
        try:
            subprocess.run([
                "nmcli",
                "device",
                "disconnect",
                "wlan0"
            ], check=True)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error(f"Error stopping hotspot: {exc}")
            return False
        except OSError as exc:
            logger.error(f"Failed to stop hotspot (OS error): {exc}")
            return False
        

    def write_wpa_supplicant(self, credentials: WiFiCredentials) -> bool:
        """Write credentials to wpa_supplicant configuration.

        This is a convenience helper; actual system activation must be
        handled separately by a provisioning flow.
        """
        wpa_path = configs.WPA_SUPPLICANT_PATH
        content = (
            "ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev\n"
            "update_config=1\n"
            "country=US\n\n"
            "network={\n"
            f"    ssid=\"{credentials.ssid}\"\n"
            f"    psk=\"{credentials.password}\"\n"
            "    key_mgmt=WPA-PSK\n"
            "}\n"
        )

        try:
            wpa_path.parent.mkdir(parents=True, exist_ok=True)
            wpa_path.write_text(content)
            logger.info(f"Wrote WiFi credentials to {wpa_path}")
            return True
        except OSError as exc:
            logger.error(f"Failed to write wpa_supplicant file: {exc}")
            return False


class HotspotProvisioningService:
    """Service responsible for WiFi provisioning and credential exchange."""

    def __init__(self, wifi_manager: Optional[WiFiManager] = None):
        self.wifi_manager = wifi_manager or WiFiManager()

    def enter_provisioning_mode(self) -> bool:
        """Prepare the robot so a user can connect and provide WiFi credentials.
        
        This method:
        1. Starts a WiFi hotspot for provisioning
        2. Waits for a client to connect via the network service
        3. Receives WiFi credentials (JSON: {"ssid": "...", "password": "..."})
        4. Validates and saves the credentials
        5. Sends confirmation back to client
        6. Cleans up and returns
        """
        logger.info("Entering WiFi provisioning mode")
        
        # Step 1: Start hotspot
        if not self.wifi_manager.start_hotspot():
            logger.error("Failed to start hotspot")
            return False
        
        logger.info("Hotspot started, waiting for client connection")
        
        # Step 2: Set up network service and accept provisioning clients.
        network_service = NetworkService()
        if not network_service.wait_for_client():
            logger.error("Failed to accept provisioning client connection")
            return False

        logger.info("Provisioning client connected; entering message loop")

        try:
            # Keep the provisioning session open so the user can retry.
            while True:
                # Receive a message (allow extra time for uploads)
                message = network_service.receive_message_with_timeout(timeout=30.0)

                if not message:
                    logger.info("No message received from client; waiting for reconnection or next message")
                    # Try to accept a new client if the previous closed connection
                    network_service.disconnect()
                    if not network_service.wait_for_client():
                        logger.info("No client reconnected; continuing to wait")
                        time.sleep(1)
                        continue
                    else:
                        logger.info("Client reconnected; continuing provisioning loop")
                        continue

                # Parse JSON payload
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError as exc:
                    logger.error(f"Invalid JSON received: {exc}")
                    network_service.send_message({"status": "error", "message": "Invalid JSON"})
                    continue

                # Handle control commands from the client
                cmd = payload.get("command") if isinstance(payload, dict) else None
                if isinstance(cmd, str):
                    cmd = cmd.strip().lower()
                if cmd:
                    if cmd == "use_saved":
                        logger.info("Client requested to try saved WiFi credentials")
                        saved = self.wifi_manager.load_saved_credentials()
                        if not saved:
                            network_service.send_message({"status": "error", "message": "No saved credentials"})
                            continue

                        if self.wifi_manager.connect(saved):
                            logger.info("Connected using saved credentials")
                            network_service.send_message({"status": "success", "message": "Connected using saved credentials"})
                            try:
                                self.wifi_manager.stop_hotspot()
                            except Exception as exc:
                                logger.warning(f"Stopping hotspot failed: {exc}")
                            return True
                        else:
                            network_service.send_message({"status": "error", "message": "Failed to connect using saved credentials"})
                            continue

                    if cmd in ("exit", "cancel", "quit"):
                        logger.info("Client requested to exit provisioning")
                        network_service.send_message({"status": "ok", "message": "Provisioning cancelled"})
                        return False

                    # Unknown command
                    network_service.send_message({"status": "error", "message": "Unknown command"})
                    continue

                # Otherwise treat payload as credentials
                if not isinstance(payload, dict):
                    network_service.send_message({"status": "error", "message": "Expected JSON object with credentials or command"})
                    continue

                ssid = payload.get("ssid", "").strip()
                password = payload.get("password", "").strip()

                if not ssid or not password:
                    network_service.send_message({"status": "error", "message": "Missing ssid or password"})
                    continue

                logger.info(f"Received credentials attempt for SSID: {ssid}")

                # Try connecting with provided credentials (do not overwrite saved until success)
                if not self.wifi_manager.connect([WiFiCredentials(ssid, password)]):
                    logger.warning("Connection attempt with provided credentials failed")
                    network_service.send_message({"status": "error", "message": "Connection failed"})
                    continue

                # Connection successful: persist credentials and stop hotspot
                if not self.accept_credentials(ssid, password):
                    logger.warning("Connected but failed to persist credentials")
                    network_service.send_message({"status": "error", "message": "Connected but failed to save credentials"})
                    # Even if saving failed, we've connected; still stop hotspot
                    try:
                        self.wifi_manager.stop_hotspot()
                    except Exception as exc:
                        logger.warning(f"Stopping hotspot failed: {exc}")
                    return True

                network_service.send_message({"status": "success", "message": "Connected and credentials saved"})
                try:
                    self.wifi_manager.stop_hotspot()
                except Exception as exc:
                    logger.warning(f"Stopping hotspot failed: {exc}")
                return True

        except Exception as exc:
            logger.error(f"Unexpected error during provisioning: {exc}")
            try:
                network_service.send_message({"status": "error", "message": "Unexpected error"})
            except Exception:
                pass
            return False

        finally:
            network_service.disconnect()
            logger.info("Provisioning mode ended, disconnected from client")

    def accept_credentials(self, ssid: str, password: str) -> bool:
        """Validate and save new WiFi credentials provided by the user."""
        credentials = WiFiCredentials(ssid=ssid, password=password)
        if not self.wifi_manager.validate_credentials([credentials]):
            return False
        return self.wifi_manager.save_credentials(credentials)
