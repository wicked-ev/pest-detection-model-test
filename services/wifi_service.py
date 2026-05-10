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

import configs

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

    def load_saved_credentials(self) -> Optional[WiFiCredentials]:
        """Load saved WiFi credentials from persistent storage."""
        if not self.credentials_path.exists():
            logger.debug("No saved WiFi credentials found")
            return None

        try:
            data = json.loads(self.credentials_path.read_text())
            ssid = data.get("ssid", "").strip()
            password = data.get("password", "").strip()
            if not ssid or not password:
                logger.warning("Saved WiFi credentials are invalid or incomplete")
                return None
            return WiFiCredentials(ssid=ssid, password=password)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"Failed to read saved WiFi credentials: {exc}")
            return None

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

    def validate_credentials(self, credentials: WiFiCredentials) -> bool:
        """Validate credentials shape before using them."""
        if not credentials.ssid:
            logger.warning("WiFi SSID is empty")
            return False
        if not credentials.password:
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

    def connect(self, credentials: WiFiCredentials) -> bool:
        """Attempt to connect to a WiFi network.
        """
        if not self.validate_credentials(credentials):
            return False

        logger.info(f"Attempting WiFi connection to SSID '{credentials.ssid}'")
        try:
            subprocess.run([
                "nmcli",
                "device",
                "wifi",
                "connect",
                credentials.ssid,
                credentials.password       
            ], check=True)
        except subprocess.CalledProcessError:
            logger.error(f"Error connecting to SSID: ${credentials.ssid}")
            return False
        
        return False

    def start_hotspot(self) -> bool:
        """Enable hotspot/access point mode for provisioning."""
        logger.info("Starting hotspot mode for WiFi provisioning")
        # TODO: Implement platform-specific hotspot activation.
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
        """Prepare the robot so a user can connect and provide WiFi credentials."""
        logger.info("Entering WiFi provisioning mode")
        return self.wifi_manager.start_hotspot()

    def accept_credentials(self, ssid: str, password: str) -> bool:
        """Validate and save new WiFi credentials provided by the user."""
        credentials = WiFiCredentials(ssid=ssid, password=password)
        if not self.wifi_manager.validate_credentials(credentials):
            return False
        return self.wifi_manager.save_credentials(credentials)
