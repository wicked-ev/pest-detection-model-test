"""
Robot application configuration.

Centralized configuration for all robot subsystems:
- Hardware connections (Arduino, camera)
- Model paths
- Network endpoints
- Logging preferences
- Tuning parameters

This module should be the ONLY place where hardcoded values appear.
All other modules import from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# DIRECTORIES
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLES_DIR = PROJECT_ROOT / "Samples"
CONFIG_DIR = PROJECT_ROOT / "config"

# Create directories if they don't exist
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# ============================================================================
# ARDUINO SERIAL CONFIGURATION
# ============================================================================

# Serial port where Arduino Uno is connected
# Linux/Raspberry Pi: "/dev/ttyACM0", "/dev/ttyUSB0"
# macOS: "/dev/tty.usbmodem14101" (varies by device)
# Windows: "COM3", "COM4" (varies by device)
ARDUINO_PORT = os.getenv("ARDUINO_PORT", "/dev/ttyUSB0")

# Serial communication speed (must match Arduino sketch)
ARDUINO_BAUDRATE = int(os.getenv("ARDUINO_BAUDRATE", 9600))

# Serial read/write timeouts (seconds)
ARDUINO_TIMEOUT = 2.0
ARDUINO_WRITE_TIMEOUT = 2.0

# Connection retry logic
ARDUINO_MAX_RECONNECT_ATTEMPTS = 3
ARDUINO_RECONNECT_DELAY = 1.0

# ============================================================================
# MOTOR CONFIGURATION
# ============================================================================

# Motor response verification parameters
MOTOR_EXPECTED_SPEED = 200  # Expected PWM value when moving (0-255)
MOTOR_MOVEMENT_VERIFY_TIMEOUT = 1.0  # Seconds to verify movement

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================

# Camera device (0 = default camera, 1 = USB camera, etc.)
CAMERA_DEVICE = 0

# Frame processing
INFERENCE_WIDTH = 320
INFERENCE_HEIGHT = 320
FRAME_SKIP = 4  # Process every Nth frame to reduce Pi CPU load
TARGET_FPS = 15  # Cap display FPS

# ============================================================================
# AI DETECTION MODEL
# ============================================================================

# Path to pest detection model
MODEL_PATH = MODELS_DIR / "pest_model.onnx"

# Model inference parameters
MODEL_CONFIDENCE_THRESHOLD = 0.5
MODEL_IOU_THRESHOLD = 0.45

# ============================================================================
# NETWORK CONFIGURATION
# ============================================================================

# WebSocket server for remote control
SERVER_HOST = os.getenv("SERVER_HOST", "localhost")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
SERVER_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}"

# Network retry logic
NETWORK_RETRY_ATTEMPTS = 3
NETWORK_RETRY_DELAY = 2.0

# WiFi provisioning and hotspot configuration
WIFI_CREDENTIALS_PATH = CONFIG_DIR / "wifi_credentials.json"
WPA_SUPPLICANT_PATH = Path(os.getenv("WPA_SUPPLICANT_PATH", "/etc/wpa_supplicant/wpa_supplicant.conf"))
WIFI_CONNECT_TIMEOUT = 20.0
WIFI_RETRY_ATTEMPTS = 3
WIFI_RETRY_DELAY = 3.0
HOTSPOT_SSID = os.getenv("HOTSPOT_SSID", "robot-hotspot")
HOTSPOT_PASSWORD = os.getenv("HOTSPOT_PASSWORD", "robot1234")
HOTSPOT_INTERFACE = os.getenv("HOTSPOT_INTERFACE", "wlan0")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Application name for log files
APP_NAME = "robot"

# ============================================================================
# ROBOT OPERATIONAL PARAMETERS
# ============================================================================

# Movement command timeout (seconds robot will move before stopping)
MOVEMENT_TIMEOUT = 5.0

# Autonomous mode route (predefined movement sequence)
AUTONOMOUS_ROUTE = [
    "FORWARD",
    "FORWARD",
    "LEFT",
    "FORWARD",
    "FORWARD",
    "RIGHT",
    "FORWARD",
]

# Detection sensitivity (0.0 to 1.0)
DETECTION_SENSITIVITY = 0.5

# ============================================================================
# STARTUP TIMEOUT
# ============================================================================

# How long to wait for startup checks before giving up (seconds)
STARTUP_CHECK_TIMEOUT = 30.0

# ============================================================================
# ERROR RECOVERY
# ============================================================================

# Maximum number of recovery attempts before entering ERROR state
MAX_RECOVERY_ATTEMPTS = 3

# Delay before attempting recovery (seconds)
RECOVERY_DELAY = 2.0


def get_config_summary() -> str:
    """Return a summary of current configuration."""
    return f"""
    ═══════════════════════════════════════════════════════════════
    ROBOT CONFIGURATION
    ═══════════════════════════════════════════════════════════════
    
    Hardware:
      Arduino Port: {ARDUINO_PORT}
      Arduino Baud: {ARDUINO_BAUDRATE}
      Camera Device: {CAMERA_DEVICE}
    
    Models:
      Detection Model: {MODEL_PATH}
      Model Confidence: {MODEL_CONFIDENCE_THRESHOLD}
    
    Network:
      Server URL: {SERVER_URL}
    
    Logging:
      Log Level: {LOG_LEVEL}
      Log Directory: {LOG_DIR}
    
    Parameters:
      Inference Size: {INFERENCE_WIDTH}x{INFERENCE_HEIGHT}
      Target FPS: {TARGET_FPS}
      Movement Timeout: {MOVEMENT_TIMEOUT}s
    
    ═══════════════════════════════════════════════════════════════
    """
