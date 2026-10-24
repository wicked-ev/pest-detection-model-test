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
import socket
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent

# Load .env from the project folder, not from wherever the program was started
# (a system service usually starts it from another directory).
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    try:
        return float(value) if value else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    try:
        return int(value) if value else default
    except ValueError:
        return default

# ============================================================================
# DIRECTORIES
# ============================================================================
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLES_DIR = PROJECT_ROOT / "Samples"
CONFIG_DIR = PROJECT_ROOT / "config"

# Create directories if they don't exist
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)

# ============================================================================
# ARDUINO SERIAL CONFIGURATION
# ============================================================================

# Serial port where Arduino Uno is connected
# Linux/Raspberry Pi: "/dev/ttyACM0", "/dev/ttyUSB0"
# macOS: "/dev/tty.usbmodem14101" (varies by device)
# Windows: "COM3", "COM4" (varies by device)
ARDUINO_PORT = os.getenv("ARDUINO_PORT", "/dev/ttyUSB0")

_baud_env = os.getenv("ARDUINO_BAUDRATE", "9600")
ARDUINO_BAUDRATE = int(_baud_env) if _baud_env.strip() else 9600

# Serial read/write timeouts (seconds)
ARDUINO_TIMEOUT = 2.0
ARDUINO_WRITE_TIMEOUT = 2.0

# Set ARDUINO_ENABLED=false to run without the motor board (e.g. camera-only testing).
ARDUINO_ENABLED = _env_bool("ARDUINO_ENABLED", True)

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
CAMERA_WIDTH = _env_int("CAMERA_WIDTH", 640)
CAMERA_HEIGHT = _env_int("CAMERA_HEIGHT", 480)
# Capture rate. Nothing downstream uses more than STREAM_FPS, so higher only costs CPU.
CAMERA_FPS = _env_int("CAMERA_FPS", 15)
# Comma-separated camera backends to try in order: picamera2, opencv, v4l2, synthetic
# ("synthetic" draws test frames, for running without a camera).
_camera_backends = os.getenv("CAMERA_BACKENDS", "").strip()
CAMERA_BACKENDS = (
    tuple(b.strip().lower() for b in _camera_backends.split(",") if b.strip())
    if _camera_backends
    else None
)

# Frame processing
INFERENCE_WIDTH = 320
INFERENCE_HEIGHT = 320
FRAME_SKIP = 4  # Process every Nth frame to reduce Pi CPU load
TARGET_FPS = 15  # Upper limit for local inference rate

# Frames per second sent to the server (live view, and detection input in server mode).
# 5 keeps a Pi 3's CPU and Wi-Fi comfortable; raise it on faster hardware.
STREAM_FPS = _env_float("STREAM_FPS", 5.0)
JPEG_QUALITY = _env_int("JPEG_QUALITY", 75)

# Where detection runs:
#   auto   - run the model on the robot if a runtime and model are available, else on the server
#   local  - only on the robot (no server fallback)
#   server - always on the server (the robot only sends frames)
# When detection runs on the robot, frames and its results are still sent to the server for monitoring.
DETECTION_MODE = os.getenv("DETECTION_MODE", "auto").strip().lower()
if DETECTION_MODE not in {"auto", "local", "server"}:
    DETECTION_MODE = "auto"

# ============================================================================
# AI DETECTION MODEL
# ============================================================================

# Model backend preference and selection.
# Use MODEL_BACKEND to force a specific backend, otherwise fall back through the preference list.
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "tflite").lower()
_RAW_BACKEND_PREFERENCE = os.getenv("MODEL_BACKEND_PREFERENCE", "")
if _RAW_BACKEND_PREFERENCE:
    MODEL_BACKEND_PREFERENCE = [item.strip().lower() for item in _RAW_BACKEND_PREFERENCE.split(",") if item.strip()]
else:
    _default_backends = ["tflite", "onnx"]
    MODEL_BACKEND_PREFERENCE = [MODEL_BACKEND] + [item for item in _default_backends if item != MODEL_BACKEND]

# Primary model path for Raspberry Pi inference.
MODEL_PATH = MODELS_DIR / os.getenv("MODEL_FILENAME", "pest_model.tflite")
ONNX_MODEL_PATH = MODELS_DIR / os.getenv("ONNX_MODEL_FILENAME", "pest_model.onnx")

# Default download URLs for model assets (override via env vars)
# These should point to stable, versioned releases in production.
MODEL_DOWNLOAD_URL = os.getenv("MODEL_DOWNLOAD_URL", "https://example.com/assets/model.tflite")
ONNX_MODEL_DOWNLOAD_URL = os.getenv("ONNX_MODEL_DOWNLOAD_URL", "https://example.com/assets/model.onnx")
ANNOTATIONS_DOWNLOAD_URL = os.getenv("ANNOTATIONS_DOWNLOAD_URL", "https://example.com/assets/_annotations.coco.json")

# CPU threads for the local model (the Pi 3B+ has 4 cores).
MODEL_NUM_THREADS = max(1, _env_int("MODEL_NUM_THREADS", os.cpu_count() or 4))

# Model inference parameters
MODEL_CONFIDENCE_THRESHOLD = 0.5
MODEL_IOU_THRESHOLD = 0.45

# ============================================================================
# NETWORK CONFIGURATION
# ============================================================================

_host_env = os.getenv("SERVER_HOST", "localhost")
SERVER_HOST = _host_env if _host_env.strip() else "localhost"
_port_env = os.getenv("SERVER_PORT", "8000")
SERVER_PORT = int(_port_env) if _port_env.strip() else 8000
SERVER_URL = f"{SERVER_HOST}:{SERVER_PORT}"
# Empty counts as unset: an empty ID made the server URL end in "/ws/control/", which the
# server rejects (HTTP 403, same as a wrong key).
ROBOT_ID = os.getenv("ROBOT_ID", "").strip() or socket.gethostname()
# Shared secret the server requires on the robot's websockets (must match the server's ROBOT_KEY).
ROBOT_KEY = os.getenv("ROBOT_KEY", "").strip()

# How long startup waits for the server before carrying on without it.
# The robot keeps reconnecting in the background either way.
SERVER_CONNECT_WAIT = _env_float("SERVER_CONNECT_WAIT", 15.0)

# Network retry logic
NETWORK_RETRY_ATTEMPTS = 3
NETWORK_RETRY_DELAY = 2.0

# WiFi provisioning and hotspot configuration
WIFI_CREDENTIALS_PATH = CONFIG_DIR / "wifi_credentials.json"
WPA_SUPPLICANT_PATH = Path(os.getenv("WPA_SUPPLICANT_PATH", "/etc/wpa_supplicant/wpa_supplicant.conf"))
WIFI_CONNECT_TIMEOUT = 20.0
WIFI_RETRY_ATTEMPTS = 3
WIFI_RETRY_DELAY = 3.0
# Set WIFI_MANAGEMENT_ENABLED=false where the OS manages networking (or on a PC).
WIFI_MANAGEMENT_ENABLED = _env_bool("WIFI_MANAGEMENT_ENABLED", True)
# Hotspot used to give the robot Wi-Fi credentials. Change the password per robot:
# anyone who joins this network can send it new Wi-Fi settings.
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

# Verbose streaming logs toggle
STREAMING_LOGS_ENABLED = os.getenv("STREAMING_LOGS_ENABLED", "False").lower() in {"true", "1", "yes", "on"}


def get_config_summary() -> str:
    """Return a summary of current configuration."""
    return f"""
    ═══════════════════════════════════════════════════════════════
    ROBOT CONFIGURATION
    ═══════════════════════════════════════════════════════════════
    
    Hardware:
      Arduino: {ARDUINO_PORT if ARDUINO_ENABLED else "disabled"} @ {ARDUINO_BAUDRATE}
      Camera: device {CAMERA_DEVICE}, {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} fps, backends {CAMERA_BACKENDS or "auto"}
    
    Models:
      Detection Mode: {DETECTION_MODE}
      Detection Model: {MODEL_PATH}
      Model Threads: {MODEL_NUM_THREADS}
      Model Confidence: {MODEL_CONFIDENCE_THRESHOLD}
    
    Network:
      Server URL: {SERVER_URL}
      Robot ID: {ROBOT_ID}
      Robot Key: {"set" if ROBOT_KEY else "NOT SET"}
      Stream FPS: {STREAM_FPS}
    
    Logging:
      Log Level: {LOG_LEVEL}
      Log Directory: {LOG_DIR}
    
    Parameters:
      Inference Size: {INFERENCE_WIDTH}x{INFERENCE_HEIGHT}
      Target FPS: {TARGET_FPS}
      Movement Timeout: {MOVEMENT_TIMEOUT}s
    
    ═══════════════════════════════════════════════════════════════
    """
