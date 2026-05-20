"""
Robot Application - Main Entry Point

Orchestrates startup sequence and robot lifecycle:

1. BOOTING: Initialize logging and load configuration
2. CONNECTING: Establish hardware connections
3. CHECKING_SYSTEMS: Run health diagnostics
4. READY: Await commands or enter autonomous mode
5. OPERATION: Execute commands, process detections
6. SHUTDOWN: Clean termination

Architecture:
- State machine drives behavior
- Services handle distinct responsibilities
- Hardware abstraction isolates from implementation details
- Event-driven communication between components

This is the only module that directly orchestrates all subsystems.
All other modules should be independent and testable.
"""

import os
import sys
import logging
import signal
import time
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Optional, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import configuration first (needed by all modules)
import configs


from utils.logger import setup_logging, get_logger

setup_logging(
    log_level=configs.LOG_LEVEL,
    log_dir=str(configs.LOG_DIR),
    app_name=configs.APP_NAME,
)

logger = get_logger(__name__)

# Import all components
from hardware.arduino import ArduinoConnection, ArduinoCommunicationError
from hardware.motors import MotorController, MovementDirection
from states.state_machine import StateMachine, RobotState
from services.health_service import HealthCheckService, HealthCheckStatus
from services.emergency_service import EmergencyStopService
from services.lifecycle_manager import LifecycleManager
from services.watchdog_service import WatchdogService
from services.wifi_service import WiFiManager, HotspotProvisioningService
from services.network_service import NetworkService
from services.camera_service import CameraService
from services.model_service import ModelService


class RobotApplication:
    """
    Main robot application class.
    
    Manages the complete lifecycle:
    - Hardware initialization
    - Startup checks
    - State transitions
    - Graceful shutdown
    
    Usage:
        robot = RobotApplication()
        robot.startup()
        robot.run()
        robot.shutdown()
    """

    def __init__(self):
        """Initialize robot application components."""
        logger.info("Initializing Robot Application...")
        
        # State machine
        self.state_machine = StateMachine()
        self.lifecycle_manager = LifecycleManager()
        
        # Hardware components
        self.arduino: Optional[ArduinoConnection] = None
        self.motors: Optional[MotorController] = None
        
        # Services
        self.health_service = HealthCheckService()
        self.wifi_manager = WiFiManager()
        self.hotspot_service = HotspotProvisioningService(self.wifi_manager)
        self.network_service = NetworkService()
        self.camera_service = CameraService()
        self.model_service = ModelService()
        self.emergency_service = EmergencyStopService()
        self.watchdog_service = WatchdogService()

        self.emergency_service.register_callback(self._stop_all_movement)
        self.emergency_service.register_callback(self._on_emergency_requested)
        self.watchdog_service.register_failure_callback(self._on_watchdog_failure)
        
        # Lifecycle
        self._event_queue: Queue[Tuple[str, Optional[object]]] = Queue()
        self._shutdown_requested = threading.Event()
        self._shutdown_started = False
        self._provisioning_active = False
        self._is_running = False
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        
        logger.info("✓ Application initialized")

    def startup(self) -> bool:
        """Execute a safe startup sequence with rollback support."""
        logger.info(f"\n{configs.get_config_summary()}\n")
        logger.info("=" * 70)
        logger.info("STARTING ROBOT STARTUP SEQUENCE")
        logger.info("=" * 70)

        self._shutdown_requested.clear()
        self._shutdown_started = False
        self._provisioning_active = False
        self._is_running = False
        self.lifecycle_manager = LifecycleManager()

        try:
            if not self.state_machine.transition_to(
                RobotState.BOOTING,
                reason="Startup initiated",
            ):
                logger.error("Failed to enter BOOTING state")
                return False

            logger.info("\n[1/4] PHASE: BOOTING")
            logger.info("-" * 70)
            time.sleep(0.5)

            if not self._startup_network():
                self._rollback_startup()
                return False

            if self._provisioning_active:
                logger.info("Provisioning mode is active; normal runtime startup is suspended")
                return True

            if not self._startup_server():
                self._rollback_startup()
                return False

            if not self._startup_hardware():
                self._rollback_startup()
                return False

            if not self._startup_services():
                self._rollback_startup()
                return False

            logger.info("\n[5/5] PHASE: ENTERING READY STATE")
            logger.info("-" * 70)
            if not self.state_machine.transition_to(
                RobotState.READY,
                reason="All systems operational",
            ):
                logger.error("Failed to enter READY state")
                self._rollback_startup()
                return False

            logger.info("\n" + "=" * 70)
            logger.info("✓ STARTUP COMPLETE - ROBOT READY")
            logger.info("=" * 70)
            self._is_running = True
            return True

        except Exception as exc:
            logger.error("Startup failed with exception", exc_info=True)
            self.state_machine.transition_to(
                RobotState.ERROR,
                reason="Startup exception",
                error_message=str(exc),
            )
            self._rollback_startup()
            return False

    def _startup_network(self) -> bool:
        """Initialize network connectivity or enter provisioning mode."""
        self.state_machine.transition_to(
            RobotState.WIFI_CONNECTING,
            reason="Connecting to WiFi",
        )
        
        # check if we already connected to wifi network
        if self.wifi_manager.is_wifi_connected():
            logger.info("Already connected to Wifi network")
            return True
        if self.network_service.has_network_connection():
            logger.info("Has network connection already")
            return True
        if self.wifi_manager.connect_saved_networks():
            logger.info("Connected to saved WiFi network")
            return True

        logger.info("No saved WiFi network available, entering provisioning mode")
        if not self.hotspot_service.enter_provisioning_mode():
            logger.error("WiFi provisioning failed")
            return False

        self._provisioning_active = True
        self.state_machine.transition_to(
            RobotState.HOTSPOT_MODE,
            reason="WiFi provisioning active",
        )
        return True

    def _startup_server(self) -> bool:
        """Establish connection to the remote control server."""
        self.state_machine.transition_to(
            RobotState.SERVER_CONNECTING,
            reason="Connecting to remote server",
        )

        if not self.network_service.connect_to_server():
            logger.error("Remote server connection failed")
            return False

        logger.info("Connected to remote control server")
        return True

    def _startup_hardware(self) -> bool:
        """Initialize Arduino and motor hardware."""
        self.arduino = ArduinoConnection(
            port=configs.ARDUINO_PORT,
            baudrate=configs.ARDUINO_BAUDRATE,
            timeout=configs.ARDUINO_TIMEOUT,
            write_timeout=configs.ARDUINO_WRITE_TIMEOUT,
        )

        if not self.arduino.connect():
            logger.error("Arduino connection failed")
            return False

        self.lifecycle_manager.register("arduino", self.arduino.disconnect)

        self.motors = MotorController(self.arduino)
        if not self.motors.initialize():
            logger.error("Motor controller initialization failed")
            return False

        self.lifecycle_manager.register("motors", self._cleanup_motors)
        return True

    def _cleanup_motors(self) -> None:
        if self.motors:
            self.motors.stop()

    def _startup_services(self) -> bool:
        """Start camera, model, and watchdog services."""
        self.state_machine.transition_to(
            RobotState.CHECKING_SYSTEMS,
            reason="Starting services and health checks",
        )

        try:
            self.camera_service.start()
            if not self.camera_service.wait_for_first_frame(timeout=5.0):
                logger.error("Camera failed to provide a first frame")
                return False
            self.lifecycle_manager.register("camera", self.camera_service.stop)

            self.model_service.start_streaming(self.camera_service, throttle_fps=configs.TARGET_FPS)
            self.lifecycle_manager.register("model", self.model_service.stop_streaming)

            self._register_watchdog_targets()
            self.lifecycle_manager.register("watchdog", self.watchdog_service.stop)

            return True
        except Exception as exc:
            logger.error(f"Service startup failed: {exc}")
            return False

    def _enqueue_event(self, event_type: str, payload: Optional[object] = None) -> None:
        if self._shutdown_requested.is_set():
            return
        self._event_queue.put((event_type, payload))

    def _handle_event(self, event_type: str, payload: Optional[object]) -> None:
        if event_type == "emergency":
            logger.warning(f"Processing emergency event: {payload}")
            self.shutdown()
        else:
            logger.debug(f"Unhandled event type: {event_type}")

    def _perform_periodic_housekeeping(self) -> None:
        # Placeholder for periodic health checks or maintenance.
        return

    def run(self) -> None:
        """
        Main robot operating loop.
        
        In READY state, the robot waits for commands.
        In production, this would:
        - Listen for WebSocket commands
        - Process detection events
        - Execute autonomous mode routes
        """
        if not self._is_running:
            logger.warning("Cannot run: startup was not successful")
            return

        logger.info("\nEntering main operating loop...")
        logger.info("Robot ready for commands")
        logger.info("(Press Ctrl+C to shutdown)\n")

        try:
            while not self._shutdown_requested.is_set():
                if self.emergency_service.is_engaged():
                    self._enqueue_event("emergency", self.emergency_service.reason())

                try:
                    event_type, payload = self._event_queue.get(timeout=0.5)
                    self._handle_event(event_type, payload)
                except Empty:
                    self._perform_periodic_housekeeping()
                    continue

                if self._shutdown_requested.is_set():
                    break

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as exc:
            logger.error("Error in main loop", exc_info=True)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """
        Gracefully shutdown robot.
        
        Flow:
        1. Transition to SHUTDOWN state
        2. Stop all movement
        3. Disconnect hardware
        4. Close resources
        """
        if self._shutdown_started:
            logger.debug("Shutdown already in progress")
            return

        self._shutdown_started = True
        self._shutdown_requested.set()
        self._is_running = False

        logger.info("\n" + "=" * 70)
        logger.info("INITIATING SHUTDOWN SEQUENCE")
        logger.info("=" * 70)

        try:
            self.state_machine.transition_to(
                RobotState.SHUTDOWN,
                reason="Graceful shutdown",
            )

            if self.emergency_service.is_engaged():
                logger.warning("Emergency engagement detected during shutdown")
                self._stop_all_movement()

            self.lifecycle_manager.cleanup_all()

            logger.info("✓ Shutdown complete")
            logger.info("=" * 70)

        except Exception:
            logger.exception("Error during shutdown")

    def _rollback_startup(self) -> None:
        """Rollback partially initialized resources after startup failure."""
        logger.warning("Rolling back startup sequence")
        self._shutdown_requested.set()
        self._is_running = False
        provisioning_was_active = self._provisioning_active
        self._provisioning_active = False

        try:
            self._stop_all_movement("startup rollback")
        except Exception as exc:
            logger.error(f"Rollback failed to stop movement: {exc}")

        try:
            self.watchdog_service.stop()
        except Exception as exc:
            logger.error(f"Rollback failed to stop watchdog service: {exc}")

        try:
            self.model_service.stop_streaming()
        except Exception as exc:
            logger.error(f"Rollback failed to stop model service: {exc}")

        try:
            self.camera_service.stop()
        except Exception as exc:
            logger.error(f"Rollback failed to stop camera service: {exc}")

        try:
            self.network_service.disconnect()
        except Exception as exc:
            logger.error(f"Rollback failed to disconnect network service: {exc}")

        try:
            if self.arduino is not None:
                self.arduino.disconnect()
        except Exception as exc:
            logger.error(f"Rollback failed to disconnect Arduino: {exc}")

        try:
            if provisioning_was_active:
                self.hotspot_service.wifi_manager.stop_hotspot()
        except Exception as exc:
            logger.error(f"Rollback failed to stop hotspot provisioning: {exc}")

        try:
            self.lifecycle_manager.cleanup_all()
        except Exception as exc:
            logger.error(f"Rollback lifecycle cleanup failed: {exc}")

        if self.state_machine.get_current_state() != RobotState.ERROR:
            self.state_machine.transition_to(
                RobotState.ERROR,
                reason="Startup rollback",
                error_message="Startup failed during initialization",
            )

    def _handle_shutdown_signal(self, signum, frame):
        """Handle OS shutdown signals (SIGINT, SIGTERM)."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown_requested.set()

    def _handle_check_failure(self, result) -> None:
        """Log and record health check failures."""
        logger.error(f"✗ {result.name}: {result.message}")

    def _register_watchdog_targets(self) -> None:
        self.watchdog_service.register_target(
            name="Camera Capture",
            check_fn=self._check_camera_watchdog,
            recovery_fn=self._restart_camera_service,
            critical=True,
        )
        self.watchdog_service.register_target(
            name="Model Inference",
            check_fn=self._check_model_watchdog,
            recovery_fn=self._restart_model_service,
            critical=True,
        )
        self.watchdog_service.register_target(
            name="Arduino Connection",
            check_fn=self._check_arduino_watchdog,
            recovery_fn=self._restart_arduino_connection,
            critical=True,
        )
        self.watchdog_service.start()

    def _check_camera_watchdog(self):
        if not self.camera_service.is_running():
            return False, "Camera capture thread inactive"
        age = self.camera_service.get_last_frame_age()
        if age is None:
            return False, "No frame received from camera"
        if age > 2.0:
            return False, f"Camera frame age stale ({age:.1f}s)"
        return True, "Camera healthy"

    def _restart_camera_service(self) -> bool:
        logger.warning("Watchdog attempting camera recovery")
        self.camera_service.stop()
        self.camera_service.start()
        return self.camera_service.wait_for_first_frame(timeout=5.0)

    def _check_model_watchdog(self):
        if not self.model_service.is_streaming():
            return False, "Model inference not running"
        age = self.model_service.get_last_inference_age()
        if age is None:
            return False, "No inference output yet"
        if age > 5.0:
            return False, f"Inference output stale ({age:.1f}s)"
        return True, "Model inference healthy"

    def _restart_model_service(self) -> bool:
        logger.warning("Watchdog attempting model recovery")
        self.model_service.restart_streaming(self.camera_service, throttle_fps=configs.TARGET_FPS)
        return self.model_service.is_streaming()

    def _check_arduino_watchdog(self):
        if self.arduino is None or not self.arduino.is_connected():
            return False, "Arduino disconnected"
        if self.arduino.last_response_age() > max(3.0, configs.ARDUINO_TIMEOUT * 2):
            if self.arduino.ping():
                return True, "Arduino responding"
            return False, "Arduino not responding"
        return True, "Arduino healthy"

    def _restart_arduino_connection(self) -> bool:
        logger.warning("Watchdog attempting Arduino recovery")
        if self.arduino is None:
            return False
        return self.arduino.reconnect_if_needed()

    def _on_emergency_requested(self, reason: str) -> None:
        logger.critical(f"Emergency requested: {reason}")
        self._stop_all_movement()
        self.shutdown()

    def _on_watchdog_failure(self, target_name: str, message: str) -> None:
        logger.critical(f"Watchdog failure for {target_name}: {message}")
        self.emergency_service.engage(
            f"Watchdog failure ({target_name}): {message}"
        )

    def _stop_all_movement(self, reason: Optional[str] = None) -> None:
        if reason:
            logger.warning(f"Stopping all movement due to emergency: {reason}")
        else:
            logger.warning("Stopping all movement due to emergency")

        if self.motors:
            try:
                self.motors.stop()
            except Exception as exc:
                logger.error(f"Failed to stop motors: {exc}")
        if self.arduino:
            try:
                self.arduino.send_emergency_stop()
            except Exception as exc:
                logger.error(f"Emergency Arduino stop failed: {exc}")

    def _check_camera_available(self) -> bool:
        """Check if camera is available (placeholder)."""
        # TODO: Implement actual camera availability check
        # For now, assume available
        logger.debug("Camera availability check (placeholder)")
        return True

    def _check_model_available(self) -> bool:
        """Check if AI model is available and loadable (placeholder)."""
        # TODO: Implement actual model loading and verification
        # For now, assume available
        logger.debug("Model availability check (placeholder)")
        return True

    def print_diagnostics(self) -> None:
        """Print diagnostic information about robot state."""
        logger.info("\n" + "=" * 70)
        logger.info("ROBOT DIAGNOSTICS")
        logger.info("=" * 70)
        
        logger.info(f"Current State: {self.state_machine}")
        
        if self.motors:
            motor_status = self.motors.get_status()
            if motor_status:
                logger.info(f"Motors: {motor_status}")
        
        logger.info("\nHealth Check Results:")
        for name, result in self.health_service.get_results().items():
            logger.info(f"  {name}: {result.status.value} ({result.duration_ms:.1f}ms)")
            if result.message:
                logger.info(f"    → {result.message}")
        
        logger.info("\nRecent State Transitions:")
        for transition in self.state_machine.get_transition_history(limit=5):
            logger.info(
                f"  {transition.timestamp.strftime('%H:%M:%S')} | "
                f"{transition.from_state.value} → {transition.to_state.value} "
                f"({transition.reason})"
            )
        
        logger.info("=" * 70 + "\n")


def main():
    """Application entry point."""
    robot = RobotApplication()
    
    # Startup
    if not robot.startup():
        logger.error("Startup failed, exiting")
        sys.exit(1)
    
    # Print diagnostics
    robot.print_diagnostics()
    
    # Run
    robot.run()
    
    # Exit
    logger.info("\nApplication terminated")
    sys.exit(0)


if __name__ == "__main__":
    main()

