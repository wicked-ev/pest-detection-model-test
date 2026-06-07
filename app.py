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
import json
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
from services.asset_manager import AssetManager


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
        self.model_service.add_listener(self._on_detection)
        self.emergency_service = EmergencyStopService()
        self.watchdog_service = WatchdogService()
        self.network_service.set_stream_detection_callback(self._on_detection)

        self.emergency_service.register_callback(self._stop_all_movement)
        self.emergency_service.register_callback(self._on_emergency_requested)
        self.watchdog_service.register_failure_callback(self._on_watchdog_failure)
        # Asset manager: ensure model assets before starting model service
        self.asset_manager = AssetManager()
        
        # Lifecycle
        self._event_queue: Queue[Tuple[str, Optional[object]]] = Queue()
        self._shutdown_requested = threading.Event()
        self._shutdown_started = False
        self._provisioning_active = False
        self._is_running = False
        self._last_housekeeping_ts = 0.0
        
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
            # if not self.state_machine.transition_to(
            #     RobotState.BOOTING,
            #     reason="Startup initiated",
            # ):
            #     logger.error("Failed to enter BOOTING state")
            #     return False

            logger.info("\n[1/4] PHASE: BOOTING")
            logger.info("-" * 70)
            if self._shutdown_requested.wait(timeout=0.5):
                logger.info("Shutdown requested during startup boot phase")
                self._rollback_startup()
                return False

            if not self._startup_network():
                self._rollback_startup()
                return False

            if self._shutdown_requested.is_set():
                logger.info("Shutdown requested during startup after network initialization")
                self._rollback_startup()
                return False

            if self._provisioning_active:
                logger.info("Provisioning mode is active; normal runtime startup is suspended")
                return True

            if not self._startup_server():
                self._rollback_startup()
                return False

            if self._shutdown_requested.is_set():
                logger.info("Shutdown requested during startup after server connection")
                self._rollback_startup()
                return False

            if not self._startup_hardware():
                self._rollback_startup()
                return False

            if self._shutdown_requested.is_set():
                logger.info("Shutdown requested during startup after hardware initialization")
                self._rollback_startup()
                return False

            if not self._startup_services():
                self._rollback_startup()
                return False

            if self._shutdown_requested.is_set():
                logger.info("Shutdown requested during startup after service initialization")
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

        if not self.network_service.connect_to_server(
            stop_event=self._shutdown_requested,
            robot_id=getattr(configs, "ROBOT_ID", None),
            state_provider=lambda: self.state_machine.get_current_state().value,
        ):
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

    def _startup_cancelled(self, stage: str) -> bool:
        if self._shutdown_requested.is_set():
            logger.info("Shutdown requested during %s; aborting startup", stage)
            return True
        return False

    def _startup_services(self) -> bool:
        """Start camera, model, and watchdog services."""
        self.state_machine.transition_to(
            RobotState.CHECKING_SYSTEMS,
            reason="Starting services and health checks",
        )

        try:
            # Ensure model assets are present and valid before starting services
            logger.info("Checking model assets before service startup")
            remote_fallback = False
            if not self.asset_manager.ensure_assets():
                logger.warning("Required model assets are missing or invalid; enabling remote fallback streaming")
                remote_fallback = True

            if self._startup_cancelled("asset validation"):
                return False

            self.camera_service.start()
            if not self.camera_service.wait_for_first_frame(timeout=60.0, stop_event=self._shutdown_requested):
                backend_name = self.camera_service._backend.name if self.camera_service._backend else "None"
                logger.error("Camera failed to provide a first frame (backend=%s, opened=%s)",
                           backend_name, self.camera_service._opened)
                return False
            self.lifecycle_manager.register("camera", self.camera_service.stop)

            if self._startup_cancelled("camera startup"):
                return False

            camera_check = self.health_service.check_camera(
                lambda: self.camera_service.get_latest(copy=False) is not None
            )
            if camera_check.status != HealthCheckStatus.OK:
                logger.error("Camera health check failed")
                return False

            if self._startup_cancelled("camera health check"):
                return False

            # Try to load local model unless remote fallback explicitly requested
            if not remote_fallback:
                if not self.model_service.load_model():
                    logger.warning("Local model load failed; attempting remote fallback if available")
                else:
                    if self._startup_cancelled("model loading"):
                        return False
                    model_check = self.health_service.check_ai_model(
                        str(self.model_service.model_path),
                        self.model_service.load_model,
                    )
                    if model_check.status != HealthCheckStatus.OK:
                        logger.warning("AI model health check failed; will attempt remote fallback")
                    else:
                        # Local model available and healthy
                        self.model_service.start_streaming(self.camera_service, throttle_fps=configs.TARGET_FPS)
                        self.lifecycle_manager.register("model", self.model_service.stop_streaming)

            if self._startup_cancelled("model startup"):
                return False

            # If no local model was started, attempt remote streaming to server
            if not self.model_service.is_streaming():
                if self.network_service.is_connected():
                    try:
                        logger.info("Starting remote frame streaming to server (fallback mode)")
                        self.model_service.start_remote_streaming(self.camera_service, self.network_service, throttle_fps=configs.TARGET_FPS)
                        self.lifecycle_manager.register("model", self.model_service.stop_streaming)
                    except Exception as exc:
                        logger.error(f"Failed to start remote fallback streaming: {exc}")
                        return False
                else:
                    logger.error("No model available and network server is not connected for fallback")
                    return False

            if self._startup_cancelled("stream startup"):
                return False

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
        elif event_type == "detection":
            logger.info("Processing detection event")
            detections = payload if isinstance(payload, list) else []
            self._send_telemetry({
                "type": "detection",
                "detections": detections,
                "state": self.state_machine.get_current_state().value,
            })
            if self.state_machine.is_ready():
                self.state_machine.transition_to(
                    RobotState.DETECTING,
                    reason="Object detected",
                )
            return
        elif event_type == "diagnostics":
            self.print_diagnostics()
            return
        else:
            logger.debug(f"Unhandled event type: {event_type}")

    def _perform_periodic_housekeeping(self) -> None:
        now = time.time()
        if now - self._last_housekeeping_ts < 10.0:
            return
        self._last_housekeeping_ts = now

        self._check_camera_watchdog()
        self._check_model_watchdog()
        self._check_arduino_watchdog()

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

                self._process_network_messages()

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

    def _process_network_messages(self) -> None:
        if not self.network_service.is_connected():
            return

        try:
            message = self.network_service.receive_message_with_timeout(timeout=0.1)
        except Exception as exc:
            logger.debug(f"Error reading remote command: {exc}")
            return

        if not message:
            return

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Received malformed remote command")
            return

        self._process_remote_command(payload)

    def _on_detection(self, detections: list) -> None:
        logger.info("Detection callback received %d detections", len(detections))
        self._enqueue_event("detection", detections)

    def _process_remote_command(self, payload: object) -> None:
        if not isinstance(payload, dict):
            logger.warning("Remote command payload must be a JSON object")
            return

        command = str(payload.get("command", "")).strip().lower()
        if not command:
            logger.warning("Remote command missing 'command' field")
            return

        if command == "move":
            direction = str(payload.get("direction", "stop")).strip().lower()
            direction_map = {
                "forward": MovementDirection.FORWARD,
                "backward": MovementDirection.BACKWARD,
                "left": MovementDirection.LEFT,
                "right": MovementDirection.RIGHT,
                "stop": MovementDirection.STOP,
            }
            if direction not in direction_map:
                logger.warning(f"Unsupported move direction: {direction}")
                return
            self._execute_movement(direction_map[direction])
            return

        if command in {"stop", "halt", "pause"}:
            self._execute_movement(MovementDirection.STOP)
            return

        if command == "status":
            self._send_telemetry({
                "type": "status",
                "status": self._build_status_payload(),
            })
            return

        if command == "diagnostics":
            self._enqueue_event("diagnostics", None)
            return

        logger.warning(f"Unknown remote command: {command}")

    def _execute_movement(self, direction: MovementDirection) -> None:
        if self.motors is None:
            logger.error("Cannot execute movement: motor controller unavailable")
            return

        if direction != MovementDirection.STOP and not self.state_machine.can_move() and not self.state_machine.is_ready():
            logger.warning("Ignoring movement command because robot is not ready")
            return

        if direction == MovementDirection.STOP:
            self.motors.stop()
            if self.state_machine.is_in_state(RobotState.DETECTING):
                self.state_machine.transition_to(
                    RobotState.READY,
                    reason="Stopped from detection",
                )
            return

        if self.motors.move(direction):
            if self.state_machine.is_ready():
                self.state_machine.transition_to(
                    RobotState.REMOTE_CONTROL,
                    reason=f"Remote move {direction.value}",
                )
        else:
            logger.error(f"Failed to execute movement: {direction.value}")

    def _send_telemetry(self, payload: dict) -> None:
        if not self.network_service.is_connected():
            return
        try:
            self.network_service.send_message(payload)
        except Exception as exc:
            logger.warning(f"Failed to send telemetry: {exc}")

    def _build_status_payload(self) -> dict:
        payload = {
            "state": self.state_machine.get_current_state().value,
            "emergency": self.emergency_service.is_engaged(),
        }
        if self.motors:
            status = self.motors.get_status()
            if status:
                payload.update({
                    "motor_health": status.is_healthy,
                    "motors_moving": status.is_moving,
                    "motor_speeds": [
                        status.motor_0_speed,
                        status.motor_1_speed,
                        status.motor_2_speed,
                        status.motor_3_speed,
                    ],
                })
        return payload

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
            critical=False,
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
        return self.camera_service.wait_for_first_frame(timeout=60.0)

    def _check_model_watchdog(self):
        stream_mode = self.model_service.get_stream_mode()
        if stream_mode is None:
            return False, "Model streaming not running"

        if stream_mode == "remote":
            age = self.model_service.get_last_remote_activity_age()
            if age is None:
                return False, "No remote frame transmission yet"
            if age > 5.0:
                return False, f"Remote frame transmission stale ({age:.1f}s)"
            return True, "Remote fallback streaming healthy"

        age = self.model_service.get_last_inference_age()
        if age is None:
            return False, "No inference output yet"
        if age > 5.0:
            return False, f"Inference output stale ({age:.1f}s)"
        return True, "Model inference healthy"

    def _restart_model_service(self) -> bool:
        logger.warning("Watchdog attempting model recovery")
        stream_mode = self.model_service.get_stream_mode()

        if stream_mode == "remote":
            if not self.network_service.is_connected():
                logger.error("Cannot recover remote fallback streaming: network server is disconnected")
                return False

            self.model_service.stop_streaming()
            self.model_service.start_remote_streaming(
                self.camera_service,
                self.network_service,
                throttle_fps=configs.TARGET_FPS,
            )
            return self.model_service.is_remote_streaming()

        self.model_service.restart_streaming(self.camera_service, throttle_fps=configs.TARGET_FPS)
        return self.model_service.is_local_streaming()

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