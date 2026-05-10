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

import sys, os
import logging
from pathlib import Path
import signal
import time

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
        
        # Hardware components
        self.arduino: ArduinoConnection | None = None
        self.motors: MotorController | None = None
        
        # Services
        self.health_service = HealthCheckService()
        self.wifi_manager = WiFiManager()
        self.hotspot_service = HotspotProvisioningService(self.wifi_manager)
        self.network_service = NetworkService()
        self.camera_service = CameraService()
        self.model_service = ModelService()
        
        # Lifecycle
        self._is_running = False
        self._shutdown_requested = False
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        
        logger.info("✓ Application initialized")

    def startup(self) -> bool:
        """
        Execute startup sequence.
        
        Flow:
        1. BOOTING → Initialize runtime environment
        2. WIFI_CONNECTING → Connect to WiFi or provision hotspot
        3. SERVER_CONNECTING → Establish remote server link
        4. CONNECTING → Establish Arduino connection
        5. CHECKING_SYSTEMS → Run diagnostics
        6. READY → All systems operational
        
        Returns:
            True if startup successful, False if failed
        """
        logger.info(f"\n{configs.get_config_summary()}\n")
        logger.info("=" * 70)
        logger.info("STARTING ROBOT STARTUP SEQUENCE")
        logger.info("=" * 70)
        
        try:
            # --- Phase 1: BOOTING ---
            if not self.state_machine.transition_to(
                RobotState.BOOTING,
                reason="Startup initiated"
            ):
                logger.error("Failed to enter BOOTING state")
                return False
            
            logger.info("\n[1/4] PHASE: BOOTING")
            logger.info("-" * 70)
            time.sleep(0.5)
            
            # --- Phase 2: WIFI CONNECTION ---
            logger.info("\n[2/5] PHASE: WIFI CONNECTION")
            logger.info("-" * 70)
            
            if not self.state_machine.transition_to(
                RobotState.WIFI_CONNECTING,
                reason="Attempting saved WiFi networks"
            ):
                logger.error("Failed to enter WIFI_CONNECTING state")
                return False
            
            wifi_connected = self.wifi_manager.connect_saved_networks()
            if not wifi_connected:
                logger.warning("Saved WiFi credentials unavailable or failed")
                if not self.hotspot_service.enter_provisioning_mode():
                    error_msg = "Unable to start hotspot provisioning"
                    self.state_machine.transition_to(
                        RobotState.ERROR,
                        reason="Hotspot provisioning failed",
                        error_message=error_msg,
                    )
                    return False
                
                self.state_machine.transition_to(
                    RobotState.HOTSPOT_MODE,
                    reason="Hotspot mode active for WiFi provisioning"
                )
                self._is_running = True
                logger.info("Hotspot mode enabled; waiting for WiFi credentials")
                return True
            
            logger.info("✓ WiFi connection established")
            
            # --- Phase 3: SERVER CONNECTION ---
            logger.info("\n[3/5] PHASE: SERVER CONNECTION")
            logger.info("-" * 70)
            
            if not self.state_machine.transition_to(
                RobotState.SERVER_CONNECTING,
                reason="Connecting to remote control server"
            ):
                logger.error("Failed to enter SERVER_CONNECTING state")
                return False
            
            if not self.network_service.connect_to_server():
                error_msg = f"Failed to connect to server at {configs.SERVER_URL}"
                logger.error(f"✗ {error_msg}")
                self.state_machine.transition_to(
                    RobotState.ERROR,
                    reason="Server connection failed",
                    error_message=error_msg,
                )
                return False
            
            logger.info("✓ Connected to remote control server")
            
            # --- Phase 4: Initialize Hardware Connection ---
            logger.info("\n[4/5] PHASE: CONNECTING HARDWARE")
            logger.info("-" * 70)
            
            # Connect to Arduino
            logger.info("Connecting to Arduino...")
            self.arduino = ArduinoConnection(
                port=configs.ARDUINO_PORT,
                baudrate=configs.ARDUINO_BAUDRATE,
                timeout=configs.ARDUINO_TIMEOUT,
                write_timeout=configs.ARDUINO_WRITE_TIMEOUT,
                max_reconnect_attempts=configs.ARDUINO_MAX_RECONNECT_ATTEMPTS,
                reconnect_delay=configs.ARDUINO_RECONNECT_DELAY,
            )
            
            if not self.arduino.connect():
                error_msg = (
                    f"Failed to connect to Arduino on {configs.ARDUINO_PORT}"
                )
                logger.error(f"✗ {error_msg}")
                self.state_machine.transition_to(
                    RobotState.ERROR,
                    reason="Arduino connection failed",
                    error_message=error_msg,
                )
                return False
            
            logger.info("✓ Arduino connected")
            
            # Initialize motors
            logger.info("Initializing motor controller...")
            self.motors = MotorController(self.arduino)
            
            # --- Phase 5: CHECKING_SYSTEMS ---
            logger.info("\n[5/5] PHASE: CHECKING SYSTEMS")
            logger.info("-" * 70)
            
            if not self.state_machine.transition_to(
                RobotState.CHECKING_SYSTEMS,
                reason="Starting system diagnostics"
            ):
                logger.error("Failed to enter CHECKING_SYSTEMS state")
                return False
            
            # Run health checks
            self.health_service.clear_results()
            
            # Check 1: Arduino Connection
            logger.info("\nCheck 1: Arduino Connection")
            result = self.health_service.check_arduino_connection(self.arduino)
            if result.status == HealthCheckStatus.ERROR:
                self._handle_check_failure(result)
                return False
            
            # Check 2: Motors
            logger.info("\nCheck 2: Motor System")
            result = self.health_service.check_motors(self.motors)
            if result.status == HealthCheckStatus.ERROR:
                self._handle_check_failure(result)
                return False
            
            # Check 3: Camera
            logger.info("\nCheck 3: Camera System")
            result = self.health_service.check_camera(
                camera_available_fn=self.camera_service.is_available
            )
            if result.status == HealthCheckStatus.ERROR:
                logger.warning(f"⚠ Camera check failed: {result.message}")
                # Camera failure is not critical, continue
            
            # Check 4: AI Model
            logger.info("\nCheck 4: AI Detection Model")
            result = self.health_service.check_ai_model(
                model_path=str(configs.MODEL_PATH),
                model_load_fn=self.model_service.verify_load
            )
            if result.status == HealthCheckStatus.ERROR:
                logger.warning(f"⚠ Model check failed: {result.message}")
                # Model failure is not critical, continue
            
            # Verify all critical checks passed
            if not self.health_service.is_all_healthy():
                error_summary = self.health_service.get_error_summary()
                logger.error(f"✗ Critical health checks failed: {error_summary}")
                self.state_machine.transition_to(
                    RobotState.ERROR,
                    reason="Health checks failed",
                    error_message=error_summary,
                )
                return False
            
            logger.info("\n✓ All health checks passed!")
            
            # --- Phase 4: READY ---
            logger.info("\n[4/4] PHASE: ENTERING READY STATE")
            logger.info("-" * 70)
            
            if not self.state_machine.transition_to(
                RobotState.READY,
                reason="All systems operational"
            ):
                logger.error("Failed to enter READY state")
                return False
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ STARTUP COMPLETE - ROBOT READY")
            logger.info("=" * 70)
            
            self._is_running = True
            return True

        except Exception as e:
            logger.error(f"Startup failed with exception: {e}", exc_info=True)
            self.state_machine.transition_to(
                RobotState.ERROR,
                reason="Startup exception",
                error_message=str(e),
            )
            return False

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
            # In a real application, this would:
            # - Listen to WebSocket for commands
            # - Run detection service in background
            # - Process autonomous mode
            # - Handle state transitions
            
            while not self._shutdown_requested and self._is_running:
                current_state = self.state_machine.get_current_state()
                
                # Handle state-specific operations
                if current_state == RobotState.READY:
                    time.sleep(0.5)
                
                elif current_state == RobotState.WIFI_CONNECTING:
                    logger.debug("Waiting for WiFi connection")
                    time.sleep(0.5)
                
                elif current_state == RobotState.SERVER_CONNECTING:
                    logger.debug("Waiting for server connection")
                    time.sleep(0.5)
                
                elif current_state == RobotState.HOTSPOT_MODE:
                    logger.info("Hotspot mode is active for provisioning")
                    time.sleep(2.0)
                
                elif current_state == RobotState.PROVISIONING:
                    logger.info("Awaiting WiFi credentials from provisioning client")
                    time.sleep(2.0)
                
                elif current_state == RobotState.REMOTE_CONTROL:
                    logger.debug("Remote control operational")
                    time.sleep(0.5)
                
                elif current_state == RobotState.AUTONOMOUS:
                    logger.debug("Autonomous mode active")
                    time.sleep(0.5)
                
                elif current_state == RobotState.ERROR:
                    logger.warning("Robot in ERROR state, awaiting recovery...")
                    time.sleep(1.0)
                
                elif current_state == RobotState.SHUTDOWN:
                    logger.info("Shutdown state reached")
                    break
                
                else:
                    logger.debug(f"Current state: {current_state.value}")
                    time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
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
        if not self._is_running:
            return
        
        logger.info("\n" + "=" * 70)
        logger.info("INITIATING SHUTDOWN SEQUENCE")
        logger.info("=" * 70)
        
        self._is_running = False
        
        try:
            # Transition to shutdown state
            self.state_machine.transition_to(
                RobotState.SHUTDOWN,
                reason="Graceful shutdown"
            )
            
            # Stop motors immediately
            if self.motors:
                logger.info("Stopping motors...")
                self.motors.stop()
                time.sleep(0.5)
            
            # Disconnect network
            if self.network_service:
                logger.info("Disconnecting network services...")
                self.network_service.disconnect()

            # Disconnect Arduino
            if self.arduino:
                logger.info("Disconnecting Arduino...")
                self.arduino.disconnect()
            
            logger.info("✓ Shutdown complete")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    def _handle_shutdown_signal(self, signum, frame):
        """Handle OS shutdown signals (SIGINT, SIGTERM)."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown_requested = True

    def _handle_check_failure(self, result) -> None:
        """Log and record health check failures."""
        logger.error(f"✗ {result.name}: {result.message}")

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

