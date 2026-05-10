"""
Quick Start Guide - Testing Individual Components

This script demonstrates how to use each module independently.
Useful for debugging and understanding the architecture.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
from utils.logger import setup_logging, get_logger

setup_logging(log_level="DEBUG")
logger = get_logger(__name__)

import configs
from hardware.arduino import ArduinoConnection
from hardware.motors import MotorController, MovementDirection
from states.state_machine import StateMachine, RobotState
from services.health_service import HealthCheckService, HealthCheckStatus


def demo_state_machine():
    """Demo: State machine transitions."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO: STATE MACHINE")
    logger.info("=" * 70)
    
    sm = StateMachine()
    
    logger.info(f"Initial state: {sm.get_current_state().value}")
    
    # Attempt valid transitions
    sm.transition_to(RobotState.CHECKING_SYSTEMS, reason="Demo transition")
    logger.info(f"After transition: {sm.get_current_state().value}")
    
    sm.transition_to(RobotState.READY, reason="Systems OK")
    logger.info(f"After transition: {sm.get_current_state().value}")
    
    # Show that we can query state
    logger.info(f"Is ready: {sm.is_ready()}")
    logger.info(f"Can move: {sm.can_move()}")
    
    # Show transition history
    logger.info("\nTransition history:")
    for t in sm.get_transition_history():
        logger.info(f"  {t.from_state.value} → {t.to_state.value}: {t.reason}")


def demo_health_service():
    """Demo: Health check service."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO: HEALTH CHECK SERVICE")
    logger.info("=" * 70)
    
    health = HealthCheckService()
    
    # Test camera check (with custom function)
    def camera_check():
        logger.debug("Camera check function called")
        return True
    
    result = health.check_camera(camera_available_fn=camera_check)
    logger.info(f"Camera check: {result.status.value} - {result.message}")
    
    # Test model check (with custom function)
    def model_check():
        logger.debug("Model check function called")
        return True
    
    result = health.check_ai_model(
        model_path=str(configs.MODEL_PATH),
        model_load_fn=model_check
    )
    logger.info(f"Model check: {result.status.value} - {result.message}")
    
    # Get all results
    logger.info("\nAll health check results:")
    for name, result in health.get_results().items():
        logger.info(f"  {name}: {result.status.value} ({result.duration_ms:.1f}ms)")
    
    # Check overall health
    logger.info(f"Overall health: {health.is_all_healthy()}")


def demo_state_callbacks():
    """Demo: State change callbacks."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO: STATE CHANGE CALLBACKS")
    logger.info("=" * 70)
    
    sm = StateMachine()
    
    # Register a callback
    def my_callback(from_state, to_state, reason):
        logger.info(f"CALLBACK: {from_state.value} → {to_state.value} ({reason})")
    
    sm.register_state_change_callback(my_callback)
    
    # Trigger transition (callback will fire)
    sm.transition_to(RobotState.CHECKING_SYSTEMS, reason="Testing callbacks")
    sm.transition_to(RobotState.READY, reason="All clear")


def demo_movement_directions():
    """Demo: Movement direction enum."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO: MOVEMENT DIRECTIONS")
    logger.info("=" * 70)
    
    from hardware.motors import MovementDirection
    
    for direction in MovementDirection:
        logger.info(f"  {direction.name} = {direction.value}")


def demo_arduino_mock():
    """Demo: Simulating Arduino communication (mock)."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO: ARDUINO COMMUNICATION (Mock)")
    logger.info("=" * 70)
    
    # This is what the real implementation looks like
    logger.info("\nArduino command flow:")
    logger.info("1. MotorController.move(FORWARD)")
    logger.info("   └─ Arduino.move('forward')")
    logger.info("       └─ Arduino.send_command(MOVE_FORWARD)")
    logger.info("           ├─ Write 'MOVE_FORWARD\\n' to serial")
    logger.info("           └─ Wait for 'OK' response")
    logger.info("2. Motors update internal state")
    logger.info("3. Return success/failure to caller")
    
    logger.info("\nArduino responses are collected by background reader thread")
    logger.info("Response queue ensures thread-safe command/response pairing")


def demo_configuration():
    """Demo: Configuration system."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO: CONFIGURATION")
    logger.info("=" * 70)
    
    logger.info(configs.get_config_summary())


def main():
    """Run all demos."""
    try:
        demo_state_machine()
        demo_health_service()
        demo_state_callbacks()
        demo_movement_directions()
        demo_arduino_mock()
        demo_configuration()
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ ALL DEMOS COMPLETED")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
