"""
Motor controller abstraction layer.

Provides high-level motor control interface above Arduino communication.
Handles:
- Movement commands (forward, backward, turn, stop)
- Motor health checking
- Speed control (PWM values)
- Emergency stop
- Movement verification

This module abstracts away Arduino protocol details so services
can simply call motor.move("forward") without knowing serial details.
"""

import logging
from typing import Optional
from enum import Enum
from dataclasses import dataclass
import time

from .arduino import ArduinoConnection, HardwareStatus

logger = logging.getLogger(__name__)


class MovementDirection(Enum):
    """Valid movement directions."""
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"


@dataclass
class MotorStatus:
    """Current status of motor system."""
    is_healthy: bool
    is_moving: bool
    left_speed: int  # 0-255
    right_speed: int  # 0-255
    battery_voltage: float
    temperature: float
    last_command: Optional[MovementDirection] = None
    timestamp: float = 0.0


class MotorController:
    """
    High-level motor control interface.
    
    Abstracts Arduino communication and provides clean API for:
    - Movement control
    - Status monitoring
    - Health checking
    - Emergency stopping
    
    Thread-safety: Uses locks from ArduinoConnection internally.
    """

    def __init__(self, arduino: ArduinoConnection):
        """
        Initialize motor controller.
        
        Args:
            arduino: Connected ArduinoConnection instance
        """
        self._arduino = arduino
        self._current_direction: Optional[MovementDirection] = None
        self._last_movement_time: float = 0.0
        self._is_healthy = False
        self._last_status: Optional[MotorStatus] = None

    def initialize(self) -> bool:
        """
        Initialize motor system and verify health.
        
        Should be called during robot startup sequence.
        
        Returns:
            True if motors initialized and healthy, False otherwise
        """
        logger.info("Initializing motor controller...")
        
        try:
            # Verify Arduino is connected
            if not self._arduino.is_connected():
                logger.error("Arduino not connected")
                self._is_healthy = False
                return False

            # Ensure motors are stopped
            if not self.stop():
                logger.error("Failed to stop motors during init")
                return False

            # Check motor health
            if not self._arduino.check_motors():
                logger.error("Motor health check failed")
                self._is_healthy = False
                return False

            # Get and verify initial status
            status = self.get_status()
            if status is None:
                logger.error("Failed to get motor status")
                self._is_healthy = False
                return False

            if not status.is_healthy:
                logger.error(f"Motors reported as unhealthy: {status}")
                self._is_healthy = False
                return False

            logger.info(f"✓ Motor controller initialized. Status: {status}")
            self._is_healthy = True
            return True

        except Exception as e:
            logger.error(f"Motor initialization failed: {e}")
            self._is_healthy = False
            return False

    def is_healthy(self) -> bool:
        """Check if motors are in healthy state."""
        return self._is_healthy

    def move(self, direction: MovementDirection) -> bool:
        """
        Execute movement in specified direction.
        
        Args:
            direction: MovementDirection enum value
            
        Returns:
            True if command sent successfully
        """
        if not self._arduino.is_connected():
            logger.error("Cannot move: Arduino not connected")
            return False

        if not self._is_healthy:
            logger.warning("Motors are not healthy, attempting move anyway")

        try:
            success = self._arduino.move(direction.value)
            
            if success:
                self._current_direction = direction
                self._last_movement_time = time.time()
                logger.info(f"Move command: {direction.value}")
            else:
                logger.warning(f"Move command failed: {direction.value}")
                self._current_direction = None

            return success

        except Exception as e:
            logger.error(f"Move operation failed: {e}")
            self._current_direction = None
            return False

    def stop(self) -> bool:
        """
        Stop all motor movement immediately.
        
        Should be used for emergency stops.
        
        Returns:
            True if stop command sent
        """
        logger.info("STOP command")
        return self.move(MovementDirection.STOP)

    def get_current_direction(self) -> Optional[MovementDirection]:
        """Get the last commanded direction."""
        return self._current_direction

    def get_status(self) -> Optional[MotorStatus]:
        """
        Query current motor status from Arduino.
        
        Returns:
            MotorStatus if successful, None if query failed
        """
        try:
            hardware_status: Optional[HardwareStatus] = self._arduino.get_hardware_status()
            
            if hardware_status is None:
                logger.warning("Failed to get hardware status from Arduino")
                return None

            # Convert HardwareStatus to MotorStatus
            motor_status = MotorStatus(
                is_healthy=hardware_status.motors_ok,
                is_moving=(self._current_direction is not None
                          and self._current_direction != MovementDirection.STOP),
                left_speed=hardware_status.motor_left_speed,
                right_speed=hardware_status.motor_right_speed,
                battery_voltage=hardware_status.battery_voltage,
                temperature=hardware_status.temperature,
                last_command=self._current_direction,
                timestamp=hardware_status.timestamp,
            )

            # Update health state based on status
            if not motor_status.is_healthy:
                logger.warning("Motor status indicates unhealthy state")
                self._is_healthy = False
            
            self._last_status = motor_status
            return motor_status

        except Exception as e:
            logger.error(f"Failed to get motor status: {e}")
            return None

    def verify_movement(
        self,
        direction: MovementDirection,
        expected_speed: int = 200,
        timeout: float = 1.0,
    ) -> bool:
        """
        Verify that motors are actually moving in expected direction.
        
        Useful for detecting motor stalls or hardware failures.
        
        Args:
            direction: Expected movement direction
            expected_speed: Expected motor speed (0-255, default 200 for movement)
            timeout: How long to wait for verification
            
        Returns:
            True if movement verified, False if not moving or stalled
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_status()
            
            if status is None:
                return False
            
            # Check if moving with expected speed
            if (status.is_moving and 
                status.left_speed >= expected_speed and 
                status.right_speed >= expected_speed):
                logger.debug(f"Movement verified: {direction.value}")
                return True
            
            time.sleep(0.1)
        
        logger.warning(f"Movement verification failed for {direction.value}")
        return False

    def get_last_status(self) -> Optional[MotorStatus]:
        """Get the last queried status without querying again."""
        return self._last_status

    def reset_health_check(self) -> bool:
        """
        Reset health status and re-verify motors.
        
        Useful for recovery after temporary failures.
        
        Returns:
            True if motors verified healthy
        """
        logger.info("Resetting motor health check...")
        self._is_healthy = False
        
        if not self._arduino.is_connected():
            logger.error("Arduino not connected during health reset")
            return False

        if self._arduino.check_motors():
            self._is_healthy = True
            logger.info("✓ Motor health verified after reset")
            return True
        else:
            logger.error("Motor health check failed during reset")
            return False


class MotorInitializationError(Exception):
    """Raised when motor initialization fails."""
    pass
