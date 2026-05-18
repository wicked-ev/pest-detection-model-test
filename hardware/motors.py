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


class MotorID(Enum):
    """Individual motor identifiers."""
    MOTOR_0 = 0
    MOTOR_1 = 1
    MOTOR_2 = 2
    MOTOR_3 = 3


@dataclass
class MotorStatus:
    """Current status of motor system."""
    is_healthy: bool
    is_moving: bool
    
    # Individual motor speeds and directions (0-255)
    motor_0_speed: int
    motor_0_direction: int
    motor_1_speed: int
    motor_1_direction: int
    motor_2_speed: int
    motor_2_direction: int
    motor_3_speed: int
    motor_3_direction: int
    
    # Pump motor
    pump_speed: int
    pump_direction: int
    
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
                self._is_healthy = False
                return False

            self._current_direction = MovementDirection.STOP

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

        if direction != MovementDirection.STOP and not self._is_healthy:
            logger.error("Cannot move: motor system is unhealthy")
            return False

        try:
            success = self._arduino.move(direction.value)
            
            if success:
                self._current_direction = direction
                self._last_movement_time = time.time()
                logger.info(f"Move command: {direction.value}")
            else:
                logger.warning(f"Move command failed: {direction.value}")
                if direction != MovementDirection.STOP:
                    self._current_direction = MovementDirection.STOP
                else:
                    self._current_direction = None

            return success

        except Exception as e:
            logger.error(f"Move operation failed: {e}")
            self._current_direction = MovementDirection.STOP
            return False

    def stop(self) -> bool:
        """
        Stop all motor movement immediately.
        
        Should be used for emergency stops.
        
        Returns:
            True if stop command sent
        """
        logger.info("STOP command")
        success = self._arduino.stop_all()
        if success:
            self._current_direction = MovementDirection.STOP
            self._last_movement_time = time.time()
        else:
            logger.error("Emergency stop failed")
        return success

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

            # Check if any motor is moving
            any_moving = (hardware_status.motor_0_speed > 0 or
                         hardware_status.motor_1_speed > 0 or
                         hardware_status.motor_2_speed > 0 or
                         hardware_status.motor_3_speed > 0)

            # Convert HardwareStatus to MotorStatus
            motor_status = MotorStatus(
                is_healthy=hardware_status.motors_ok,
                is_moving=any_moving,
                motor_0_speed=hardware_status.motor_0_speed,
                motor_0_direction=hardware_status.motor_0_direction,
                motor_1_speed=hardware_status.motor_1_speed,
                motor_1_direction=hardware_status.motor_1_direction,
                motor_2_speed=hardware_status.motor_2_speed,
                motor_2_direction=hardware_status.motor_2_direction,
                motor_3_speed=hardware_status.motor_3_speed,
                motor_3_direction=hardware_status.motor_3_direction,
                pump_speed=hardware_status.pump_speed,
                pump_direction=hardware_status.pump_direction,
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
            
            # Check if all motors are moving with expected speed
            all_moving = (status.motor_0_speed >= expected_speed and
                         status.motor_1_speed >= expected_speed and
                         status.motor_2_speed >= expected_speed and
                         status.motor_3_speed >= expected_speed)
            
            if status.is_moving and all_moving:
                logger.debug(f"Movement verified: {direction.value}")
                return True
            
            time.sleep(0.1)
        
        logger.warning(f"Movement verification failed for {direction.value}")
        return False

    def get_last_status(self) -> Optional[MotorStatus]:
        """Get the last queried status without querying again."""
        return self._last_status

    def set_motor(self, motor_id: int, speed: int, direction: int) -> bool:
        """
        Set speed and direction for a specific motor.
        
        Args:
            motor_id: 0-3 (motor identifier)
            speed: 0-255 (PWM speed)
            direction: 0 (backward) or 1 (forward)
            
        Returns:
            True if command sent successfully
        """
        if motor_id < 0 or motor_id > 3:
            logger.error(f"Invalid motor ID: {motor_id}")
            return False

        if not self._arduino.is_connected():
            logger.error("Cannot set motor: Arduino not connected")
            return False

        try:
            success = self._arduino.set_motor(motor_id, speed, direction)
            if success:
                logger.info(f"Motor {motor_id} set to speed {speed}, direction {direction}")
            else:
                logger.warning(f"Failed to set motor {motor_id}")
            return success
        except Exception as e:
            logger.error(f"Set motor {motor_id} failed: {e}")
            return False

    def set_pump(self, speed: int, direction: int) -> bool:
        """
        Set speed and direction for the pump motor.
        
        Args:
            speed: 0-255 (PWM speed)
            direction: 0 (backward/off) or 1 (forward/on)
            
        Returns:
            True if command sent successfully
        """
        if not self._arduino.is_connected():
            logger.error("Cannot set pump: Arduino not connected")
            return False

        try:
            success = self._arduino.set_pump(speed, direction)
            if success:
                logger.info(f"Pump set to speed {speed}, direction {direction}")
            else:
                logger.warning("Failed to set pump")
            return success
        except Exception as e:
            logger.error(f"Set pump failed: {e}")
            return False

    def get_motor_speed(self, motor_id: int) -> Optional[int]:
        """Get the last reported speed for a specific motor (0-255)."""
        if self._last_status is None:
            return None
        
        if motor_id == 0:
            return self._last_status.motor_0_speed
        elif motor_id == 1:
            return self._last_status.motor_1_speed
        elif motor_id == 2:
            return self._last_status.motor_2_speed
        elif motor_id == 3:
            return self._last_status.motor_3_speed
        else:
            return None

    def get_pump_speed(self) -> Optional[int]:
        """Get the last reported pump speed (0-255)."""
        if self._last_status is None:
            return None
        return self._last_status.pump_speed

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
            status = self.get_status()
            if status and status.is_healthy:
                self._is_healthy = True
                logger.info("✓ Motor health verified after reset")
                return True

        logger.error("Motor health check failed during reset")
        return False

    def shutdown(self) -> None:
        """Stop the motors and release any motor resources."""
        if self._arduino.is_connected():
            self.stop()


class MotorInitializationError(Exception):
    """Raised when motor initialization fails."""
    pass
