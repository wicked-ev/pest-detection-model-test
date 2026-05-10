"""
Health check service - System diagnostics and startup verification.

Manages the startup sequence and ongoing health monitoring:

Startup sequence (CHECKING_SYSTEMS state):
1. Verify Arduino connection
2. Check motors respond
3. Check camera available
4. Load AI model
5. Verify motor response to test command

Recovery logic:
- Retry failed checks with exponential backoff
- Log all failures for debugging
- Transition to ERROR state with details if unrecoverable
- Allow manual recovery attempts

Ongoing monitoring:
- Periodic health checks during operation
- Watch for sensor/connection degradation
- Alert state machine of critical failures
"""

import logging
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HealthCheckStatus(Enum):
    """Result of a health check."""
    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: HealthCheckStatus
    message: str
    duration_ms: float
    timestamp: float = 0.0


class HealthCheckService:
    """
    Manages robot health checking and startup diagnostics.
    
    Intended to be used during startup (CHECKING_SYSTEMS state)
    and can be called periodically for ongoing monitoring.
    
    Dependencies injected to allow testing and flexibility:
    - Arduino connection
    - Motor controller
    - Camera interface
    - AI model loader
    
    Design: Each health check is independent and can be retried.
    Failures are logged with context for debugging.
    """

    def __init__(self):
        """Initialize health check service."""
        self._results: Dict[str, HealthCheckResult] = {}
        self._last_check_time: Optional[float] = None

    def check_arduino_connection(
        self,
        arduino,
        timeout: float = 5.0,
    ) -> HealthCheckResult:
        """
        Verify Arduino connection with PING test.
        
        Args:
            arduino: ArduinoConnection instance
            timeout: How long to wait for response
            
        Returns:
            HealthCheckResult with OK or ERROR status
        """
        start_time = time.time()
        check_name = "Arduino Connection"

        try:
            if not arduino.is_connected():
                duration = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.ERROR,
                    message="Arduino not connected",
                    duration_ms=duration,
                    timestamp=time.time(),
                )
                logger.error(f"✗ {check_name}: Not connected")
                self._results[check_name] = result
                return result

            # Send PING command
            if arduino.ping():
                duration = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.OK,
                    message="Arduino responding to PING",
                    duration_ms=duration,
                    timestamp=time.time(),
                )
                logger.info(f"✓ {check_name}: OK ({duration:.1f}ms)")
                self._results[check_name] = result
                return result
            else:
                duration = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.ERROR,
                    message="Arduino PING timeout",
                    duration_ms=duration,
                    timestamp=time.time(),
                )
                logger.error(f"✗ {check_name}: PING timeout")
                self._results[check_name] = result
                return result

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=check_name,
                status=HealthCheckStatus.ERROR,
                message=f"Exception: {str(e)}",
                duration_ms=duration,
                timestamp=time.time(),
            )
            logger.error(f"✗ {check_name}: {str(e)}")
            self._results[check_name] = result
            return result

    def check_motors(
        self,
        motors,
    ) -> HealthCheckResult:
        """
        Verify motors are healthy and responsive.
        
        Args:
            motors: MotorController instance
            
        Returns:
            HealthCheckResult with health status
        """
        start_time = time.time()
        check_name = "Motor System"

        try:
            if not motors.initialize():
                duration = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.ERROR,
                    message="Motor initialization failed",
                    duration_ms=duration,
                    timestamp=time.time(),
                )
                logger.error(f"✗ {check_name}: Initialization failed")
                self._results[check_name] = result
                return result

            # Get status
            status = motors.get_status()
            if status is None:
                duration = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.ERROR,
                    message="Failed to get motor status",
                    duration_ms=duration,
                    timestamp=time.time(),
                )
                logger.error(f"✗ {check_name}: Status query failed")
                self._results[check_name] = result
                return result

            if not status.is_healthy:
                duration = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.ERROR,
                    message=f"Motors unhealthy - L:{status.left_speed} R:{status.right_speed}",
                    duration_ms=duration,
                    timestamp=time.time(),
                )
                logger.error(f"✗ {check_name}: Unhealthy")
                self._results[check_name] = result
                return result

            duration = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=check_name,
                status=HealthCheckStatus.OK,
                message=(
                    f"Motors OK - L:{status.left_speed} R:{status.right_speed} "
                    f"V:{status.battery_voltage:.1f}V T:{status.temperature:.1f}°C"
                ),
                duration_ms=duration,
                timestamp=time.time(),
            )
            logger.info(f"✓ {check_name}: OK ({duration:.1f}ms)")
            self._results[check_name] = result
            return result

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=check_name,
                status=HealthCheckStatus.ERROR,
                message=f"Exception: {str(e)}",
                duration_ms=duration,
                timestamp=time.time(),
            )
            logger.error(f"✗ {check_name}: {str(e)}")
            self._results[check_name] = result
            return result

    def check_camera(
        self,
        camera_available_fn: Optional[Callable[[], bool]] = None,
    ) -> HealthCheckResult:
        """
        Verify camera is accessible.
        
        Args:
            camera_available_fn: Callable that returns True if camera available
            
        Returns:
            HealthCheckResult with camera status
        """
        start_time = time.time()
        check_name = "Camera"

        try:
            if camera_available_fn is None:
                # Placeholder: assume camera available for now
                logger.warning(f"⚠ {check_name}: No check function provided")
                duration = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.SKIPPED,
                    message="Camera check not implemented",
                    duration_ms=duration,
                    timestamp=time.time(),
                )
                self._results[check_name] = result
                return result

            if camera_available_fn():
                duration = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.OK,
                    message="Camera available and responsive",
                    duration_ms=duration,
                    timestamp=time.time(),
                )
                logger.info(f"✓ {check_name}: OK ({duration:.1f}ms)")
                self._results[check_name] = result
                return result
            else:
                duration = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.ERROR,
                    message="Camera not available or not responding",
                    duration_ms=duration,
                    timestamp=time.time(),
                )
                logger.error(f"✗ {check_name}: Not available")
                self._results[check_name] = result
                return result

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=check_name,
                status=HealthCheckStatus.ERROR,
                message=f"Exception: {str(e)}",
                duration_ms=duration,
                timestamp=time.time(),
            )
            logger.error(f"✗ {check_name}: {str(e)}")
            self._results[check_name] = result
            return result

    def check_ai_model(
        self,
        model_path: str,
        model_load_fn: Optional[Callable[[], bool]] = None,
    ) -> HealthCheckResult:
        """
        Verify AI detection model exists and loads.
        
        Args:
            model_path: Path to model file
            model_load_fn: Callable to load and test model
            
        Returns:
            HealthCheckResult with model status
        """
        start_time = time.time()
        check_name = "AI Detection Model"

        try:
            if model_load_fn is None:
                logger.warning(f"⚠ {check_name}: No load function provided")
                duration = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.SKIPPED,
                    message="Model check not implemented",
                    duration_ms=duration,
                    timestamp=time.time(),
                )
                self._results[check_name] = result
                return result

            if model_load_fn():
                duration = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.OK,
                    message=f"Model loaded from {model_path}",
                    duration_ms=duration,
                    timestamp=time.time(),
                )
                logger.info(f"✓ {check_name}: OK ({duration:.1f}ms)")
                self._results[check_name] = result
                return result
            else:
                duration = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.ERROR,
                    message=f"Failed to load model from {model_path}",
                    duration_ms=duration,
                    timestamp=time.time(),
                )
                logger.error(f"✗ {check_name}: Load failed")
                self._results[check_name] = result
                return result

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=check_name,
                status=HealthCheckStatus.ERROR,
                message=f"Exception: {str(e)}",
                duration_ms=duration,
                timestamp=time.time(),
            )
            logger.error(f"✗ {check_name}: {str(e)}")
            self._results[check_name] = result
            return result

    def get_results(self) -> Dict[str, HealthCheckResult]:
        """Get all health check results."""
        return self._results.copy()

    def get_result(self, check_name: str) -> Optional[HealthCheckResult]:
        """Get result for a specific health check."""
        return self._results.get(check_name)

    def is_all_healthy(self) -> bool:
        """
        Check if all critical systems are healthy.
        
        Skipped checks don't affect overall status.
        Any ERROR means unhealthy.
        """
        for result in self._results.values():
            if result.status == HealthCheckStatus.ERROR:
                return False
        return True

    def get_error_summary(self) -> str:
        """Get summary of all failures for error reporting."""
        errors = [
            f"{r.name}: {r.message}"
            for r in self._results.values()
            if r.status == HealthCheckStatus.ERROR
        ]
        return "; ".join(errors) if errors else "No errors"

    def clear_results(self) -> None:
        """Clear all previous check results."""
        self._results.clear()
        logger.debug("Cleared health check results")
