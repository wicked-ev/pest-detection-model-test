"""
Architecture Documentation - Phase 1

This document explains the Phase 1 implementation and how components interact.

## Module Overview

### 1. Hardware Layer (hardware/)

#### arduino.py - Serial Communication

- **Responsibility**: Low-level serial protocol with Arduino Uno
- **Key Classes**:
  - `ArduinoConnection`: Manages serial connection with retry logic
  - `ArduinoCommand`: Enum of valid commands sent to Arduino
  - `ArduinoResponse`: Enum of responses from Arduino
  - `HardwareStatus`: Dataclass representing motor/sensor status
- **Design Patterns**:
  - Thread-safe command/response with queue-based reader
  - Automatic connection retry with exponential backoff
  - Background reader thread to continuously parse serial responses
- **Key Methods**:
  - `connect()`: Establish connection with retry logic
  - `send_command()`: Send command and wait for acknowledgment
  - `ping()`: Test connection
  - `move()`: Send movement command
  - `get_hardware_status()`: Query full sensor status
- **Arduino Protocol**:
  - Commands: "PING", "CHECK", "MOVE\_\*", "STOP", "GET_STATUS"
  - Responses: "OK", "ERROR", "MOTORS_OK", "PONG"
  - Status format: "STATUS:motor_ok,left_speed,right_speed,voltage,temp"

#### motors.py - Motor Abstraction

- **Responsibility**: High-level motor control abstraction
- **Key Classes**:
  - `MotorController`: Clean API for movement commands
  - `MotorStatus`: Motor system status
  - `MovementDirection`: Enum of valid directions
- **Design Patterns**:
  - Hardware abstraction: Hide Arduino protocol details
  - Health tracking: Maintain motor health state
  - Verification: Test that motors actually move
- **Key Methods**:
  - `initialize()`: Startup sequence with health check
  - `move(direction)`: Send movement command
  - `stop()`: Emergency stop
  - `get_status()`: Query current motor status
  - `verify_movement()`: Confirm motors are responding
  - `reset_health_check()`: Recovery after temporary failure

### 2. State Management (states/)

#### state_machine.py - Robot Behavior Orchestration

- **Responsibility**: Manage robot operational states and transitions
- **Valid States**:
  - BOOTING: Initial startup
  - CHECKING_SYSTEMS: Running diagnostics
  - READY: All systems OK, awaiting commands
  - REMOTE_CONTROL: Responding to server commands
  - AUTONOMOUS: Executing predefined route
  - DETECTING: Responding to object detection
  - ERROR: System failure state
  - SHUTDOWN: Terminal state
- **Design Patterns**:
  - State pattern: Strict state transitions
  - Observer pattern: Callbacks on state change
  - Immutable state graph: Transitions defined at class level
- **Key Methods**:
  - `transition_to()`: Attempt state change (guarded)
  - `get_current_state()`: Query current state
  - `is_ready()`: Check if robot can accept commands
  - `can_move()`: Check if movement is allowed
  - `register_state_change_callback()`: Listen for transitions
  - `get_transition_history()`: Debug previous transitions
- **Thread Safety**:
  - All state access protected by lock
  - Callbacks invoked outside lock to prevent deadlock

### 3. Services (services/)

#### health_service.py - System Diagnostics

- **Responsibility**: Run startup checks and monitor system health
- **Key Classes**:
  - `HealthCheckService`: Orchestrates health checks
  - `HealthCheckResult`: Result of single check
  - `HealthCheckStatus`: OK / WARNING / ERROR / SKIPPED
- **Design Patterns**:
  - Independent checks: Each can be retried separately
  - Dependency injection: Pass components to check
  - Results collection: Aggregate all check results
- **Check Methods**:
  - `check_arduino_connection()`: PING test with timeout
  - `check_motors()`: Motor health verification
  - `check_camera()`: Camera availability (placeholder)
  - `check_ai_model()`: Model loading test (placeholder)
- **Key Features**:
  - Timing information for each check
  - Error messages for diagnostics
  - Status aggregation
  - Recovery capability

### 4. Application Entry Point (app.py)

#### RobotApplication Class

- **Responsibility**: Orchestrate startup sequence and lifecycle
- **Lifecycle Phases**:
  1. BOOTING: Initialize components
  2. CONNECTING: Establish hardware connections
  3. CHECKING_SYSTEMS: Run all diagnostics
  4. READY: Transition to operational state
- **Key Methods**:
  - `startup()`: Execute startup sequence
  - `run()`: Main operating loop
  - `shutdown()`: Graceful termination
  - `print_diagnostics()`: Debug information
- **Responsibilities**:
  - Initialize logging
  - Create hardware components
  - Execute startup sequence
  - Handle signals (Ctrl+C)
  - Manage graceful shutdown

### 5. Utilities (utils/)

#### logger.py - Structured Logging

- **Responsibility**: Consistent logging configuration
- **Features**:
  - Console and file output
  - Rotating file handler (10MB max, 5 backups)
  - Consistent format with timestamps
  - Configurable log level

### 6. Configuration (configs.py)

- **Responsibility**: Centralized configuration
- **Sections**:
  - Arduino: Port, baudrate, timeouts
  - Motors: Speed, verification parameters
  - Camera: Device, frame processing
  - AI Model: Path, thresholds
  - Network: Server URL, retries
  - Operation: Movement timeout, routes
  - Logging: Level, directory
- **Pattern**: Single source of truth for all settings
- **Environment Variables**: Loads from .env for secrets

## Startup Sequence Flow

```
main()
  └─ RobotApplication()
      │
      ├─ setup_logging()
      │
      └─ startup()
          │
          ├─ BOOTING state
          │
          ├─ Arduino.connect() ──────┐
          │                           │ Hardware
          ├─ MotorController(arduino) │ Initialization
          │                           │
          └─ CHECKING_SYSTEMS state
              │
              ├─ HealthService.check_arduino_connection()
              ├─ HealthService.check_motors()
              ├─ HealthService.check_camera()
              └─ HealthService.check_ai_model()
                  │
                  ├─ All checks PASSED
                  │   └─ transition to READY state ✓
                  │
                  └─ Any check FAILED
                      └─ transition to ERROR state ✗
```

## Component Interaction Example

### Example 1: Moving Forward

```
1. app.py calls: motors.move(MovementDirection.FORWARD)
2. motors.py calls: arduino.move("forward")
3. arduino.py:
   - Sends "MOVE_FORWARD\n" to serial port
   - Waits for "OK" response in queue (background reader thread)
   - Returns True if received
4. motors.py:
   - Updates internal state: _current_direction
   - Records last movement time
   - Returns True to caller
5. app.py:
   - Robot now moving forward
```

### Example 2: State Transition

```
1. app.py detects all health checks passed
2. app.py calls: state_machine.transition_to(RobotState.READY, reason="...")
3. state_machine.py:
   - Validates transition is in VALID_TRANSITIONS graph
   - Acquires lock
   - Updates _current_state
   - Records transition in history
   - Releases lock
4. state_machine.py calls all registered callbacks:
   - Detection service could enable here
   - Network service could enable listening here
   - Logging service records transition
5. app.py:
   - Robot now in READY state
   - Can accept movement commands
```

### Example 3: Health Check

```
1. app.py calls: health_service.check_motors(motors)
2. health_service.py calls: motors.initialize()
3. motors.py:
   - Calls: arduino.check_motors()
   - arduino.py sends "CHECK" command
   - Waits for "MOTORS_OK" response
4. motors.py calls: get_status()
   - arduino.py queries "GET_STATUS"
   - Parses hardware status
5. health_service.py:
   - Records result with timing
   - Returns HealthCheckResult
6. app.py:
   - If OK: continues to next check
   - If ERROR: transitions to ERROR state with details
```

## Thread Safety

### ArduinoConnection

- Uses `_connection_lock` to protect serial access
- Background reader thread continuously reads responses into queue
- Command/response synchronized via queue + timeout

### StateMachine

- Uses `_state_lock` to protect state access
- Callbacks invoked outside lock
- History maintained as append-only list

### MotorController

- Uses locks inherited from ArduinoConnection
- Maintains local state (\_current_direction, \_is_healthy)
- No explicit locking needed (read-only after init)

## Error Handling Strategy

### Connection Errors

- Retry with configurable attempts and delay
- Log each attempt with details
- Fail to ERROR state if exhausted

### Health Check Failures

- Critical failures (Arduino, Motors): Stop startup
- Non-critical failures (Camera, Model): Log warning but continue
- Can retry individual checks after transition to ERROR state

### Timeout Errors

- Arduino commands: 2s timeout, retry logic
- Motor verification: 1s timeout per check
- Startup: 30s timeout for all checks

## Testing Strategy

Each module can be tested independently:

```python
# Test Arduino communication
arduino = ArduinoConnection("/dev/ttyACM0")
assert arduino.connect()
assert arduino.ping()

# Test motor control
motors = MotorController(arduino)
assert motors.initialize()
motors.move(MovementDirection.FORWARD)
status = motors.get_status()

# Test state machine
sm = StateMachine()
assert sm.transition_to(RobotState.CHECKING_SYSTEMS)
assert sm.is_in_state(RobotState.CHECKING_SYSTEMS)

# Test health service
health = HealthCheckService()
result = health.check_arduino_connection(arduino)
assert result.status == HealthCheckStatus.OK
```

## Next Phases (Phase 2+)

### Phase 2: Event System & WebSocket

- Event bus for inter-component communication
- WebSocket client for remote control
- Network telemetry sender

### Phase 3: Detection Service

- Background detection thread/asyncio
- Detection event emission
- Movement interruption on detection

### Phase 4: Autonomous Mode

- Route executor service
- Continuous detection during autonomous
- Return-to-base on failure

## Design Principles Applied

1. **Single Responsibility**: Each class has one reason to change
2. **Dependency Injection**: Components injected, not created
3. **Fail Fast**: Errors caught and logged immediately
4. **Type Hints**: Full type annotations for IDE support
5. **Documentation**: Docstrings on all public methods
6. **Testability**: Each component works independently
7. **Logging**: Structured logging at all levels
8. **Configuration**: Externalized, not hardcoded
9. **State Explicitness**: Behavior driven by state machine
10. **Clean Interfaces**: Simple, predictable APIs
    """
