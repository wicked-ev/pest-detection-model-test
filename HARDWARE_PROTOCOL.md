# Hardware Protocol Documentation

## System Architecture

### Hardware Components

- **Raspberry Pi**: Main controller
- **Arduino Uno**: Motor driver interface
- **3 Motor Drivers**:
  - Driver 1: 2 motors (Motor 0, Motor 1) - Movement
  - Driver 2: 2 motors (Motor 2, Motor 3) - Movement
  - Driver 3: 1 motor (Pump) - Auxiliary/Fluid handling

### Serial Communication

- **Port**: `/dev/ttyUSB0` (configurable in `configs.py`)
- **Baud Rate**: 9600 bps
- **Protocol**: Line-delimited ASCII commands and responses
- **Timeout**: 2.0 seconds for command responses
- **Failsafe**: 6000ms timeout triggers `STOP_ALL` if no command received while motors running

## Pin Mapping

### Motor Driver 1 (Motors 0 & 1)

| Motor | Signal | Pin | Type    | Function              |
| ----- | ------ | --- | ------- | --------------------- |
| 0     | SPEED  | D3  | PWM     | Motor 0 speed (0-255) |
| 0     | IN1    | A0  | Digital | Motor 0 direction A   |
| 0     | IN2    | A1  | Digital | Motor 0 direction B   |
| 1     | SPEED  | D5  | PWM     | Motor 1 speed (0-255) |
| 1     | IN3    | A2  | Digital | Motor 1 direction A   |
| 1     | IN4    | A3  | Digital | Motor 1 direction B   |

### Motor Driver 2 (Motors 2 & 3)

| Motor | Signal | Pin | Type    | Function              |
| ----- | ------ | --- | ------- | --------------------- |
| 2     | SPEED  | D6  | PWM     | Motor 2 speed (0-255) |
| 2     | IN1    | A4  | Digital | Motor 2 direction A   |
| 2     | IN2    | A5  | Digital | Motor 2 direction B   |
| 3     | SPEED  | D9  | PWM     | Motor 3 speed (0-255) |
| 3     | IN3    | D2  | Digital | Motor 3 direction A   |
| 3     | IN4    | D4  | Digital | Motor 3 direction B   |

### Motor Driver 3 (Pump)

| Motor | Signal    | Pin | Type    | Function             |
| ----- | --------- | --- | ------- | -------------------- |
| Pump  | SPEED     | D10 | PWM     | Pump speed (0-255)   |
| Pump  | DIRECTION | D7  | Digital | Pump direction (0/1) |

**Note**: Analog pins (A0-A5) are configured as digital outputs for direction control to maximize available PWM pins on the Arduino.

## Serial Protocol

### Command Format

All commands are line-delimited (end with `\n`) ASCII strings.

### Supported Commands

#### PING

**Purpose**: Echo/keep-alive command

**Request**: `PING\n`

**Response**: `PONG\n`

---

#### CHECK

**Purpose**: Verify motor system is operational

**Request**: `CHECK\n`

**Response**: `MOTORS_OK\n`

---

#### SET_MOTOR

**Purpose**: Control speed and direction of individual motors (0-3)

**Request**: `SET_MOTOR <id> <speed> <direction>\n`

| Parameter | Range  | Description           |
| --------- | ------ | --------------------- |
| id        | 0-3    | Motor identifier      |
| speed     | 0-255  | PWM speed value       |
| direction | 0 or 1 | 0=backward, 1=forward |

**Response**: `OK\n` on success, `ERROR:*` on failure

**Example**: `SET_MOTOR 0 200 1\n` → Set Motor 0 to speed 200, forward direction

---

#### SET_PUMP

**Purpose**: Control pump speed and direction

**Request**: `SET_PUMP <speed> <direction>\n`

| Parameter | Range  | Description                  |
| --------- | ------ | ---------------------------- |
| speed     | 0-255  | PWM speed value              |
| direction | 0 or 1 | 0=backward/off, 1=forward/on |

**Response**: `OK\n` on success, `ERROR:*` on failure

**Example**: `SET_PUMP 255 1\n` → Set pump to full speed, forward

---

#### STOP_ALL

**Purpose**: Emergency stop - stops all motors and pump

**Request**: `STOP_ALL\n` or `STOP\n`

**Response**: `OK\n`

**Behavior**:

- All motor speeds set to 0
- All pump speed set to 0
- All direction pins set LOW
- Immediate (non-blocking)

---

#### GET_STATUS

**Purpose**: Query complete system status

**Request**: `GET_STATUS\n`

**Response**: `STATUS:M0_spd,M0_dir,M1_spd,M1_dir,M2_spd,M2_dir,M3_spd,M3_dir,P_spd,P_dir\n`

**Example Response**: `STATUS:200,1,200,1,180,0,180,0,255,1\n`

| Field  | Index | Description                   |
| ------ | ----- | ----------------------------- |
| M0_spd | 0     | Motor 0 current speed (0-255) |
| M0_dir | 1     | Motor 0 direction (0/1)       |
| M1_spd | 2     | Motor 1 current speed (0-255) |
| M1_dir | 3     | Motor 1 direction (0/1)       |
| M2_spd | 4     | Motor 2 current speed (0-255) |
| M2_dir | 5     | Motor 2 direction (0/1)       |
| M3_spd | 6     | Motor 3 current speed (0-255) |
| M3_dir | 7     | Motor 3 direction (0/1)       |
| P_spd  | 8     | Pump speed (0-255)            |
| P_dir  | 9     | Pump direction (0/1)          |

---

### Error Responses

| Error Response               | Cause                               |
| ---------------------------- | ----------------------------------- |
| `ERROR:UNKNOWN_COMMAND`      | Received unrecognized command       |
| `ERROR:INVALID_MOTOR_ID`     | Motor ID outside 0-3 range          |
| `ERROR:INVALID_MOTOR_PARAMS` | Malformed SET_MOTOR parameters      |
| `ERROR:INVALID_PUMP_PARAMS`  | Malformed SET_PUMP parameters       |
| `ERROR:CMD_TOO_LONG`         | Command buffer overflow (>64 chars) |
| `ERROR:FAILSAFE_STOP`        | Failsafe triggered (6000ms timeout) |

---

## Python API

### `hardware/arduino.py`

#### ArduinoConnection class

**Key Methods:**

- `connect() -> bool`: Establish serial connection to Arduino
- `disconnect()`: Close serial connection and cleanup
- `is_connected() -> bool`: Check connection status
- `ping() -> bool`: Send PING and verify PONG response
- `check_motors() -> bool`: Send CHECK and verify MOTORS_OK response
- `set_motor(motor_id, speed, direction) -> bool`: Control individual motor
- `set_pump(speed, direction) -> bool`: Control pump
- `move(direction) -> bool`: Convenience method for movement (forward/backward/left/right/stop)
- `stop_all() -> bool`: Emergency stop all motors and pump
- `get_hardware_status() -> Optional[HardwareStatus]`: Query complete status

**HardwareStatus dataclass:**

```python
@dataclass
class HardwareStatus:
    motors_ok: bool
    motor_0_speed: int
    motor_0_direction: int
    motor_1_speed: int
    motor_1_direction: int
    motor_2_speed: int
    motor_2_direction: int
    motor_3_speed: int
    motor_3_direction: int
    pump_speed: int
    pump_direction: int
    timestamp: float
```

### `hardware/motors.py`

#### MotorController class

**Key Methods:**

- `initialize() -> bool`: Initialize motor system and verify health
- `is_healthy() -> bool`: Check motor health status
- `move(direction: MovementDirection) -> bool`: Execute movement
- `stop() -> bool`: Emergency stop
- `set_motor(motor_id, speed, direction) -> bool`: Control individual motor
- `set_pump(speed, direction) -> bool`: Control pump
- `get_motor_speed(motor_id) -> Optional[int]`: Get last reported motor speed
- `get_pump_speed() -> Optional[int]`: Get last reported pump speed
- `get_status() -> Optional[MotorStatus]`: Query current motor status
- `verify_movement(direction, expected_speed=200, timeout=1.0) -> bool`: Verify motors are moving
- `reset_health_check() -> bool`: Re-verify motor health
- `shutdown()`: Gracefully stop motors

**MotorStatus dataclass:**

```python
@dataclass
class MotorStatus:
    is_healthy: bool
    is_moving: bool
    motor_0_speed: int
    motor_0_direction: int
    motor_1_speed: int
    motor_1_direction: int
    motor_2_speed: int
    motor_2_direction: int
    motor_3_speed: int
    motor_3_direction: int
    pump_speed: int
    pump_direction: int
    last_command: Optional[MovementDirection] = None
    timestamp: float = 0.0
```

---

## Movement Command Convenience API

The `MotorController.move()` method provides high-level movement control by setting multiple motors at once:

| Direction | Motors Set              | Speed | Direction |
| --------- | ----------------------- | ----- | --------- |
| FORWARD   | All 4 motors            | 200   | 1         |
| BACKWARD  | All 4 motors            | 200   | 0         |
| LEFT      | Motors 0,2 (left side)  | 175   | 0         |
| LEFT      | Motors 1,3 (right side) | 175   | 1         |
| RIGHT     | Motors 0,2 (left side)  | 175   | 1         |
| RIGHT     | Motors 1,3 (right side) | 175   | 0         |
| STOP      | All motors              | 0     | 0         |

---

## Code Changes Summary

### Files Modified

#### 1. `hardware/arduino_controller.ino`

- **Added**: Multi-motor driver support (3 drivers, 4 motors + 1 pump)
- **Added**: MotorState struct for managing motor speed/direction
- **Added**: Motor driver arrays for independent motor control
- **Added**: `setMotor()`, `setPump()`, `safeStop()` functions
- **Added**: `parseSetMotor()` command parser
- **Changed**: Serial protocol from movement directions to motor ID-based control
- **Changed**: STATUS response format from 5 values to 10 values
- **Added**: Non-blocking failsafe timeout checking (6000ms)

#### 2. `hardware/arduino.py`

- **Added**: `set_motor()` method for individual motor control
- **Added**: `set_pump()` method for pump control
- **Added**: `stop_all()` method for emergency stop
- **Modified**: `ArduinoCommand` enum - removed movement directions, added SET_MOTOR/SET_PUMP
- **Modified**: `HardwareStatus` dataclass - expanded from 5 fields to 10 (motor speeds + directions + pump)
- **Modified**: `_expected_responses_for()` - updated command-response mappings
- **Modified**: `get_hardware_status()` - updated status parsing for 10-value format
- **Modified**: `move()` - converted to use individual `set_motor()` calls with directional mappings

#### 3. `hardware/motors.py`

- **Added**: `MotorID` enum for motor identifiers
- **Added**: `set_motor()` method for individual motor control
- **Added**: `set_pump()` method for pump control
- **Added**: `get_motor_speed()` method to query individual motor speeds
- **Added**: `get_pump_speed()` method to query pump speed
- **Modified**: `MotorStatus` dataclass - expanded from 6 fields to 12 (4 motors + pump detailed status)
- **Modified**: `get_status()` - updated to handle 10-value status format
- **Modified**: `verify_movement()` - changed to verify all 4 motors instead of just left/right
- **Modified**: `stop()` - now calls `stop_all()` directly
- **Note**: `move()` method maintains backward compatibility with MovementDirection enum

---

## Migration Notes

### For Existing Code

If your code previously called:

- `motors.move(MovementDirection.FORWARD)` - ✅ Still works (uses new multi-motor implementation)
- `arduino.move("forward")` - ✅ Still works (maps to all 4 motors)

### New Capabilities

You can now control motors independently:

```python
# Control individual motors
arduino.set_motor(0, 200, 1)  # Motor 0 at speed 200, forward
motors.set_motor(1, 150, 0)   # Motor 1 at speed 150, backward

# Control pump
arduino.set_pump(255, 1)      # Pump at full speed
motors.set_pump(128, 1)       # Pump at half speed

# Query individual motor status
speed = motors.get_motor_speed(0)  # Motor 0 current speed
pump_spd = motors.get_pump_speed() # Pump current speed
```

---

## Testing Checklist

- [ ] Arduino sketch compiles and uploads successfully
- [ ] Serial connection established at 9600 baud
- [ ] PING/PONG echo working
- [ ] CHECK returns MOTORS_OK
- [ ] GET_STATUS returns 10 comma-separated values
- [ ] Motor 0 responds to SET_MOTOR 0 commands
- [ ] Motor 1 responds to SET_MOTOR 1 commands
- [ ] Motor 2 responds to SET_MOTOR 2 commands
- [ ] Motor 3 responds to SET_MOTOR 3 commands
- [ ] Pump responds to SET_PUMP commands
- [ ] STOP_ALL emergency stop works
- [ ] Forward/backward movement activates all 4 motors
- [ ] Left/right movement activates correct motor pairs
- [ ] Failsafe timeout triggers after 6000ms of inactivity
- [ ] Motor health check passes during boot

---

## Troubleshooting

### Motors not responding to commands

1. Verify Arduino is connected: `arduino.ping()`
2. Check motor health: `arduino.check_motors()`
3. Verify motor power and driver connections
4. Check pin assignments match hardware wiring
5. Test individual motor via: `arduino.set_motor(0, 100, 1)`

### Uneven motor speeds

- Check for mechanical binding
- Verify power supply delivers consistent voltage
- Check PWM pin connections to motor drivers
- Test with fixed speed commands and measure actual speed

### Failsafe triggering unexpectedly

- Check serial communication latency
- Verify Pi can send commands within 6000ms
- Consider increasing FAILSAFE_TIMEOUT_MS if system is slow

### Status response parsing errors

- Verify Arduino is sending exactly 10 comma-separated values
- Check serial connection stability
- Review Arduino sketch STATUS response format
