# Communication Protocols Overview

This document defines the **wire‑level protocols** used by the pest‑detection robot system.
It covers two communication channels:

1. **Server ↔ Raspberry Pi** – WebSocket / HTTP JSON messages.
2. **Raspberry Pi ↔ Arduino** – Serial command/response protocol.

---

## 1. Server ↔ Raspberry Pi

The robot connects to a remote control server using **WebSocket** endpoints defined in `NetworkService`.

### Control WebSocket (`/ws/control/<robot_id>`)

| Direction | Message Type | JSON Schema | Description |
|----------|-------------|------------|-------------|
| **Pi → Server** | `heartbeat` | `{ "type": "heartbeat", "state": "<RobotState>", "timestamp": <float> }` | Periodic status ping (configurable interval). |
| **Server → Pi** | `command` | `{ "type": "command", "command": "move", "direction": "forward|backward|left|right|stop", "toggle": <bool> }` | Remote movement instruction. |
| **Server → Pi** | `start_stream` / `stop_stream` | `{ "type": "start_stream" }` or `{ "type": "stop_stream" }` | Control AI inference streaming. |
| **Server → Pi** | `shutdown` | `{ "type": "shutdown" }` | Request graceful shutdown. |
| **Pi → Server** | `status` | See below. |
| **Pi → Server** | `detection` | See below. |

### Status Message (`type: "status"`)
```json
{
  "type": "status",
  "state": "READY|BOOTING|ERROR|...",
  "emergency": false,
  "stream_enabled": true,
  "streaming": false,
  "stream_mode": "local|remote",
  "motor_health": true,
  "motors_moving": false,
  "motor_speeds": [0,0,0,0]
}
```
Sent on demand (e.g., when the server sends `status` command) or periodically for monitoring.

### Detection Message (`type: "detection"`)
```json
{
  "type": "detection",
  "detections": [
    { "class": "pest", "bbox": [x, y, w, h], "confidence": 0.92 }
  ],
  "state": "DETECTING"
}
```
Emitted by the Pi when the model service produces new detections. The callback is registered via `NetworkService.set_stream_detection_callback`.

---

## 2. Raspberry Pi ↔ Arduino (Serial Protocol)

Implemented in **`hardware/arduino.py`** (`ArduinoConnection`). All messages are newline‑terminated ASCII strings.

### Command Format
```
<COMMAND> [ARG1] [ARG2] ...\n
```
Supported commands (`ArduinoCommand` enum):
- `PING` – health check, expects `PONG`.
- `CHECK` – motor health verification, expects `MOTORS_OK`.
- `SET_MOTOR <id> <speed> <direction>` – control individual motor (speed 0‑255, direction 0/1).
- `SET_PUMP <speed> <direction>` – pump control.
- `GET_STATUS` – returns `STATUS:<csv>` where CSV contains motor speeds/directions and pump data.
- `STOP_ALL` – emergency stop of all actuators.
- `STOP` – stop movement (alias for STOP_ALL in some contexts).

### Response Format
Responses are plain strings (may be prefixed with an enum name):
- `OK` – generic success.
- `ERROR <msg>` – generic error.
- `MOTORS_OK` / `MOTORS_ERROR` – motor health.
- `PONG` – ping reply.
- `STATUS:<motor0_speed>,<motor0_dir>,...,<pump_dir>` – status CSV.

All responses are **queued** in `ArduinoConnection._response_queue` and parsed by the reader thread.

### Timing & Reliability
- **Command timeout**: configurable via `ArduinoConnection.timeout` (default 2 s).
- **Retry logic**: `NetworkService` will retry WS connections up to `server_connection_attempts` (default 25) with `connection_retry_delay`.
- **Auto‑reconnect** on serial disconnect is handled by a background reconnect thread (`_reconnect_loop`).

---

## 3. Message Flow Examples

### 3.1. Server → Pi → Arduino (Move Forward)
1. Server sends WebSocket `command` JSON:
   ```json
   {"type": "command", "command": "move", "direction": "forward"}
   ```
2. `RobotApplication._process_remote_command` maps to `MotorController.move` → `ArduinoConnection.set_motor` calls:
   `SET_MOTOR 0 200 1` and `SET_MOTOR 1 200 1` (plus motors 2/3).
3. Arduino replies `OK` for each motor; the Pi aggregates success and updates state.

### 3.2. Pi → Server (Heartbeat)
- Every `heartbeat_interval` seconds, `NetworkService._control_heartbeat_loop` sends:
  ```json
  {"type":"heartbeat","state":"READY","timestamp":1697201234.12}
  ```

### 3.3. Detection Streaming (Local)
1. Pi captures frame → `ModelService.start_streaming` calls `CameraService` → `NetworkService.send_message` not used.
2. When detections are ready, `ModelService` invokes the registered callback (`NetworkService._stream_detection_callback`).
3. Callback builds detection payload (see Detection Message) and sends via `NetworkService.send_message`.

---

## 4. Extensibility Guidelines
- **Add new server commands**: extend the `command` parsing block in `app.py` and define a matching JSON schema.
- **Add new Arduino commands**: add to `ArduinoCommand` enum, implement handling in `ArduinoConnection.send_command`, and ensure the firmware (`arduino_controller.ino`) acknowledges it.
- **Versioning**: include a `protocol_version` field in every top‑level JSON message to allow future backward compatibility.

---

*Document generated on 2026‑06‑13.*
