# Pest Detection Robot (Raspberry Pi)

Software that runs on the robot's Raspberry Pi: it reads the camera, detects pests,
drives the motors through an Arduino, and talks to the monitoring server
(the `Pest-detection-robot-server-client-side` project).

## How detection works

`DETECTION_MODE` in `.env` decides where the pest model runs:

| Mode | What happens |
|---|---|
| `auto` (default) | Use a model on the robot if a runtime and model file are available; otherwise let the server detect. |
| `local` | Only on the robot. |
| `server` | Always on the server; the robot only sends camera frames. |

Either way the server receives the live frames, so the dashboard shows the camera and
records sightings:

- **On the robot** ("monitor mode"): frames at `STREAM_FPS`, plus each detection together with
  the frame it was found in. The server runs no model.
- **On the server**: frames at `STREAM_FPS`; the server runs the model and sends the results back.

The robot never exits because the server, Wi-Fi, camera or Arduino is unavailable: it logs the
problem, keeps working with what it has, and keeps reconnecting in the background.

## Setting up the Pi

Tested target: Raspberry Pi 3B+ with the Pi camera module.

**Use pip with system packages, not `uv`, on the Pi.** Raspberry Pi OS gets ready-made ARM
builds of numpy and Pillow from piwheels, which pip uses automatically; uv does not, and then
tries to compile them on the Pi. picamera2 cannot be installed with pip at all: it comes from apt,
and the virtual environment must be allowed to see it (`--system-site-packages`).

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-numpy python3-pil python3-venv
cd ~/pest-detection-model-test-PI
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env    # then set SERVER_HOST, ROBOT_KEY, HOTSPOT_PASSWORD
.venv/bin/python app.py
```

Check which OS you have with `uname -m`: `armv7l` is 32-bit, `aarch64` is 64-bit.

### Running the model on the robot

- **32-bit OS (armv7l):** onnxruntime publishes no builds, and a TFLite runtime may not be
  available for your Python version. Without one, `auto` mode uses the server, which is fine.
- **64-bit OS (aarch64):** `pip install onnxruntime` works, and so does a TFLite runtime.
  The Pi 3B+ can run 64-bit Raspberry Pi OS.
- **Model size matters more than the runtime.** The server's RF-DETR model is far too heavy for a
  Pi 3B+. On-robot detection needs a small model made for edge devices (for example a nano-size
  YOLO at 320 px, exported to TFLite). Set `MODEL_NUM_THREADS=4` to use all four cores.

### Wi-Fi

If the robot has no network, it starts a setup hotspot (`HOTSPOT_SSID` / `HOTSPOT_PASSWORD`)
and waits for credentials as JSON (`{"ssid": "...", "password": "..."}`) on its local socket.
Saved networks live in `config/wifi_credentials.json` (not in git; see the `.example` file).
Set `WIFI_MANAGEMENT_ENABLED=false` if the OS already manages Wi-Fi.

## Running on a PC (no camera, no Arduino)

Useful for testing against the server:

```bash
CAMERA_BACKENDS=synthetic ARDUINO_PORT=simulated WIFI_MANAGEMENT_ENABLED=false \
SERVER_HOST=127.0.0.1 python app.py
```

`synthetic` draws a moving test pattern instead of using a camera, and `simulated` stands in
for the motor board, so driving and spraying can be tried from the dashboard.

## Tests

```bash
python -m pytest
```

## Driving and spraying

**First, find the wiring.** With the wheels off the ground and `app.py` stopped:

```bash
.venv/bin/python tools/hardware_test.py
```

It runs each motor and the pump briefly, asks what happened, and prints the `LEFT_MOTORS`,
`RIGHT_MOTORS`, `*_FORWARD_DIRECTION` and `PUMP_DIRECTION` lines for `.env`.

**Driving** is hold-to-drive from the dashboard (buttons, or arrows / WASD, Space stops).
The dashboard repeats the command while a key is held; if commands stop arriving for
`DRIVE_COMMAND_TIMEOUT` seconds (released key, closed tab, lost Wi-Fi) the robot stops itself.

**Spraying:** "Spray now" in the dashboard, or **auto-spray** (off at every start): when a pest is
detected with at least `AUTO_SPRAY_MIN_CONFIDENCE`, the robot stops its wheels, sprays for
`SPRAY_DURATION` seconds, and then waits `SPRAY_COOLDOWN` seconds before it may spray again.
It stays stopped afterwards. Sprays show on the video and in the dashboard history.

**Without the hardware:** `ARDUINO_PORT=simulated` uses a software stand-in for the motor board
that answers like the real firmware and logs every motor/pump change.

## Remote commands (from the dashboard)

`start_stream`, `stop_stream`, `move` (`direction`: forward/backward/left/right/stop,
`speed`: 10-100 %), `halt` (stop wheels and sprayer), `spray` (`duration` in seconds),
`set_auto_spray` (`enabled`), `reboot` (restarts this program), `shutdown` (stops this program).
