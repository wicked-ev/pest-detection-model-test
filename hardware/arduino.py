"""
Serial communication layer for Arduino Uno.

Handles low-level serial protocol with the Arduino, including:
- Connection management
- Command sending/receiving
- Response parsing
- Error handling and recovery
- Timeout management

The Arduino is responsible ONLY for:
- Motor control (PWM)
- Motor feedback
- Hardware status reporting

High-level logic remains on the Raspberry Pi.
"""

import serial
import time
import logging
from typing import Optional, Sequence
from enum import Enum
from dataclasses import dataclass
from threading import Lock, Thread
from queue import Queue, Empty, Full

logger = logging.getLogger(__name__)


class ArduinoCommand(Enum):
    """Commands sent to Arduino."""
    PING = "PING"
    CHECK = "CHECK"
    STOP = "STOP"
    STOP_ALL = "STOP_ALL"
    GET_STATUS = "GET_STATUS"
    SET_MOTOR = "SET_MOTOR"  # Parameterized: SET_MOTOR <id> <speed> <direction>
    SET_PUMP = "SET_PUMP"    # Parameterized: SET_PUMP <speed> <direction>


class ArduinoResponse(Enum):
    """Expected responses from Arduino."""
    OK = "OK"
    ERROR = "ERROR"
    MOTORS_OK = "MOTORS_OK"
    MOTORS_ERROR = "MOTORS_ERROR"
    STATUS = "STATUS"
    PING_RESPONSE = "PONG"


@dataclass
class HardwareStatus:
    """Status report from Arduino."""
    motors_ok: bool
    motor_0_speed: int      # 0-255
    motor_0_direction: int  # 0 or 1
    motor_1_speed: int      # 0-255
    motor_1_direction: int  # 0 or 1
    motor_2_speed: int      # 0-255
    motor_2_direction: int  # 0 or 1
    motor_3_speed: int      # 0-255
    motor_3_direction: int  # 0 or 1
    pump_speed: int         # 0-255
    pump_direction: int     # 0 or 1
    timestamp: float


class ArduinoCommunicationError(Exception):
    """Raised when Arduino communication fails."""
    pass


class ArduinoConnection:
    """
    Manages serial communication with Arduino Uno.
    
    Features:
    - Thread-safe command/response handling
    - Automatic reconnection logic
    - Command timeout management
    - Response buffering with queue
    - Graceful shutdown
    
    Design pattern: This wraps PySerial to provide a higher-level
    abstraction that enforces request-response semantics.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",  # Arduino serial port on Raspberry Pi
        baudrate: int = 9600,
        timeout: float = 2.0,
        write_timeout: float = 2.0,
        read_timeout: float = 0.1,
        max_reconnect_attempts: int = 10,
        reconnect_delay: float = 1.0,
    ):
        """
        Initialize Arduino connection.
        
        Args:
            port: Serial port (e.g., '/dev/ttyACM0' on Pi, 'COM3' on Windows)
            baudrate: Serial communication speed (default 9600 matches Arduino)
            timeout: Read timeout in seconds
            write_timeout: Write timeout in seconds
            max_reconnect_attempts: Max attempts before giving up
            reconnect_delay: Delay between reconnection attempts (seconds)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self._read_timeout = read_timeout
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._serial: Optional[serial.Serial] = None
        self._connection_lock = Lock()
        self._is_connected = False
        self._response_queue: Queue[str] = Queue(maxsize=100)
        self._reader_thread: Optional[Thread] = None
        self._stop_reader = False

    def _expected_responses_for(self, command: ArduinoCommand) -> Sequence[str]:
        """Return acceptable response strings for a given command."""
        mapping = {
            ArduinoCommand.PING: [ArduinoResponse.PING_RESPONSE.value],
            ArduinoCommand.CHECK: [ArduinoResponse.MOTORS_OK.value],
            ArduinoCommand.STOP: [ArduinoResponse.OK.value],
            ArduinoCommand.STOP_ALL: [ArduinoResponse.OK.value],
            ArduinoCommand.SET_MOTOR: [ArduinoResponse.OK.value],
            ArduinoCommand.SET_PUMP: [ArduinoResponse.OK.value],
        }
        return mapping.get(command, [ArduinoResponse.OK.value])

    def connect(self) -> bool:
        """
        Establish connection to Arduino.
        
        Implements retry logic with delay and clears stale serial data.
        
        Returns:
            True if connected, False if all attempts failed
        """
        for attempt in range(self.max_reconnect_attempts):
            try:
                with self._connection_lock:
                    logger.info(
                        f"Connecting to Arduino on {self.port} "
                        f"(attempt {attempt + 1}/{self.max_reconnect_attempts})"
                    )

                    self._response_queue = Queue(maxsize=100)
                    self._serial = serial.Serial(
                        port=self.port,
                        baudrate=self.baudrate,
                        timeout=self._read_timeout,
                        write_timeout=self.write_timeout,
                    )

                    # Give Arduino time to reset and settle before use
                    time.sleep(3.0)
                    self._serial.reset_input_buffer()
                    self._serial.reset_output_buffer()

                    self._is_connected = True
                    logger.info(f"✓ Connected to Arduino on {self.port}")

                    # Start background reader thread
                    self._start_reader_thread()

                    return True

            except (serial.SerialException, OSError) as e:
                logger.warning(
                    f"Connection attempt {attempt + 1} failed: {e}"
                )
                
                if attempt < self.max_reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)
                else:
                    logger.error(
                        f"Failed to connect to Arduino after "
                        f"{self.max_reconnect_attempts} attempts"
                    )
                    self._is_connected = False

        return False

    def disconnect(self) -> None:
        """Safely disconnect from Arduino."""
        with self._connection_lock:
            self._stop_reader = True

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)

        with self._connection_lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.reset_input_buffer()
                    self._serial.reset_output_buffer()
                except Exception:
                    pass
                self._serial.close()
                logger.info("Disconnected from Arduino")
            self._is_connected = False

    def is_connected(self) -> bool:
        """Check if currently connected to Arduino."""
        return self._is_connected

    def _write_message(self, message: str) -> None:
        if not self._serial or not self._serial.is_open:
            raise ArduinoCommunicationError("Serial port is not open")

        try:
            self._serial.write(message.encode())
            self._serial.flush()
        except serial.SerialException as e:
            logger.error(f"Serial write failed: {e}")
            self._is_connected = False
            raise ArduinoCommunicationError(f"Serial communication failed: {e}")

    def _wait_for_response(
        self,
        expected_responses: Sequence[str],
        timeout: float,
    ) -> Optional[str]:
        saved_responses = []
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = self._response_queue.get(timeout=0.1)
                logger.debug(f"← Received: {response}")

                if response.startswith(ArduinoResponse.ERROR.value) or response.startswith(ArduinoResponse.MOTORS_ERROR.value):
                    for saved in saved_responses:
                        self._response_queue.put(saved)
                    return response

                if any(response.startswith(prefix) for prefix in expected_responses):
                    for saved in saved_responses:
                        self._response_queue.put(saved)
                    return response

                saved_responses.append(response)
            except Empty:
                continue

        logger.warning("Timeout waiting for Arduino response")
        for saved in saved_responses:
            self._response_queue.put(saved)
        return None

    def send_command(
        self,
        command: ArduinoCommand,
        timeout: Optional[float] = None,
        expected_responses: Optional[Sequence[str]] = None,
    ) -> bool:
        """
        Send command to Arduino and wait for an expected response.
        
        Args:
            command: The command to send
            timeout: How long to wait for response (uses default if None)
            expected_responses: Optional list of accepted response prefixes
            
        Returns:
            True if an expected response is received, False if timeout/error
            
        Raises:
            ArduinoCommunicationError: If not connected or send fails
        """
        if not self._is_connected:
            raise ArduinoCommunicationError("Not connected to Arduino")

        timeout = timeout or self.timeout
        expected_responses = list(expected_responses or self._expected_responses_for(command))

        with self._connection_lock:
            message = f"{command.value}\n"
            self._write_message(message)
            logger.debug(f"→ Sent: {command.value}")

        response = self._wait_for_response(expected_responses, timeout)
        if response is None:
            logger.warning(f"Timeout waiting for Arduino response to {command.value}")
            return False

        if response.startswith(ArduinoResponse.ERROR.value) or response.startswith(ArduinoResponse.MOTORS_ERROR.value):
            logger.warning(f"Arduino returned error response for {command.value}: {response}")
            return False

        return True

    def ping(self) -> bool:
        """
        Test connection with a PING command.
        
        Returns:
            True if Arduino responds with PONG
        """
        try:
            return self.send_command(ArduinoCommand.PING)
        except ArduinoCommunicationError:
            return False

    def check_motors(self) -> bool:
        """
        Request motor status check from Arduino.
        
        Returns:
            True if motors report as OK
        """
        try:
            return self.send_command(ArduinoCommand.CHECK)
        except ArduinoCommunicationError:
            logger.error("Failed to check motors")
            return False

    def get_hardware_status(self) -> Optional[HardwareStatus]:
        """
        Request full hardware status from Arduino.
        
        Parses CSV response: "M0_speed,M0_dir,M1_speed,M1_dir,M2_speed,M2_dir,M3_speed,M3_dir,P_speed,P_dir"
        
        Returns:
            HardwareStatus if successful, None if failed
        """
        if not self._is_connected:
            logger.error("Cannot get hardware status: Arduino not connected")
            return None

        try:
            with self._connection_lock:
                self._write_message("GET_STATUS\n")
                logger.debug("→ Sent: GET_STATUS")

            response = self._wait_for_response([f"{ArduinoResponse.STATUS.value}:"], self.timeout)
            if response is None:
                logger.warning("Timeout getting hardware status")
                return None

            if response.startswith(f"{ArduinoResponse.STATUS.value}:"):
                parts = response[len(f"{ArduinoResponse.STATUS.value}:"):].split(",")
                if len(parts) >= 10:
                    try:
                        return HardwareStatus(
                            motors_ok=True,
                            motor_0_speed=int(parts[0]),
                            motor_0_direction=int(parts[1]),
                            motor_1_speed=int(parts[2]),
                            motor_1_direction=int(parts[3]),
                            motor_2_speed=int(parts[4]),
                            motor_2_direction=int(parts[5]),
                            motor_3_speed=int(parts[6]),
                            motor_3_direction=int(parts[7]),
                            pump_speed=int(parts[8]),
                            pump_direction=int(parts[9]),
                            timestamp=time.time(),
                        )
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse STATUS response: {response} ({e})")
                        return None

            logger.warning(f"Unexpected hardware status response: {response}")
            return None

        except ArduinoCommunicationError as e:
            logger.error(f"Failed to get status: {e}")
            return None

    def move(self, direction: str) -> bool:
        """
        Send movement command to Arduino.
        
        Args:
            direction: 'forward', 'backward', 'left', 'right', 'stop'
            
        Returns:
            True if command sent successfully
        """
        direction_map = {
            "forward": (0, 200, 1, 1, 200, 1),    # (M0_spd, M0_dir, M1_spd, M1_dir, ...)
            "backward": (0, 200, 0, 1, 200, 0),
            "left": (0, 175, 0, 1, 175, 1),
            "right": (0, 175, 1, 1, 175, 0),
            "stop": (0, 0, 0, 1, 0, 0),
        }

        if direction not in direction_map:
            logger.error(f"Invalid direction: {direction}")
            return False

        try:
            motor_0_spd, motor_0_dir, motor_1_spd, motor_1_dir, motor_2_spd, motor_2_dir = direction_map[direction]
            
            # Set motor 0
            if not self.set_motor(0, motor_0_spd, motor_0_dir):
                return False
            
            # Set motor 1
            if not self.set_motor(1, motor_1_spd, motor_1_dir):
                return False
            
            # Set motors 2 and 3 (usually same as 0 and 1)
            if not self.set_motor(2, motor_0_spd, motor_0_dir):
                return False
            
            if not self.set_motor(3, motor_1_spd, motor_1_dir):
                return False
            
            return True
        except Exception as e:
            logger.error(f"Move command failed: {e}")
            return False

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

        speed = max(0, min(255, speed))
        direction = 1 if direction != 0 else 0

        try:
            with self._connection_lock:
                message = f"SET_MOTOR {motor_id} {speed} {direction}\n"
                self._write_message(message)
                logger.debug(f"→ Sent: SET_MOTOR {motor_id} {speed} {direction}")

            response = self._wait_for_response([ArduinoResponse.OK.value], self.timeout)
            if response is None:
                logger.warning(f"Timeout setting motor {motor_id}")
                return False

            if response.startswith(ArduinoResponse.ERROR.value):
                logger.warning(f"Arduino returned error for motor {motor_id}: {response}")
                return False

            return True
        except Exception as e:
            logger.error(f"Set motor {motor_id} failed: {e}")
            return False

    def set_pump(self, speed: int, direction: int) -> bool:
        """
        Set speed and direction for the pump motor.
        
        Args:
            speed: 0-255 (PWM speed)
            direction: 0 (backward) or 1 (forward)
            
        Returns:
            True if command sent successfully
        """
        speed = max(0, min(255, speed))
        direction = 1 if direction != 0 else 0

        try:
            with self._connection_lock:
                message = f"SET_PUMP {speed} {direction}\n"
                self._write_message(message)
                logger.debug(f"→ Sent: SET_PUMP {speed} {direction}")

            response = self._wait_for_response([ArduinoResponse.OK.value], self.timeout)
            if response is None:
                logger.warning("Timeout setting pump")
                return False

            if response.startswith(ArduinoResponse.ERROR.value):
                logger.warning(f"Arduino returned error for pump: {response}")
                return False

            return True
        except Exception as e:
            logger.error(f"Set pump failed: {e}")
            return False

    def stop_all(self) -> bool:
        """
        Stop all motors and pump.
        
        Returns:
            True if command sent successfully
        """
        try:
            with self._connection_lock:
                message = "STOP_ALL\n"
                self._write_message(message)
                logger.debug("→ Sent: STOP_ALL")

            response = self._wait_for_response([ArduinoResponse.OK.value], self.timeout)
            if response is None:
                logger.warning("Timeout stopping all motors")
                return False

            return True
        except Exception as e:
            logger.error(f"Stop all failed: {e}")
            return False

    def _start_reader_thread(self) -> None:
        """Start background thread to read serial responses."""
        self._stop_reader = False
        self._reader_thread = Thread(
            target=self._reader_loop,
            daemon=True,
            name="ArduinoReader",
        )
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        """
        Background thread that continuously reads from serial port.
        
        Parses responses line-by-line and puts them in response queue.
        """
        buffer = ""
        
        while not self._stop_reader:
            try:
                if not self._serial or not self._serial.is_open:
                    break
                
                # Read one byte at a time to handle partial responses
                byte = self._serial.read(1)
                
                if not byte:
                    continue
                
                char = byte.decode("utf-8", errors="ignore")
                buffer += char
                
                # Process complete lines (end with newline)
                if char == "\n":
                    line = buffer.strip()
                    if line:
                            try:
                                self._response_queue.put_nowait(line)
                            except Full:
                                logger.warning("Arduino response queue full, dropping oldest response")
                                try:
                                    self._response_queue.get_nowait()
                                except Empty:
                                    pass
                                self._response_queue.put_nowait(line)
            except (serial.SerialException, UnicodeDecodeError) as e:
                logger.warning(f"Reader thread error: {e}")
                self._is_connected = False
                break

            except Exception as e:
                logger.error(f"Unexpected reader thread error: {e}")
                break
