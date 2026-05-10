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
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class ArduinoCommand(Enum):
    """Commands sent to Arduino."""
    PING = "PING"
    CHECK = "CHECK"
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    STOP = "STOP"
    GET_STATUS = "GET_STATUS"


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
    motor_left_speed: int  # 0-255
    motor_right_speed: int  # 0-255
    battery_voltage: float
    temperature: float
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
        port: str = "/dev/ttyACM0",  # Arduino serial port on Raspberry Pi
        baudrate: int = 9600,
        timeout: float = 2.0,
        write_timeout: float = 2.0,
        max_reconnect_attempts: int = 3,
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
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._serial: Optional[serial.Serial] = None
        self._connection_lock = Lock()
        self._is_connected = False
        self._response_queue: Queue[str] = Queue()
        self._reader_thread: Optional[Thread] = None
        self._stop_reader = False

    def _expected_responses_for(self, command: ArduinoCommand) -> Sequence[str]:
        """Return acceptable response strings for a given command."""
        mapping = {
            ArduinoCommand.PING: [ArduinoResponse.PING_RESPONSE.value, ArduinoResponse.OK.value],
            ArduinoCommand.CHECK: [ArduinoResponse.MOTORS_OK.value, ArduinoResponse.OK.value],
            ArduinoCommand.MOVE_FORWARD: [ArduinoResponse.OK.value],
            ArduinoCommand.MOVE_BACKWARD: [ArduinoResponse.OK.value],
            ArduinoCommand.MOVE_LEFT: [ArduinoResponse.OK.value],
            ArduinoCommand.MOVE_RIGHT: [ArduinoResponse.OK.value],
            ArduinoCommand.STOP: [ArduinoResponse.OK.value],
        }
        return mapping.get(command, [ArduinoResponse.OK.value])

    def connect(self) -> bool:
        """
        Establish connection to Arduino.
        
        Implements retry logic with exponential backoff.
        
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
                    
                    self._serial = serial.Serial(
                        port=self.port,
                        baudrate=self.baudrate,
                        timeout=self.timeout,
                        write_timeout=self.write_timeout,
                    )
                    
                    # Give Arduino time to reset and initialize
                    time.sleep(2.0)
                    
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
            
            if self._serial and self._serial.is_open:
                self._serial.close()
                logger.info("Disconnected from Arduino")
            
            self._is_connected = False

    def is_connected(self) -> bool:
        """Check if currently connected to Arduino."""
        return self._is_connected

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

        try:
            with self._connection_lock:
                message = f"{command.value}\n"
                self._serial.write(message.encode())
                logger.debug(f"→ Sent: {command.value}")

            saved_responses = []
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = self._response_queue.get(timeout=0.1)
                    logger.debug(f"← Received: {response}")
                    if any(response.startswith(prefix) for prefix in expected_responses):
                        for saved in saved_responses:
                            self._response_queue.put(saved)
                        return True
                    saved_responses.append(response)
                except Empty:
                    continue

            logger.warning(f"Timeout waiting for Arduino response to {command.value}")
            for saved in saved_responses:
                self._response_queue.put(saved)
            return False

        except serial.SerialException as e:
            logger.error(f"Serial write failed: {e}")
            self._is_connected = False
            raise ArduinoCommunicationError(f"Serial communication failed: {e}")

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
        
        Parses CSV response: "motor_ok,left_speed,right_speed,voltage,temp"
        
        Returns:
            HardwareStatus if successful, None if failed
        """
        try:
            with self._connection_lock:
                self._serial.write(b"GET_STATUS\n")
                logger.debug("→ Sent: GET_STATUS")

            # Read status line
            start_time = time.time()
            timeout = self.timeout
            
            while time.time() - start_time < timeout:
                try:
                    response = self._response_queue.get(timeout=0.1)
                    
                    # Parse CSV status
                    if response.startswith("STATUS:"):
                        parts = response[7:].split(",")
                        if len(parts) >= 5:
                            return HardwareStatus(
                                motors_ok=parts[0] == "1",
                                motor_left_speed=int(parts[1]),
                                motor_right_speed=int(parts[2]),
                                battery_voltage=float(parts[3]),
                                temperature=float(parts[4]),
                                timestamp=time.time(),
                            )

                except (Empty, ValueError, IndexError) as e:
                    logger.debug(f"Status parse error: {e}")
                    continue

            logger.warning("Timeout getting hardware status")
            return None

        except serial.SerialException as e:
            logger.error(f"Failed to get status: {e}")
            self._is_connected = False
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
            "forward": ArduinoCommand.MOVE_FORWARD,
            "backward": ArduinoCommand.MOVE_BACKWARD,
            "left": ArduinoCommand.MOVE_LEFT,
            "right": ArduinoCommand.MOVE_RIGHT,
            "stop": ArduinoCommand.STOP,
        }

        if direction not in direction_map:
            logger.error(f"Invalid direction: {direction}")
            return False

        try:
            cmd = direction_map[direction]
            return self.send_command(cmd)
        except ArduinoCommunicationError as e:
            logger.error(f"Move command failed: {e}")
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
                        self._response_queue.put(line)
                        logger.debug(f"Response queued: {line}")
                    buffer = ""

            except (serial.SerialException, UnicodeDecodeError) as e:
                logger.warning(f"Reader thread error: {e}")
                self._is_connected = False
                break

            except Exception as e:
                logger.error(f"Unexpected reader thread error: {e}")
                break
