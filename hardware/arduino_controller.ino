/*
  Arduino Motor Controller - 3 Motor Drivers (4 movement + 1 pump)
  
  Hardware:
    - Motor Driver 1: 2 motors (0, 1) - Movement
    - Motor Driver 2: 2 motors (2, 3) - Movement
    - Motor Driver 3: 1 motor (pump)  - Auxiliary/Pump
  
  Protocol:
    - SET_MOTOR <id> <speed> <direction>
      id: 0-3 (motors) or P (pump)
      speed: 0-255
      direction: 0 (backward) or 1 (forward)
    - STOP_ALL
    - GET_STATUS
    - PING
    - CHECK
*/

#include <Arduino.h>

// ============================================================================
// PIN MAPPING
// ============================================================================

// Motor Driver 1 (Motors 0, 1)
const uint8_t MOTOR_0_SPEED = 3;   // PWM
const uint8_t MOTOR_0_IN1 = A0;    // Analog as digital
const uint8_t MOTOR_0_IN2 = A1;    // Analog as digital

const uint8_t MOTOR_1_SPEED = 5;   // PWM
const uint8_t MOTOR_1_IN3 = A2;    // Analog as digital
const uint8_t MOTOR_1_IN4 = A3;    // Analog as digital

// Motor Driver 2 (Motors 2, 3)
const uint8_t MOTOR_2_SPEED = 6;   // PWM
const uint8_t MOTOR_2_IN1 = A4;    // Analog as digital
const uint8_t MOTOR_2_IN2 = A5;    // Analog as digital

const uint8_t MOTOR_3_SPEED = 9;   // PWM
const uint8_t MOTOR_3_IN3 = 2;     // Digital
const uint8_t MOTOR_3_IN4 = 4;     // Digital

// Motor Driver 3 (Pump)
const uint8_t PUMP_SPEED = 10;     // PWM
const uint8_t PUMP_DIRECTION = 7;  // Digital

// ============================================================================
// MOTOR STATE
// ============================================================================

struct MotorState {
  uint8_t speed;      // 0-255
  uint8_t direction;  // 0 = backward, 1 = forward
};

MotorState motors[4];  // Motors 0-3
MotorState pump;       // Pump motor

// ============================================================================
// SERIAL PROTOCOL
// ============================================================================

const uint8_t MAX_COMMAND_LENGTH = 64;
char command_buffer[MAX_COMMAND_LENGTH];
uint8_t command_index = 0;

unsigned long last_command_timestamp = 0;
const unsigned long FAILSAFE_TIMEOUT_MS = 6000;

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

void initializePins();
void safeStop();
void setMotor(uint8_t motor_id, uint8_t speed, uint8_t direction);
void setPump(uint8_t speed, uint8_t direction);
void handleCommand(const char *command);
void sendResponse(const char *message);
void sendStatusResponse();
void checkFailsafe();
void parseSetMotor(const char *params);

// ============================================================================
// SETUP AND MAIN LOOP
// ============================================================================

void setup() {
  Serial.begin(9600);
  initializePins();
  safeStop();
  last_command_timestamp = millis();
}

void loop() {
  // Non-blocking serial input parsing
  while (Serial.available() > 0) {
    char incoming = Serial.read();

    if (incoming == '\r') {
      continue;
    }

    if (incoming == '\n') {
      command_buffer[command_index] = '\0';
      if (command_index > 0) {
        handleCommand(command_buffer);
        last_command_timestamp = millis();
      }
      command_index = 0;
      continue;
    }

    if (command_index < MAX_COMMAND_LENGTH - 1) {
      command_buffer[command_index++] = incoming;
    } else {
      command_index = 0;
      safeStop();
      sendResponse("ERROR:CMD_TOO_LONG");
    }
  }

  checkFailsafe();
}

// ============================================================================
// PIN INITIALIZATION
// ============================================================================

void initializePins() {
  // Motor Driver 1
  pinMode(MOTOR_0_SPEED, OUTPUT);
  pinMode(MOTOR_0_IN1, OUTPUT);
  pinMode(MOTOR_0_IN2, OUTPUT);
  
  pinMode(MOTOR_1_SPEED, OUTPUT);
  pinMode(MOTOR_1_IN3, OUTPUT);
  pinMode(MOTOR_1_IN4, OUTPUT);

  // Motor Driver 2
  pinMode(MOTOR_2_SPEED, OUTPUT);
  pinMode(MOTOR_2_IN1, OUTPUT);
  pinMode(MOTOR_2_IN2, OUTPUT);

  pinMode(MOTOR_3_SPEED, OUTPUT);
  pinMode(MOTOR_3_IN3, OUTPUT);
  pinMode(MOTOR_3_IN4, OUTPUT);

  // Motor Driver 3 (Pump)
  pinMode(PUMP_SPEED, OUTPUT);
  pinMode(PUMP_DIRECTION, OUTPUT);

  // Initialize all motor states to OFF
  for (int i = 0; i < 4; i++) {
    motors[i].speed = 0;
    motors[i].direction = 0;
  }
  pump.speed = 0;
  pump.direction = 0;
}

// ============================================================================
// MOTOR CONTROL
// ============================================================================

void setMotor(uint8_t motor_id, uint8_t speed, uint8_t direction) {
  if (motor_id > 3) {
    return;  // Invalid motor ID
  }

  speed = constrain(speed, 0, 255);
  direction = (direction != 0) ? 1 : 0;

  motors[motor_id].speed = speed;
  motors[motor_id].direction = direction;

  if (motor_id == 0) {
    digitalWrite(MOTOR_0_IN1, direction ? HIGH : LOW);
    digitalWrite(MOTOR_0_IN2, direction ? LOW : HIGH);
    analogWrite(MOTOR_0_SPEED, speed);
  } else if (motor_id == 1) {
    digitalWrite(MOTOR_1_IN3, direction ? HIGH : LOW);
    digitalWrite(MOTOR_1_IN4, direction ? LOW : HIGH);
    analogWrite(MOTOR_1_SPEED, speed);
  } else if (motor_id == 2) {
    digitalWrite(MOTOR_2_IN1, direction ? HIGH : LOW);
    digitalWrite(MOTOR_2_IN2, direction ? LOW : HIGH);
    analogWrite(MOTOR_2_SPEED, speed);
  } else if (motor_id == 3) {
    digitalWrite(MOTOR_3_IN3, direction ? HIGH : LOW);
    digitalWrite(MOTOR_3_IN4, direction ? LOW : HIGH);
    analogWrite(MOTOR_3_SPEED, speed);
  }
}

void setPump(uint8_t speed, uint8_t direction) {
  speed = constrain(speed, 0, 255);
  direction = (direction != 0) ? 1 : 0;

  pump.speed = speed;
  pump.direction = direction;

  digitalWrite(PUMP_DIRECTION, direction ? HIGH : LOW);
  analogWrite(PUMP_SPEED, speed);
}

void safeStop() {
  // Stop all motors
  analogWrite(MOTOR_0_SPEED, 0);
  digitalWrite(MOTOR_0_IN1, LOW);
  digitalWrite(MOTOR_0_IN2, LOW);
  motors[0].speed = 0;
  motors[0].direction = 0;

  analogWrite(MOTOR_1_SPEED, 0);
  digitalWrite(MOTOR_1_IN3, LOW);
  digitalWrite(MOTOR_1_IN4, LOW);
  motors[1].speed = 0;
  motors[1].direction = 0;

  analogWrite(MOTOR_2_SPEED, 0);
  digitalWrite(MOTOR_2_IN1, LOW);
  digitalWrite(MOTOR_2_IN2, LOW);
  motors[2].speed = 0;
  motors[2].direction = 0;

  analogWrite(MOTOR_3_SPEED, 0);
  digitalWrite(MOTOR_3_IN3, LOW);
  digitalWrite(MOTOR_3_IN4, LOW);
  motors[3].speed = 0;
  motors[3].direction = 0;

  // Stop pump
  analogWrite(PUMP_SPEED, 0);
  digitalWrite(PUMP_DIRECTION, LOW);
  pump.speed = 0;
  pump.direction = 0;
}

// ============================================================================
// COMMAND HANDLING
// ============================================================================

void handleCommand(const char *command) {
  if (command[0] == '\0') {
    return;
  }

  // PING
  if (strcasecmp(command, "PING") == 0) {
    sendResponse("PONG");
    return;
  }

  // CHECK
  if (strcasecmp(command, "CHECK") == 0) {
    sendResponse("MOTORS_OK");
    return;
  }

  // STOP_ALL
  if (strcasecmp(command, "STOP_ALL") == 0 || strcasecmp(command, "STOP") == 0) {
    safeStop();
    sendResponse("OK");
    return;
  }

  // GET_STATUS
  if (strcasecmp(command, "GET_STATUS") == 0) {
    sendStatusResponse();
    return;
  }

  // SET_MOTOR <id> <speed> <direction>
  if (strncasecmp(command, "SET_MOTOR ", 10) == 0) {
    parseSetMotor(command + 10);
    return;
  }

  // SET_PUMP <speed> <direction>
  if (strncasecmp(command, "SET_PUMP ", 9) == 0) {
    char params[32];
    strncpy(params, command + 9, sizeof(params) - 1);
    params[sizeof(params) - 1] = '\0';

    uint8_t speed = 0;
    uint8_t direction = 0;
    if (sscanf(params, "%hhu %hhu", &speed, &direction) == 2) {
      setPump(speed, direction);
      sendResponse("OK");
    } else {
      sendResponse("ERROR:INVALID_PUMP_PARAMS");
    }
    return;
  }

  // Unknown command: safe behavior
  safeStop();
  sendResponse("ERROR:UNKNOWN_COMMAND");
}

void parseSetMotor(const char *params) {
  uint8_t motor_id = 255;
  uint8_t speed = 0;
  uint8_t direction = 0;

  // Parse: <id> <speed> <direction>
  // id can be 0-3 or P for pump
  if (params[0] == 'P' || params[0] == 'p') {
    // Redirect to pump
    uint8_t speed = 0, direction = 0;
    if (sscanf(params + 2, "%hhu %hhu", &speed, &direction) == 2) {
      setPump(speed, direction);
      sendResponse("OK");
    } else {
      sendResponse("ERROR:INVALID_PUMP_PARAMS");
    }
    return;
  }

  // Parse motor ID (0-3)
  motor_id = params[0] - '0';
  if (motor_id > 3) {
    sendResponse("ERROR:INVALID_MOTOR_ID");
    return;
  }

  // Parse speed and direction
  if (sscanf(params + 2, "%hhu %hhu", &speed, &direction) == 2) {
    setMotor(motor_id, speed, direction);
    sendResponse("OK");
  } else {
    sendResponse("ERROR:INVALID_MOTOR_PARAMS");
  }
}

// ============================================================================
// RESPONSES
// ============================================================================

void sendResponse(const char *message) {
  Serial.println(message);
}

void sendStatusResponse() {
  char status_line[192];

  snprintf(
    status_line,
    sizeof(status_line),
    "STATUS:%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
    motors[0].speed, motors[0].direction,
    motors[1].speed, motors[1].direction,
    motors[2].speed, motors[2].direction,
    motors[3].speed, motors[3].direction,
    pump.speed, pump.direction
  );

  sendResponse(status_line);
}

// ============================================================================
// FAILSAFE
// ============================================================================

void checkFailsafe() {
  bool any_motor_on = false;
  
  for (int i = 0; i < 4; i++) {
    if (motors[i].speed > 0) {
      any_motor_on = true;
      break;
    }
  }
  
  if (!any_motor_on && pump.speed == 0) {
    return;  // Nothing running
  }

  if (millis() - last_command_timestamp > FAILSAFE_TIMEOUT_MS) {
    safeStop();
    sendResponse("ERROR:FAILSAFE_STOP");
    last_command_timestamp = millis();
  }
}
