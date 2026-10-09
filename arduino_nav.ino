/*
 * Arduino vehicle runner for Jetson pathfinding.
 *
 * Jetson (jetson_nav.py + pathfind.py) plans the path and sends:
 *   L<left> R<right>\n
 *   STOP\n
 *
 * This sketch executes those commands over CAN (Roboteq) and streams
 * FEEDBACK so the Jetson can estimate pose.
 *
 * Derived from main.cpp hardware bring-up (CAN, BNO085, HC-05, USB).
 */

#include <SPI.h>
#include <mcp_can.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>

// ============================================================
// PRECEDENTED (from main.cpp)
// ============================================================

#define CAN_CS_PIN 53
#define HC05_BAUD 115200
#define CAN_BAUD  CAN_250KBPS
#define CAN_CLOCK MCP_8MHZ
#define BNO08X_RESET -1

const byte NODE_ID = 1;
const unsigned long NMT_ID = 0x000;
const unsigned long RPDO1_ID = 0x200 + NODE_ID;
const unsigned long SDO_TX_ID = 0x600 + NODE_ID;
const unsigned long SDO_RX_ID = 0x580 + NODE_ID;
const unsigned long HEARTBEAT_ID = 0x700 + NODE_ID;
const uint16_t OBJ_MOTOR_SPEED = 0x210A;
const uint16_t OBJ_HALL_COUNT = 0x2105;
const uint16_t OBJ_MOTOR_CMD = 0x2000;  // Roboteq !G, -1000..1000 per channel
const int32_t BLUETOOTH_SPEED = 200;

// ============================================================
// ASSUMED (no precedent for nav-mode policy — tune as needed)
// ============================================================

// When true, HC-05 commands are ignored so Jetson owns the motors.
const bool AUTONOMOUS_USB_PRIORITY = true;

// Command freshness watchdog (ms). If Jetson goes silent, stop.
const unsigned long CMD_WATCHDOG_MS = 500;

// Loop periods (ms) — match main.cpp defaults unless retuned.
const unsigned long MOTOR_SEND_MS = 40;
const unsigned long FEEDBACK_REQ_MS = 25;
const unsigned long FEEDBACK_PRINT_MS = 50;  // 20 ms saturated USB and blocked CAN reads

// Roboteq maps RPDO1 to user variables (VAR1/VAR2) by default, which only
// moves the motors if a MicroBasic script or a remap forwards them to !G.
// true = write !G (0x2000) directly over SDO instead of RPDO1.
const bool USE_SDO_MOTOR_CMD = true;

// ============================================================
// HARDWARE OBJECTS
// ============================================================

MCP_CAN CAN(CAN_CS_PIN);
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

float imu_qw = 1.0, imu_qx = 0.0, imu_qy = 0.0, imu_qz = 0.0;
float imu_gx = 0.0, imu_gy = 0.0, imu_gz = 0.0;
float imu_ax = 0.0, imu_ay = 0.0, imu_az = 0.0;

int32_t leftCommand = 0;
int32_t rightCommand = 0;
unsigned long lastUsbCommandMs = 0;

int16_t leftRPM = 0;
int16_t rightRPM = 0;
int32_t leftHall = 0;
int32_t rightHall = 0;

bool gotLeftRPM = false;
bool gotRightRPM = false;
bool gotLeftHall = false;
bool gotRightHall = false;

String usbBuffer = "";

// ============================================================
// MOTOR / CANOPEN
// ============================================================

void writeSDO32(byte subIndex, uint16_t objectIndex, int32_t value)
{
    byte data[8] = {
        0x23,
        (byte)(objectIndex & 0xFF),
        (byte)((objectIndex >> 8) & 0xFF),
        subIndex,
        (byte)(value & 0xFF),
        (byte)((value >> 8) & 0xFF),
        (byte)((value >> 16) & 0xFF),
        (byte)((value >> 24) & 0xFF)
    };

    if (CAN.sendMsgBuf(SDO_TX_ID, 0, 8, data) != CAN_OK)
    {
        Serial.print("[SDO TX ERROR] 0x");
        Serial.println(objectIndex, HEX);
    }
}

void sendMotorCommands(int32_t left, int32_t right)
{
    left = constrain(left, -1000, 1000);
    right = constrain(right, -1000, 1000);

    if (USE_SDO_MOTOR_CMD)
    {
        writeSDO32(1, OBJ_MOTOR_CMD, left);
        writeSDO32(2, OBJ_MOTOR_CMD, right);
        return;
    }

    byte data[8];
    data[0] = (byte)(left & 0xFF);
    data[1] = (byte)((left >> 8) & 0xFF);
    data[2] = (byte)((left >> 16) & 0xFF);
    data[3] = (byte)((left >> 24) & 0xFF);
    data[4] = (byte)(right & 0xFF);
    data[5] = (byte)((right >> 8) & 0xFF);
    data[6] = (byte)((right >> 16) & 0xFF);
    data[7] = (byte)((right >> 24) & 0xFF);

    if (CAN.sendMsgBuf(RPDO1_ID, 0, 8, data) != CAN_OK)
    {
        Serial.println("[CAN TX ERROR] Motor command failed");
    }
}

void startNode()
{
    byte data[2] = {0x01, NODE_ID};
    if (CAN.sendMsgBuf(NMT_ID, 0, 2, data) == CAN_OK)
        Serial.println("[NMT] Node 1 START sent");
    else
        Serial.println("[NMT ERROR]");
}

void requestSDO(byte subIndex, uint16_t objectIndex)
{
    byte data[8] = {
        0x40,
        (byte)(objectIndex & 0xFF),
        (byte)((objectIndex >> 8) & 0xFF),
        subIndex,
        0, 0, 0, 0
    };

    if (CAN.sendMsgBuf(SDO_TX_ID, 0, 8, data) != CAN_OK)
    {
        Serial.print("[SDO TX ERROR] 0x");
        Serial.println(objectIndex, HEX);
    }
}

void requestMotorSpeed(byte motor) { requestSDO(motor, OBJ_MOTOR_SPEED); }
void requestHallCount(byte motor) { requestSDO(motor, OBJ_HALL_COUNT); }

int16_t decodeInt16(byte *data)
{
    uint16_t value = ((uint16_t)data[4]) | ((uint16_t)data[5] << 8);
    return (int16_t)value;
}

int32_t decodeInt32(byte *data)
{
    uint32_t value =
        ((uint32_t)data[4]) |
        ((uint32_t)data[5] << 8) |
        ((uint32_t)data[6] << 16) |
        ((uint32_t)data[7] << 24);
    return (int32_t)value;
}

void resetFeedbackFlags()
{
    gotLeftRPM = false;
    gotRightRPM = false;
    gotLeftHall = false;
    gotRightHall = false;
}

void printCompleteFeedback()
{
    Serial.print("FEEDBACK: L");
    Serial.print(leftRPM);
    Serial.print(" R");
    Serial.print(rightRPM);
    Serial.print(" RPM | HALL L");
    Serial.print(leftHall);
    Serial.print(" R");
    Serial.print(rightHall);
    Serial.print(" | IMU ");
    Serial.print(imu_qw, 4); Serial.print(" ");
    Serial.print(imu_qx, 4); Serial.print(" ");
    Serial.print(imu_qy, 4); Serial.print(" ");
    Serial.print(imu_qz, 4); Serial.print(" ");
    Serial.print(imu_gx, 4); Serial.print(" ");
    Serial.print(imu_gy, 4); Serial.print(" ");
    Serial.print(imu_gz, 4); Serial.print(" ");
    Serial.print(imu_ax, 4); Serial.print(" ");
    Serial.print(imu_ay, 4); Serial.print(" ");
    Serial.println(imu_az, 4);
}

void printSDOResponse(const char *tag, uint16_t objectIndex, byte subIndex, byte *data)
{
    Serial.print(tag);
    Serial.print(" cmd=0x");
    Serial.print(data[0], HEX);
    Serial.print(" obj=0x");
    Serial.print(objectIndex, HEX);
    Serial.print(" sub=");
    Serial.print(subIndex);
    Serial.print(" value=");
    Serial.print(decodeInt32(data));
    Serial.print(" hex=0x");
    Serial.println((uint32_t)decodeInt32(data), HEX);
}

void readCAN()
{
    while (CAN.checkReceive() == CAN_MSGAVAIL)
    {
        unsigned long rawRxId;
        byte len;
        byte rxBuf[8];
        CAN.readMsgBuf(&rawRxId, &len, rxBuf);
        unsigned long rxId = rawRxId & 0x7FF;

        if (rxId == HEARTBEAT_ID)
        {
            // Controller powered up after us (or reset): it's not operational
            // and ignores RPDOs until it gets NMT START again. 0x05 = operational.
            if (len >= 1 && (rxBuf[0] & 0x7F) != 0x05)
                startNode();
            continue;
        }

        if (rxId != SDO_RX_ID || len != 8)
            continue;

        byte command = rxBuf[0];
        uint16_t objectIndex = ((uint16_t)rxBuf[2] << 8) | rxBuf[1];
        byte subIndex = rxBuf[3];

        if (command == 0x80)
        {
            // SDO abort: print the reason (rate-limited for the periodic motor writes)
            static unsigned long lastCmdAbortMs = 0;
            if (objectIndex == OBJ_MOTOR_CMD)
            {
                if (millis() - lastCmdAbortMs < 1000)
                    continue;
                lastCmdAbortMs = millis();
            }
            printSDOResponse("[SDO ABORT]", objectIndex, subIndex, rxBuf);
            continue;
        }

        if (objectIndex == OBJ_MOTOR_SPEED)
        {
            int16_t speed;
            if (command == 0x4B)
                speed = decodeInt16(rxBuf);
            else if (command == 0x43)
                speed = (int16_t)decodeInt32(rxBuf);
            else
                continue;

            if (subIndex == 1) { leftRPM = speed; gotLeftRPM = true; }
            else if (subIndex == 2) { rightRPM = speed; gotRightRPM = true; }
            continue;
        }

        if (objectIndex == OBJ_HALL_COUNT && command == 0x43)
        {
            int32_t hallCount = decodeInt32(rxBuf);
            if (subIndex == 1) { leftHall = hallCount; gotLeftHall = true; }
            else if (subIndex == 2) { rightHall = hallCount; gotRightHall = true; }
            continue;
        }

        // Write acks for the periodic motor command are expected; skip them
        if (objectIndex == OBJ_MOTOR_CMD && command == 0x60)
            continue;

        // Anything else is a reply to a SDOR/SDOW diagnostic command
        printSDOResponse("[SDO]", objectIndex, subIndex, rxBuf);
    }
}

// ============================================================
// USB / JETSON COMMANDS  (pathfinding output ends here)
// ============================================================

void processMotorCommand(String command)
{
    command.trim();
    command.toUpperCase();

    int leftValue;
    int rightValue;
    int parsed = sscanf(command.c_str(), "L%d R%d", &leftValue, &rightValue);

    if (parsed == 2)
    {
        leftCommand = constrain(leftValue, -1000, 1000);
        rightCommand = constrain(rightValue, -1000, 1000);
        lastUsbCommandMs = millis();
        resetFeedbackFlags();

        Serial.print("[COMMAND] L");
        Serial.print(leftCommand);
        Serial.print(" R");
        Serial.println(rightCommand);
        return;
    }

    // Diagnostics: SDOR <index hex> <sub>   /   SDOW <index hex> <sub> <int32 value>
    unsigned int sdoIndex;
    int sdoSub;
    long sdoValue;
    if (sscanf(command.c_str(), "SDOR %x %d", &sdoIndex, &sdoSub) == 2)
    {
        requestSDO((byte)sdoSub, (uint16_t)sdoIndex);
        return;
    }
    if (sscanf(command.c_str(), "SDOW %x %d %ld", &sdoIndex, &sdoSub, &sdoValue) == 3)
    {
        writeSDO32((byte)sdoSub, (uint16_t)sdoIndex, (int32_t)sdoValue);
        return;
    }

    if (command == "STOP")
    {
        leftCommand = 0;
        rightCommand = 0;
        lastUsbCommandMs = millis();
        resetFeedbackFlags();
        Serial.println("[COMMAND] STOP");
    }
}

void readUSB()
{
    while (Serial.available())
    {
        char c = Serial.read();
        if (c == '\r')
            continue;

        if (c == '\n')
        {
            if (usbBuffer.length() > 0)
            {
                processMotorCommand(usbBuffer);
                usbBuffer = "";
            }
        }
        else
        {
            usbBuffer += c;
            if (usbBuffer.length() > 40)
                usbBuffer = "";
        }
    }
}

// ============================================================
// BLUETOOTH (manual override when not in autonomous priority)
// ============================================================

void processBluetoothCommand(char command)
{
    if (AUTONOMOUS_USB_PRIORITY)
        return;

    command = toupper(command);

    if (command == 'F')
    {
        leftCommand = BLUETOOTH_SPEED;
        rightCommand = BLUETOOTH_SPEED;
    }
    else if (command == 'B')
    {
        leftCommand = -BLUETOOTH_SPEED;
        rightCommand = -BLUETOOTH_SPEED;
    }
    else if (command == 'L')
    {
        leftCommand = -BLUETOOTH_SPEED;
        rightCommand = BLUETOOTH_SPEED;
    }
    else if (command == 'R')
    {
        leftCommand = BLUETOOTH_SPEED;
        rightCommand = -BLUETOOTH_SPEED;
    }
    else if (command == 'S')
    {
        leftCommand = 0;
        rightCommand = 0;
    }
    else
    {
        return;
    }

    resetFeedbackFlags();
    Serial.print("[BT COMMAND] ");
    Serial.println(command);
}

void readBluetooth()
{
    while (Serial1.available())
    {
        char c = Serial1.read();
        if (c == '\r' || c == '\n')
            continue;
        processBluetoothCommand(c);
    }
}

// ============================================================
// IMU
// ============================================================

void readIMU()
{
    if (bno08x.wasReset())
    {
        bno08x.enableReport(SH2_ROTATION_VECTOR, 20000);
        bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 20000);
        bno08x.enableReport(SH2_ACCELEROMETER, 20000);
    }

    if (!bno08x.getSensorEvent(&sensorValue))
        return;

    if (sensorValue.sensorId == SH2_ROTATION_VECTOR)
    {
        imu_qw = sensorValue.un.rotationVector.real;
        imu_qx = sensorValue.un.rotationVector.i;
        imu_qy = sensorValue.un.rotationVector.j;
        imu_qz = sensorValue.un.rotationVector.k;
    }
    else if (sensorValue.sensorId == SH2_GYROSCOPE_CALIBRATED)
    {
        imu_gx = sensorValue.un.gyroscope.x;
        imu_gy = sensorValue.un.gyroscope.y;
        imu_gz = sensorValue.un.gyroscope.z;
    }
    else if (sensorValue.sensorId == SH2_ACCELEROMETER)
    {
        imu_ax = sensorValue.un.accelerometer.x;
        imu_ay = sensorValue.un.accelerometer.y;
        imu_az = sensorValue.un.accelerometer.z;
    }
}

// ============================================================
// WATCHDOG: stop if Jetson stops talking
// ============================================================

void applyCommandWatchdog()
{
    if (!AUTONOMOUS_USB_PRIORITY)
        return;

    if (millis() - lastUsbCommandMs > CMD_WATCHDOG_MS)
    {
        if (leftCommand != 0 || rightCommand != 0)
        {
            leftCommand = 0;
            rightCommand = 0;
            Serial.println("[WATCHDOG] USB command timeout -> STOP");
        }
    }
}

// ============================================================
// SETUP / LOOP
// ============================================================

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println("========================================");
    Serial.println(" ARDUINO NAV (Jetson pathfind client)");
    Serial.println(" USB: L<left> R<right> | STOP");
    Serial.println("========================================");

    Serial1.begin(HC05_BAUD);
    SPI.begin();

    // Retry: after a cold power-up the MCP2515 sometimes fails its first init
    bool canOk = false;
    for (int attempt = 0; attempt < 5 && !canOk; attempt++)
    {
        if (attempt > 0)
            delay(200);
        canOk = CAN.begin(MCP_ANY, CAN_BAUD, CAN_CLOCK) == CAN_OK;
    }
    if (canOk)
        Serial.println("[OK] MCP2515");
    else
        Serial.println("[ERROR] MCP2515 FAILED");

    CAN.setMode(MCP_NORMAL);
    startNode();
    sendMotorCommands(0, 0);
    lastUsbCommandMs = millis();

    // The BNO08x can still be booting when we get here after a reset; retry
    bool imuOk = false;
    for (int attempt = 0; attempt < 5 && !imuOk; attempt++)
    {
        delay(200);
        imuOk = bno08x.begin_I2C();
    }
    if (!imuOk)
    {
        Serial.println("[ERROR] Failed to find BNO08x chip");
    }
    else
    {
        Serial.println("[OK] BNO08x Found!");
        bno08x.enableReport(SH2_ROTATION_VECTOR, 20000);
        bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 20000);
        bno08x.enableReport(SH2_ACCELEROMETER, 20000);
    }

    Serial.println("SYSTEM READY (waiting for Jetson USB commands)");
}

void loop()
{
    readBluetooth();
    readUSB();
    readCAN();
    readIMU();
    applyCommandWatchdog();

    static unsigned long lastCANSend = 0;
    if (millis() - lastCANSend >= MOTOR_SEND_MS)
    {
        sendMotorCommands(leftCommand, rightCommand);
        lastCANSend = millis();
    }

    static unsigned long lastFeedback = 0;
    static byte feedbackState = 0;
    if (millis() - lastFeedback >= FEEDBACK_REQ_MS)
    {
        if (feedbackState == 0) requestMotorSpeed(1);
        else if (feedbackState == 1) requestMotorSpeed(2);
        else if (feedbackState == 2) requestHallCount(1);
        else if (feedbackState == 3) requestHallCount(2);

        feedbackState = (feedbackState + 1) % 4;
        lastFeedback = millis();
    }

    static unsigned long lastPrint = 0;
    if (millis() - lastPrint >= FEEDBACK_PRINT_MS)
    {
        printCompleteFeedback();
        lastPrint = millis();
    }
}
