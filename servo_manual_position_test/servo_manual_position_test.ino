#include <Arduino.h>
#include <HardwareSerial.h>

const int RX_PIN = 16;
const int TX_PIN = 17;
HardwareSerial BusSerial(2);

const uint8_t SERVO_IDS[3] = {1, 2, 3};
const uint16_t MOVE_TIME_MS = 300;

void sendMovePosition(uint8_t id, uint16_t position, uint16_t moveTimeMs) {
  const uint8_t length = 7;
  const uint8_t posL = position & 0xFF;
  const uint8_t posH = (position >> 8) & 0xFF;
  const uint8_t timeL = moveTimeMs & 0xFF;
  const uint8_t timeH = (moveTimeMs >> 8) & 0xFF;
  const uint8_t checksum = static_cast<uint8_t>(
      ~(id + length + 1 + posL + posH + timeL + timeH));

  const uint8_t packet[10] = {
      0x55, 0x55, id, length, 1,
      posL, posH, timeL, timeH, checksum};
  BusSerial.write(packet, sizeof(packet));
}

bool parsePositions(String line, uint16_t positions[3]) {
  line.trim();
  const int firstComma = line.indexOf(',');
  const int secondComma = line.indexOf(',', firstComma + 1);
  if (firstComma <= 0 || secondComma <= firstComma + 1) return false;

  const String first = line.substring(0, firstComma);
  const String second = line.substring(firstComma + 1, secondComma);
  const String third = line.substring(secondComma + 1);
  if (first.length() == 0 || second.length() == 0 || third.length() == 0) return false;

  const long values[3] = {first.toInt(), second.toInt(), third.toInt()};
  for (int i = 0; i < 3; ++i) {
    if (values[i] < 0 || values[i] > 1000) return false;
    positions[i] = static_cast<uint16_t>(values[i]);
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  BusSerial.begin(115200, SERIAL_8N1, RX_PIN, TX_PIN);
  delay(500);

  Serial.println("===== MANUAL SERVO POSITION TEST =====");
  Serial.println("Enter: servo1,servo2,servo3");
  Serial.println("Example: 604,604,604");
  Serial.println("Range: 0..1000");
}

void loop() {
  if (Serial.available() == 0) {
    delay(10);
    return;
  }

  String line = Serial.readStringUntil('\n');
  uint16_t positions[3];
  if (!parsePositions(line, positions)) {
    Serial.println("ERROR: use three values from 0 to 1000, e.g. 604,604,604");
    return;
  }

  for (int i = 0; i < 3; ++i) {
    sendMovePosition(SERVO_IDS[i], positions[i], MOVE_TIME_MS);
  }

  Serial.print("SENT: servo1=");
  Serial.print(positions[0]);
  Serial.print(", servo2=");
  Serial.print(positions[1]);
  Serial.print(", servo3=");
  Serial.println(positions[2]);
}
