#include <Arduino.h>
#include <HardwareSerial.h>

// Read the current position of all three servos.
// Manually place all Link 1 arms parallel to the floor before reset.

const int RX_PIN = 16;
const int TX_PIN = 17;
HardwareSerial BusSerial(2);

const uint8_t SERVO_IDS[] = {1, 2, 3};
const size_t SERVO_COUNT = sizeof(SERVO_IDS) / sizeof(SERVO_IDS[0]);
const uint8_t SERVO_POS_READ = 28;
const uint32_t READ_TIMEOUT_MS = 50;
const uint8_t READ_RETRIES = 5;
const uint16_t RETRY_DELAY_MS = 50;

uint8_t checksumFor(const uint8_t *data, size_t first, size_t lastExclusive) {
  uint16_t sum = 0;
  for (size_t i = first; i < lastExclusive; ++i) sum += data[i];
  return static_cast<uint8_t>(~sum);
}

void clearBusInput() {
  while (BusSerial.available() > 0) BusSerial.read();
}

void sendPositionRead(uint8_t id) {
  const uint8_t length = 3;
  const uint8_t checksum = static_cast<uint8_t>(~(id + length + SERVO_POS_READ));
  const uint8_t packet[6] = {0x55, 0x55, id, length, SERVO_POS_READ, checksum};
  BusSerial.write(packet, sizeof(packet));
}

bool readServoPosition(uint8_t expectedId, uint16_t &position) {
  clearBusInput();
  sendPositionRead(expectedId);

  const uint32_t startMs = millis();
  uint8_t packet[20];
  size_t count = 0;
  bool sawHeader = false;

  while (millis() - startMs < READ_TIMEOUT_MS) {
    if (BusSerial.available() <= 0) {
      delay(1);
      continue;
    }

    const uint8_t byteValue = static_cast<uint8_t>(BusSerial.read());
    if (!sawHeader) {
      if (byteValue == 0x55) {
        if (count == 0) packet[count++] = byteValue;
        else {
          packet[count++] = byteValue;
          sawHeader = true;
        }
      } else count = 0;
      continue;
    }

    if (count >= sizeof(packet)) return false;
    packet[count++] = byteValue;

    if (count == 4) {
      const size_t totalLength = static_cast<size_t>(packet[3]) + 3;
      if (totalLength < 7 || totalLength > sizeof(packet)) return false;
    }

    if (count >= 4) {
      const size_t totalLength = static_cast<size_t>(packet[3]) + 3;
      if (count == totalLength) {
        if (packet[2] != expectedId || packet[4] != SERVO_POS_READ) return false;
        const uint8_t expectedChecksum = checksumFor(packet, 2, totalLength - 1);
        if (packet[totalLength - 1] != expectedChecksum) return false;
        position = static_cast<uint16_t>(packet[5]) |
                   (static_cast<uint16_t>(packet[6]) << 8);
        return position <= 1000;
      }
    }
  }
  return false;
}

bool readWithRetry(uint8_t id, uint16_t &position) {
  for (uint8_t attempt = 1; attempt <= READ_RETRIES; ++attempt) {
    if (readServoPosition(id, position)) return true;
    delay(RETRY_DELAY_MS);
  }
  return false;
}

void readAndPrintThresholds() {
  uint16_t positions[SERVO_COUNT] = {0, 0, 0};
  bool allRead = true;

  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    Serial.print("READING,servo_id=");
    Serial.println(SERVO_IDS[i]);
    if (!readWithRetry(SERVO_IDS[i], positions[i])) {
      Serial.print("READ_ERROR,servo_id=");
      Serial.println(SERVO_IDS[i]);
      allRead = false;
    } else {
      Serial.print("READ_OK,servo_id=");
      Serial.print(SERVO_IDS[i]);
      Serial.print(",position=");
      Serial.println(positions[i]);
    }
  }

  if (!allRead) {
    Serial.println("THRESHOLD_ABORT: could not read all three servo positions.");
    return;
  }

  Serial.println();
  Serial.println("===== READ SERVO FLOOR THRESHOLDS =====");
  Serial.println("Current positions are stored as the floor-parallel limits.");
  Serial.print("{\"type\":\"floor_threshold\",\"servo_1\":");
  Serial.print(positions[0]);
  Serial.print(",\"servo_2\":");
  Serial.print(positions[1]);
  Serial.print(",\"servo_3\":");
  Serial.print(positions[2]);
  Serial.println("}");

  Serial.println();
  Serial.println("Copy this into the loaded test:");
  Serial.print("const uint16_t FLOOR_THRESHOLDS[3] = {");
  Serial.print(positions[0]);
  Serial.print(", ");
  Serial.print(positions[1]);
  Serial.print(", ");
  Serial.print(positions[2]);
  Serial.println("};");
  Serial.println("Each DOWN target must satisfy: target >= FLOOR_THRESHOLDS[i]");
}

void setup() {
  Serial.begin(115200);
  BusSerial.begin(115200, SERIAL_8N1, RX_PIN, TX_PIN);
  delay(1000);
  Serial.println();
  Serial.println("===== SERVO FLOOR THRESHOLD READER =====");
  Serial.println("Place all Link 1 arms parallel to the floor before reset.");
  readAndPrintThresholds();
}

void loop() {
  delay(1000);
}
