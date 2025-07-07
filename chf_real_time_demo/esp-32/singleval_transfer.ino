
#include <Wire.h>
#include "MAX30105.h"

MAX30105 sensor;

#define SAMPLE_RATE      125  // Desired sample rate
#define SAMPLE_INTERVAL  (1000 / SAMPLE_RATE)

unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!sensor.begin()) {
    Serial.println("❌ MAX30102 not found");
    while (true);
  }

  sensor.setup(); // Default settings
  sensor.setPulseAmplitudeRed(0x1F); // Reduce LED brightness
  sensor.setPulseAmplitudeIR(0x1F);
  sensor.setSampleRate(SAMPLE_RATE);

  Serial.println("✅ Streaming IR values at 125 Hz...");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleTime >= SAMPLE_INTERVAL) {v./,
    uint16_t ir = sensor.getIR();
    Serial.println(ir);  // Stream one IR value at a time
    lastSampleTime = now;
  }
}