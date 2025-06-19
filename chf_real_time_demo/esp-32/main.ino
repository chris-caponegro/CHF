#include <Wire.h>
#include "MAX30105.h"

MAX30105 sensor;

#define SAMPLE_RATE      25
#define WINDOW_SIZE      512
#define SAMPLE_INTERVAL  (1000 / SAMPLE_RATE)

uint16_t ppg_ir[WINDOW_SIZE];
int sampleIndex = 0;
unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!sensor.begin()) {
    Serial.println("❌ MAX30102 not found");
    while (true);
  }

  sensor.setup();
  sensor.setPulseAmplitudeRed(0x1F);
  sensor.setPulseAmplitudeIR(0x1F);
  sensor.setSampleRate(SAMPLE_RATE);

  Serial.println("✅ Sensor initialized, starting stream...");
}

void loop() {
  unsigned long now = millis();

  if (now - lastSampleTime >= SAMPLE_INTERVAL && sampleIndex < WINDOW_SIZE) {
    ppg_ir[sampleIndex++] = sensor.getIR();
    lastSampleTime = now;

    // if (sampleIndex % 100 == 0) {
    //   Serial.print("📈 Sampled: ");
    //   Serial.println(sampleIndex);
    // }
  }

  if (sampleIndex == WINDOW_SIZE) {
    for (int i = 0; i < WINDOW_SIZE; i++) {
      Serial.print(ppg_ir[i]);
      if (i < WINDOW_SIZE - 1) Serial.print(",");
    }
    Serial.println();  // End of window
    sampleIndex = 0;
  }

  yield();
}
