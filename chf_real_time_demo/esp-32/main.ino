#include <Wire.h>
#include "MAX30105.h"
MAX30105 particleSensor;
const int windowSize = 512;
uint16_t buffer[windowSize];
int idx = 0;

void setup() {
  Serial.begin(115200);
  if (!particleSensor.begin(Wire)) {
    Serial.println("MAX30102 not found.");
    while (1);
  }
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.setSampleRate(125);
  delay(1000);
}

void loop() {
  if (particleSensor.available()) {
    int ir = particleSensor.getIR();
    buffer[idx++] = ir;
    if (idx >= windowSize) {
      for (int i = 0; i < windowSize; i++) {
        Serial.print(buffer[i]);
        if (i < windowSize - 1) Serial.print(",");
      }
      Serial.println();
      idx = 0;
    }
    particleSensor.nextSample();
  }
}
