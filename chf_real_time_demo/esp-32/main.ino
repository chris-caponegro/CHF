#include <Wire.h>
#include "MAX30105.h"

MAX30105 sensor;

#define SAMPLE_RATE      125           // Desired sampling frequency
#define PULSE_WIDTH      69            // Shortest (fastest) pulse
#define ADC_RESOLUTION   16            // Fastest ADC conversion
#define WINDOW_SIZE      512           // Model expects 512-sample windows
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

  // -------- Sensor Configuration --------
  // setPulseWidth() indirectly sets ADC resolution.
  sensor.setup();  // Basic default setup (required)

  sensor.setPulseAmplitudeIR(0x1F);    // IR LED brightness
  sensor.setPulseAmplitudeRed(0x00);   // Disable red LED
  sensor.setSampleRate(SAMPLE_RATE);   // 125 Hz
  sensor.setPulseWidth(PULSE_WIDTH);   // 69 μs for fastest ADC response

  Serial.println("✅ Sensor configured for 125 Hz. Streaming...");
}


void loop() {
  unsigned long now = millis();

  if (now - lastSampleTime >= SAMPLE_INTERVAL && sampleIndex < WINDOW_SIZE) {
    ppg_ir[sampleIndex++] = sensor.getIR();
    lastSampleTime = now;
  }

  if (sampleIndex == WINDOW_SIZE) {
    for (int i = 0; i < WINDOW_SIZE; i++) {
      Serial.print(ppg_ir[i]);
      if (i < WINDOW_SIZE - 1) Serial.print(",");
    }
    Serial.println();  // End of window
    sampleIndex = 0;
  }

  yield();  // Allow background tasks
}
