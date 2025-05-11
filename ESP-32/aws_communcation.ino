#include <WiFiManager.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include "MAX30105.h"
#include <time.h>

MAX30105 sensor;

const char* serverName = "https://oximetr.xyz/submit";
const char* device_id = "esp32-001";
const char* api_key = "sk-Rtv9JjbSckCeiIJ6wHD0M8mqBgtbHoQQ"; //Needs to be changed eventually

#define SAMPLE_RATE      25
#define RECORD_DURATION  5
#define POST_INTERVAL    10000
#define SAMPLE_INTERVAL  (1000 / SAMPLE_RATE)
#define BUFFER_SIZE      (SAMPLE_RATE * RECORD_DURATION)

long ppg_ir[BUFFER_SIZE];
long ppg_red[BUFFER_SIZE];

unsigned long lastSampleTime = 0;
unsigned long lastPostTime = 0;
int sampleIndex = 0;

void sendDataBuffer() {
  time_t now = time(nullptr);

  String json = "{";
  json += "\"timestamp\":" + String(now) + ",";
  json += "\"device_id\":\"" + String(device_id) + "\",";
  json += "\"api_key\":\"" + String(api_key) + "\",";
  json += "\"ppg_ir\":[";
  for (int i = 0; i < sampleIndex; i++) {
    json += String(ppg_ir[i]);
    if (i < sampleIndex - 1) json += ",";
  }
  json += "],\"ppg_red\":[";
  for (int i = 0; i < sampleIndex; i++) {
    json += String(ppg_red[i]);
    if (i < sampleIndex - 1) json += ",";
  }
  json += "]}";

  HTTPClient http;
  http.begin(serverName);
  http.addHeader("Content-Type", "application/json");

  int code = http.POST(json);
  Serial.print("📡 POST result: ");
  Serial.println(code);
  http.end();
  sampleIndex = 0;
}

void setup() {
  Serial.begin(115200);
  WiFiManager wm;
  wm.autoConnect("ESP32-Setup", "setup123");

  configTime(0, 0, "pool.ntp.org");
  while (time(nullptr) < 100000) delay(500);

  if (!sensor.begin()) {
    Serial.println("❌ MAX30102 not found");
    while (true);
  }

  sensor.setup();
  Serial.println("✅ Sensor ready");
}

void loop() {
  unsigned long now = millis();

  if (now - lastSampleTime >= SAMPLE_INTERVAL && sampleIndex < BUFFER_SIZE) {
    ppg_ir[sampleIndex] = sensor.getIR();
    ppg_red[sampleIndex] = sensor.getRed();
    sampleIndex++;
    lastSampleTime = now;

    if (sampleIndex % 25 == 0) {
      Serial.print("📈 Sampled: ");
      Serial.println(sampleIndex);
    }
    yield();
  }

  if (now - lastPostTime >= POST_INTERVAL && sampleIndex >= BUFFER_SIZE) {
    sendDataBuffer();
    lastPostTime = now;
  }

  yield();
}
