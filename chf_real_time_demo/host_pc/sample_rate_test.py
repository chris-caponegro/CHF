import serial
import time

# --- Adjust this to your actual port ---
SERIAL_PORT = '/dev/ttyUSB0'  # or COM3 on Windows
BAUD_RATE = 115200

# --- Open serial connection ---
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
print("⏱️ Waiting for 512-sample windows...")

# --- Initialize ---
prev_time = time.time()
window_count = 0

try:
    while True:
        line = ser.readline().decode().strip()
        if line.count(',') == 511:  # Window of 512 samples
            now = time.time()
            duration = now - prev_time
            prev_time = now
            window_count += 1
            print(f"📦 Window {window_count}: {duration:.2f} sec → ~{512/duration:.1f} Hz")
except KeyboardInterrupt:
    ser.close()
    print("🛑 Exiting.")
