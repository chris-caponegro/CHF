import time
import serial

# --- Settings ---
SERIAL_PORT = "/dev/ttyUSB0"  # Update if needed
BAUD_RATE = 115200
WINDOW_SIZE = 512

# --- Initialize Serial ---
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
print("📡 Listening for PPG data...")

timestamps = []
buffer = []

try:
    while True:
        line = ser.readline().decode().strip()
        try:
            val = int(line)
            buffer.append(val)
        except ValueError:
            continue

        if len(buffer) == WINDOW_SIZE:
            now = time.time()
            timestamps.append(now)

            if len(timestamps) > 1:
                duration = timestamps[-1] - timestamps[-2]
                rate = WINDOW_SIZE / duration
                print(f"📦 512 samples in {duration:.2f} sec → ~{rate:.1f} Hz")

            buffer = []

except KeyboardInterrupt:
    print("⛔ Exiting.")
    ser.close()
except Exception as e:
    print(f"❌ Error: {e}")
    ser.close()
