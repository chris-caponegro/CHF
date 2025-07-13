import tkinter as tk
import serial
import numpy as np
import torch
import joblib
from threading import Thread
from collections import deque
from time import sleep
from hybrid_chf_model_pipeline import CNNLSTM, extract_features
from scipy.signal import find_peaks, welch
from scipy.stats import entropy

# --- Signal Quality Check ---
def verify_ppg_waveform(wave, fs=50, min_peaks=3, min_std=0.01, max_entropy=3.5, freq_range=(0.5, 2.5)):
    if len(wave) < fs * 2:
        return False
    if np.std(wave) < min_std:
        return False
    try:
        peaks, _ = find_peaks(wave, distance=int(0.4 * fs))
    except:
        return False
    if len(peaks) < min_peaks:
        return False
    f, Pxx = welch(wave, fs=fs, nperseg=min(256, len(wave)))
    Pxx_norm = Pxx / np.sum(Pxx)
    if entropy(Pxx_norm) > max_entropy:
        return False
    dominant_freq = f[np.argmax(Pxx)]
    if not (freq_range[0] <= dominant_freq <= freq_range[1]):
        return False
    return True

# --- Load Models ---
cnn = CNNLSTM()
cnn.load_state_dict(torch.load("models/cnn_lstm_chf_model.pth"))
cnn.eval()
xgb = joblib.load("models/xgb_model.joblib")

def predict(window):
    std = np.std(window)
    feats = extract_features([window])
    if feats.shape[0] == 0:
        return None
    if std >= 0.3:
        prob = cnn(torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).item()
        return "CHF" if prob >= 0.3 else "NO CHF"
    else:
        prob = xgb.predict_proba(feats)[0][1]
        return "CHF" if prob >= 0.5 else "NO CHF"

# --- GUI App ---
class CHFApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("600x300")
        self.root.title("CHF Screening")
        self.label = tk.Label(root, text="Please place your finger on the sensor.", font=("Arial", 20))
        self.label.pack(expand=True)

        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        self.buffer = deque(maxlen=2000)  # enough for 40 sec @ 50 Hz
        self.recording = False
        self.countdown = 0
        self.root.after(1000, self.listen_serial)

    def listen_serial(self):
        try:
            line = self.ser.readline().decode().strip()
            try:
                val = int(line)
                self.buffer.append(val)
            except:
                pass
        except:
            pass

        if not self.recording:
            # Use last 3 seconds (150 samples) to detect finger
            if len(self.buffer) >= 150:
                window = np.array(self.buffer)[-150:]
                window = window - np.mean(window)
                if verify_ppg_waveform(window, fs=50):
                    self.label.configure(text="Good signal detected. Starting 30s recording...", fg="blue", bg="white")
                    self.recording = True
                    self.record_start_len = len(self.buffer)
                    self.countdown = 30
                    self.root.after(1000, self.update_countdown)
                else:
                    self.label.configure(text="Please place your finger on the sensor.", fg="black", bg="white")
        else:
            self.label.configure(text=f"Recording in progress: {self.countdown} sec remaining", fg="blue", bg="white")

        self.root.after(8, self.listen_serial)

    def update_countdown(self):
        if self.countdown > 0:
            self.countdown -= 1
            self.root.after(1000, self.update_countdown)
        else:
            self.analyze()

    def analyze(self):
        self.label.configure(text="Analyzing...", fg="orange", bg="white")

        # Extract 30 seconds of data from the buffer
        signal = list(self.buffer)[-1500:]  # 30 sec × 50 Hz
        signal = np.array(signal) - np.mean(signal)
        self.buffer.clear()

        if len(signal) < 1500 or not verify_ppg_waveform(signal, fs=50):
            self.label.configure(text="Poor signal quality. Try again.", fg="red", bg="white")
        else:
            result = predict(signal)
            if result == "CHF":
                self.root.configure(bg="red")
                self.label.configure(text="CHF Detected", fg="white", bg="red")
            elif result == "NO CHF":
                self.root.configure(bg="green")
                self.label.configure(text="NO CHF Detected", fg="white", bg="green")
            else:
                self.label.configure(text="Analysis failed. Try again.", fg="red", bg="white")

        self.recording = False
        self.root.after(5000, self.reset_ui)

    def reset_ui(self):
        self.root.configure(bg="white")
        self.label.configure(text="Please place your finger on the sensor.", fg="black", bg="white")

# --- Run App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = CHFApp(root)
    root.mainloop()
