import tkinter as tk
import serial
import threading
import time
import numpy as np
from hybrid_chf_model_pipeline import CNNLSTM, extract_features
from scipy.signal import find_peaks, welch
from scipy.stats import entropy
import torch
import joblib

# ---------- CONFIG ----------
FS = 50  # Sample rate
SAMPLE_DURATION = 3  # seconds
SAMPLE_SIZE = FS * SAMPLE_DURATION

# ---------- Load Models ----------
cnn_model = CNNLSTM()
cnn_model.load_state_dict(torch.load("models/cnn_lstm_chf_model.pth"))
cnn_model.eval()
xgb_model = joblib.load("models/xgb_model.joblib")

def verify_ppg_waveform(wave, fs=FS, min_peaks=2, min_std=0.005, max_entropy=4.5):
    if len(wave) < fs * 2:
        return False
    wave = wave - np.mean(wave)
    std = np.std(wave)
    if std < min_std:
        return False
    try:
        peaks, _ = find_peaks(wave, distance=int(0.4 * fs))
    except:
        return False
    if len(peaks) < min_peaks:
        return False
    f, Pxx = welch(wave, fs=fs, nperseg=min(128, len(wave)))
    Pxx_norm = Pxx / np.sum(Pxx)
    if entropy(Pxx_norm) > max_entropy:
        return False
    dominant_freq = f[np.argmax(Pxx)]
    if not (0.5 <= dominant_freq <= 2.5):
        return False
    return True

def predict(window):
    std = np.std(window)
    feats = extract_features([window])
    if feats.shape[0] == 0:
        return None
    if std >= 0.3:
        prob = cnn_model(torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).item()
        src, thr = "cnn", 0.3
    else:
        prob = xgb_model.predict_proba(feats)[0][1]
        src, thr = "xgb", 0.5
    return {"prob": prob, "src": src, "label": int(prob >= thr)}

class CHFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CHF Detection")
        self.label = tk.Label(root, text="Please place your finger on the sensor.", font=("Arial", 18))
        self.label.pack(padx=20, pady=20)
        self.ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
        self.buffer = []
        self.running = True
        threading.Thread(target=self.listen_serial).start()

    def listen_serial(self):
        while self.running:
            try:
                line = self.ser.readline().decode().strip()
                val = int(line)
                self.buffer.append(val)
                if len(self.buffer) >= SAMPLE_SIZE:
                    self.handle_window()
            except:
                continue

    def handle_window(self):
        window = np.array(self.buffer[:SAMPLE_SIZE])
        self.buffer = []
        if verify_ppg_waveform(window):
            self.label.config(text="Analyzing...", fg="black", bg="white")
            self.root.update()
            time.sleep(1)
            res = predict(window)
            if res:
                if res['label'] == 0:
                    self.label.config(text="NO CHF Detected", fg="white", bg="green")
                else:
                    self.label.config(text="CHF Detected", fg="white", bg="red")
            else:
                self.label.config(text="Analysis failed", fg="white", bg="orange")
        else:
            self.label.config(text="Poor signal. Please reposition finger.", fg="white", bg="gray")

        self.root.update()
        time.sleep(5)
        self.label.config(text="Please place your finger on the sensor.", fg="black", bg="white")
        self.buffer = []

if __name__ == "__main__":
    root = tk.Tk()
    app = CHFApp(root)
    root.mainloop()
