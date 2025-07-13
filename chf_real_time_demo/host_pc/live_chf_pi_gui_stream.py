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
def verify_ppg_waveform(wave, fs=50, min_peaks=2, min_std=0.003, max_entropy=4.5, freq_range=(0.5, 2.5)):
    #return True
    if len(wave) < fs * 5: return False
    if np.std(wave) < min_std: return False
    try:
        peaks, _ = find_peaks(wave, distance=int(0.4 * fs))
    except: return False
    if len(peaks) < min_peaks: return False
    f, Pxx = welch(wave, fs=fs, nperseg=min(256, len(wave)))
    Pxx_norm = Pxx / np.sum(Pxx)
    if entropy(Pxx_norm) > max_entropy: return False
    dominant_freq = f[np.argmax(Pxx)]
    if not (freq_range[0] <= dominant_freq <= freq_range[1]): return False
    return True

# --- Load Models ---
cnn = CNNLSTM()
cnn.load_state_dict(torch.load("models/cnn_lstm_chf_model.pth"))
cnn.eval()
xgb = joblib.load("models/xgb_model.joblib")

def predict(window):
    std = np.std(window)
    feats = extract_features([window])
    if feats.shape[0] == 0: return None
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

        self.label = tk.Label(root, text="Please place your finger on the sensor and press Start", font=("Arial", 16))
        self.label.pack(pady=20)

        self.start_button = tk.Button(root, text="Start", font=("Arial", 14), command=self.start_recording)
        self.start_button.pack()

        self.continue_button = tk.Button(root, text="Continue", font=("Arial", 14), command=self.reset_ui)
        self.continue_button.pack()
        self.continue_button.pack_forget()  # hide at start

        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        self.buffer = deque(maxlen=2000)  # enough for 40 seconds
        self.recording = False
        self.countdown = 0

        self.root.after(10, self.listen_serial)

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

        self.root.after(8, self.listen_serial)

    def start_recording(self):
        self.buffer.clear()
        self.recording = True
        self.countdown = 30
        self.start_button.config(state=tk.DISABLED)
        self.label.config(text="Recording in progress: 30 sec remaining", fg="blue", bg="white")
        self.root.after(1000, self.update_countdown)

    def update_countdown(self):
        if self.countdown > 0:
            self.countdown -= 1
            self.label.config(text=f"Recording in progress: {self.countdown} sec remaining", fg="blue", bg="white")
            self.root.after(1000, self.update_countdown)
        else:
            self.analyze()

    def analyze(self):
        self.label.config(text="Analyzing...", fg="orange", bg="white")
        signal = list(self.buffer)[-1500:]  # 30 sec × 50 Hz
        signal = np.array(signal) - np.mean(signal)
        self.buffer.clear()

        if len(signal) < 1500 or not verify_ppg_waveform(signal, fs=50):
            self.label.config(text="Poor signal quality. Try again.", fg="red", bg="white")
            self.root.config(bg="white")
        else:
            result = predict(signal)
            if result == "CHF":
                self.root.config(bg="red")
                self.label.config(text="CHF Detected", fg="white", bg="red")
            elif result == "NO CHF":
                self.root.config(bg="green")
                self.label.config(text="NO CHF Detected", fg="white", bg="green")
            else:
                self.label.config(text="Analysis failed. Try again.", fg="red", bg="white")
                self.root.config(bg="white")

        self.continue_button.pack()

    def reset_ui(self):
        self.root.config(bg="white")
        self.label.config(text="Please place your finger on the sensor and press Start", fg="black", bg="white")
        self.start_button.config(state=tk.NORMAL)
        self.continue_button.pack_forget()
        self.recording = False
        self.buffer.clear()

# --- Run App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = CHFApp(root)
    root.mainloop()
