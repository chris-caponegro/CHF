import serial, numpy as np, torch, joblib
import sys, os
from scipy.signal import find_peaks, welch
from scipy.stats import entropy
import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Add model pipeline path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from host_pc.hybrid_chf_model_pipeline import CNNLSTM, extract_features

# --------- Signal Quality Check Function ----------
def verify_ppg_waveform(wave, fs, min_peaks=3, min_std=0.01, max_entropy=3.5,
                        freq_range=(0.5, 2.5), verbose=False):
    if len(wave) < fs * 5:
        if verbose: print("❌ Too short")
        return False, 0.0
    std = np.std(wave)
    if std < min_std:
        if verbose: print("❌ Too flat")
        return False, 0.0
    try:
        peaks, _ = find_peaks(wave, distance=int(0.4 * fs))
    except:
        if verbose: print("❌ Peak detection error")
        return False, 0.0
    if len(peaks) < min_peaks:
        if verbose: print(f"❌ Too few peaks: {len(peaks)}")
        return False, 0.2
    f, Pxx = welch(wave, fs=fs, nperseg=min(256, len(wave)))
    Pxx_norm = Pxx / np.sum(Pxx)
    spec_entropy = entropy(Pxx_norm)
    if spec_entropy > max_entropy:
        if verbose: print(f"❌ Noisy (entropy = {spec_entropy:.2f})")
        return False, 0.3
    dominant_freq = f[np.argmax(Pxx)]
    if not (freq_range[0] <= dominant_freq <= freq_range[1]):
        if verbose: print(f"❌ Abnormal HR frequency = {dominant_freq:.2f} Hz")
        return False, 0.4
    return True, 1.0

# --------- Model Loading ----------
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
        src, thr = "cnn", 0.3
    else:
        prob = xgb.predict_proba(feats)[0][1]
        src, thr = "xgb", 0.5
    return {"prob": prob, "src": src, "label": int(prob >= thr)}

# --------- Tkinter UI ----------
root = tk.Tk()
root.title("CHF Risk Monitor")

# Current risk display
risk_var = tk.StringVar(value="Waiting for data...")
label_risk = tk.Label(root, textvariable=risk_var, font=("Arial", 24), fg="blue")
label_risk.pack(pady=10)

# Waveform plot
fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10)
line_plot, = ax.plot([], [], lw=1)
ax.set_title("Most Recent PPG Wave")
#ax.set_xlim(512, 0)  # Flip to show recent samples on right
ax.set_xlim(0,512)
ax.set_ylabel("IR Intensity")
ax.set_xlabel("Sample Index")

# Scrollable log
log_box = scrolledtext.ScrolledText(root, height=15, width=70, state='disabled')
log_box.pack(padx=10, pady=10)

def log_result(text):
    log_box.configure(state='normal')
    log_box.insert(tk.END, text + "\n")
    log_box.yview(tk.END)
    log_box.configure(state='disabled')

# Serial port
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

def read_serial():
    try:
        line = ser.readline().decode().strip()
        if not line or not line[0].isdigit():
            root.after(100, read_serial)
            return

        vals = list(map(int, line.split(',')))
        if len(vals) != 512:
            root.after(100, read_serial)
            return

        vals_np = np.array(vals)
        is_valid, _ = verify_ppg_waveform(vals_np, fs=96.2, verbose=False)

        # Plot the wave
        line_plot.set_data(range(512), vals_np[::-1])  # Reverse to show newest on right
        ax.set_ylim(min(vals_np) * 0.95, max(vals_np) * 1.05)
        canvas.draw()

        if not is_valid:
            risk_var.set("Signal Rejected")
            log_result("⚠️ Rejected: Poor quality signal.")
        else:
            result = predict(vals_np)
            if result:
                timestamp = datetime.now().strftime("%H:%M:%S")
                msg = f"[{timestamp}] CHF Risk: {result['prob']:.2f} | Model: {result['src']} | Label: {result['label']}"
                risk_var.set(f"{result['prob']:.2f} ({result['src'].upper()}, Label={result['label']})")
                log_result(msg)
            else:
                log_result("⚠️ Model rejected input.")

    except Exception as e:
        log_result(f"❌ Error: {e}")
    root.after(100, read_serial)

read_serial()
root.mainloop()
