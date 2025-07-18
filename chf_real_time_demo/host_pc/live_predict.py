import serial, numpy as np, torch, joblib
import sys, os
from scipy.signal import find_peaks, welch
from scipy.stats import entropy

# Add parent path to find model pipeline
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

# --------- Inference Logic ----------
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

# --------- Serial Input ----------
ser = serial.Serial('/dev/ttyUSB0', 115200)
print("📡 Listening for PPG data...")

while True:
    try:
        line = ser.readline().decode().strip()
        if not line or not line[0].isdigit():
            continue

        vals = list(map(int, line.split(',')))
        if len(vals) != 512:
            continue

        vals_np = np.array(vals)
        is_valid, _ = verify_ppg_waveform(vals_np, fs=25, verbose=True)

        if not is_valid:
            print("⚠️ Skipped: Poor quality signal.")
            continue

        res = predict(vals_np)
        if res:
            print(f"✅ CHF Risk: {res['prob']:.2f} | Source: {res['src']} | Label: {res['label']}")
        else:
            print("⚠️ Signal weak or model rejected input.")

    except Exception as e:
        print("❌ Error:", e)
