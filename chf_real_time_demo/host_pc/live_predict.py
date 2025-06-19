import serial, numpy as np, torch, joblib
from hybrid_chf_model_pipeline import CNNLSTM, extract_features
print("what")
# Load models
cnn = CNNLSTM(); cnn.load_state_dict(torch.load("models/cnn_lstm_chf_model.pth")); cnn.eval()
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

# Setup serial (adjust COM/tty port accordingly)
ser = serial.Serial('/dev/ttyUSB0', 115200)
print("Listening...")

while True:
    line = ser.readline().decode().strip()
    vals = list(map(int, line.split(',')))
    if len(vals) == 512:
        res = predict(np.array(vals))
        if res:
            print(f"✅ CHF Risk: {res['prob']:.2f} | {res['src']} | Label: {res['label']}")
        else:
            print("⚠️ Signal too weak or invalid.")
