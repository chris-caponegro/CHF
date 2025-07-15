# hybrid_chf_model_pipeline.py

"""
This script collection includes:
1. Preprocessing raw PPG .npy files
2. Feature extraction
3. Training XGBoost (tabular model)
4. Training CNN+LSTM (deep learning model)
5. Inference hybrid logic
6. Utility to generate 60-second 1D PPG .npy files (25Hz)
7. Extract PPG from .dat/.hea files into raw .npy for training
"""
# hybrid_chf_model_pipeline.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.fft import fft

# --- STEP 1: PPG windowing ---
def window_ppg(ppg, window_size=512):
    n_windows = len(ppg) // window_size
    return np.array(np.split(ppg[:n_windows * window_size], n_windows))

# --- STEP 2: Feature extraction for XGBoost ---
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import numpy as np

def extract_features(windows, fs=50):
    features = []
    for w in windows:
        if len(w) < 10 or np.std(w) < 1e-5 or np.isnan(w).any():
            continue

        mean_ppg = np.mean(w)
        std_ppg = np.std(w)
        skew_ppg = skew(w)
        kurt_ppg = kurtosis(w)

        # Time-domain peak features
        peaks, _ = find_peaks(w)
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / fs  # convert to seconds
            mean_interval = np.mean(peak_intervals)
            std_interval = np.std(peak_intervals)
            min_interval = np.min(peak_intervals)
            max_interval = np.max(peak_intervals)
        else:
            mean_interval = std_interval = min_interval = max_interval = 0

        # Frequency-domain features
        N = len(w)
        yf = np.abs(fft(w))
        xf = fftfreq(N, 1/fs)
        dominant_freq = xf[np.argmax(yf[:N // 2])]
        spectral_entropy = -np.sum((yf / np.sum(yf)) * np.log2(yf / np.sum(yf) + 1e-9))

        features.append([
            mean_ppg, std_ppg, skew_ppg, kurt_ppg,
            mean_interval, std_interval, min_interval, max_interval,
            dominant_freq, spectral_entropy
        ])
    return np.array(features)


# --- STEP 3: Train XGBoost model on safe batches ---
def train_xgboost_safe(X, y, batch_size=100):
    X_feat_all = []
    y_all = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        Xi = X[start:end]
        yi = y[start:end]

        feats = extract_features(Xi)
        if feats.shape[0] == 0 or len(np.unique(yi[:len(feats)])) < 2:
            continue

        X_feat_all.append(feats)
        y_all.append(yi[:len(feats)])

    X_clean = np.vstack(X_feat_all)
    y_clean = np.concatenate(y_all)

    X_clean = np.clip(X_clean, -1e3, 1e3)
    print(f"✅ Training XGBoost on {X_clean.shape[0]} safe samples...")

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', nthread=1, verbosity=1)
    model.fit(X_clean, y_clean)
    return model

# --- STEP 4: Define CNN+LSTM model ---
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        return self.classifier(hn[-1])

# --- STEP 5: Train CNN+LSTM ---
def train_cnn_lstm(X, y, epochs=10, batch_size=32):
    model = CNNLSTM()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    return model

# --- STEP 6: Hybrid Inference ---
def hybrid_predict(ppg_window, xgb_model, cnn_model, threshold_quality=0.3):
    features = extract_features([ppg_window], fs=50)
    if features.shape[0] == 0:
        return 0.0
    quality_score = features[0][1]  # std_ppg
    if quality_score >= threshold_quality:
        cnn_model.eval()
        with torch.no_grad():
            inp = torch.tensor(ppg_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            pred = cnn_model(inp).item()
    else:
        pred = xgb_model.predict_proba(features)[0][1]
    return pred

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("📂 Loading dataset...")
    X = np.load("X_windows.npy")
    y = np.load("y_labels.npy")

    print("✂️ Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("🌲 Training XGBoost on safe batches...")
    xgb_model = train_xgboost_safe(X_train, y_train)
    joblib.dump(xgb_model, "xgb_model.joblib")
    print("✅ Saved: xgb_model.joblib")

    print("🧠 Training CNN+LSTM...")
    cnn_model = train_cnn_lstm(X_train, y_train)
    torch.save(cnn_model.state_dict(), "cnn_lstm_chf_model.pth")
    print("✅ Saved: cnn_lstm_chf_model.pth")

    print("\n📈 Evaluating CNN+LSTM...")
    cnn_model.eval()
    with torch.no_grad():
        preds_cnn = cnn_model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)).squeeze().numpy()
    bin_cnn = (preds_cnn > 0.5).astype(int)
    print(classification_report(y_test, bin_cnn, digits=3))
    print("ROC-AUC:", roc_auc_score(y_test, preds_cnn))

    print("\n📈 Evaluating XGBoost...")
    X_test_feat = extract_features(X_test)
    preds_xgb = xgb_model.predict_proba(X_test_feat)[:, 1]
    bin_xgb = (preds_xgb > 0.5).astype(int)
    print(classification_report(y_test[:len(preds_xgb)], bin_xgb, digits=3))
    print("ROC-AUC:", roc_auc_score(y_test[:len(preds_xgb)], preds_xgb))
