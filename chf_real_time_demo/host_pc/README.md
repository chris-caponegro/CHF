# CHF Real-Time Demo

## Setup

1. Copy `cnn_lstm_chf_model.pth` and `xgb_model.joblib` to `models/`.
2. Adjust serial port in `live_predict.py`.

## Run

- Upload `esp32/main.ino` via Arduino IDE or PlatformIO.
- On your PC:
  ```bash
  cd host_pc
  pip install -r requirements.txt
  python live_predict.py
