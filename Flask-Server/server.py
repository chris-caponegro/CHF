from flask import Flask, request, jsonify, render_template_string
import json
import time
import numpy as np

app = Flask(__name__)

VALID_KEYS = {
    "sk-Rtv9JjbSckCeiIJ6wHD0M8mqBgtbHoQQ": "esp32-001"
}

DATA_FILE = "ppg_log.jsonl"

dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>PPG Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h2>Latest PPG Heartwave (IR & RED)</h2>
    <canvas id="ppgChart" width="800" height="400"></canvas>
    <script>
        async function fetchData() {
            const res = await fetch('/data');
            const json = await res.json();
            return json;
        }

        async function drawChart() {
            const data = await fetchData();
            const ctx = document.getElementById('ppgChart').getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: data.ir.length}, (_, i) => i),
                    datasets: [
                        {
                            label: 'IR (heartwave)',
                            data: data.ir,
                            borderWidth: 1,
                            borderColor: 'red',
                            fill: false
                        },
                        {
                            label: 'RED (heartwave)',
                            data: data.red,
                            borderWidth: 1,
                            borderColor: 'blue',
                            fill: false
                        }
                    ]
                },
                options: {
                    animation: false,
                    responsive: true,
                    scales: {
                        x: { display: true },
                        y: { display: true }
                    }
                }
            });
        }
        drawChart();
        setInterval(drawChart, 5000);
    </script>
</body>
</html>
"""

def preprocess(signal):
    centered = np.array(signal) - np.mean(signal)
    return centered.tolist()

@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = request.get_json(force=True)
        api_key = data.get("api_key")
        device_id = data.get("device_id")

        if api_key not in VALID_KEYS or VALID_KEYS[api_key] != device_id:
            return "Unauthorized", 401

        entry = {
            "timestamp": int(time.time()),
            "device_id": device_id,
            "ir": data["ppg_ir"],
            "red": data["ppg_red"]
        }
        with open(DATA_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return "OK", 200

    except Exception as e:
        print("❌ Failed to process request:", e)
        return "Bad request", 400

@app.route("/dashboard")
def dashboard():
    return render_template_string(dashboard_html)

@app.route("/data")
def data():
    try:
        with open(DATA_FILE, "r") as f:
            lines = f.readlines()
            last = json.loads(lines[-1])
            ir_wave = preprocess(last["ir"])
            red_wave = preprocess(last["red"])
            return jsonify({"ir": ir_wave, "red": red_wave})
    except:
        return jsonify({"ir": [], "red": []})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
