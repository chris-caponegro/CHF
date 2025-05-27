import os
import csv
import time
import datetime
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# === CONFIG ===
input_csv = 'Downloads/CHF_with_pleth_mimic4.csv'
output_root = 'Downloads/CHF'
BASE_URL = 'https://physionet.org/files/mimic3wdb-matched/1.0'

# 🔧 Define folder prefixes to include (e.g., ['p00', 'p01', 'p02'])
folder_whitelist = ['p00', 'p01', 'p02']

# === Robust session with retries ===
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount('http://', adapter)
session.mount('https://', adapter)

# === Utilities ===
def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

def subject_to_path(subject_id):
    padded = str(subject_id).zfill(6)
    folder = f"p{padded[:2]}"
    subfolder = f"p{padded}"
    return folder, subfolder

def download_waveform_files(subject_id):
    folder, subfolder = subject_to_path(subject_id)

    # ✅ Skip folders not in whitelist
    if folder not in folder_whitelist:
        log(f"🚫 Skipping {subject_id} (folder {folder} not in whitelist)")
        return

    base_url = f'{BASE_URL}/{folder}/{subfolder}/'
    local_dir = os.path.join(output_root, folder, subfolder)
    os.makedirs(local_dir, exist_ok=True)

    log(f"🔍 Checking {subject_id} at {base_url}")
    try:
        response = session.get(base_url, timeout=10)
        if response.status_code != 200:
            log(f"⚠️  Could not access {base_url}")
            return

        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        hea_files = [l for l in links if l.endswith('.hea')]

        downloaded = 0
        for hea_file in hea_files:
            hea_url = base_url + hea_file
            hea_response = session.get(hea_url, timeout=10)
            if hea_response.status_code != 200:
                continue

            if 'PLETH' in hea_response.text:
                prefix = hea_file.replace('.hea', '')
                matched_files = [l for l in links if l.startswith(prefix)]

                for file in matched_files:
                    file_url = base_url + file
                    local_path = os.path.join(local_dir, file)
                    if not os.path.exists(local_path):
                        try:
                            with session.get(file_url, stream=True, timeout=30) as r:
                                with open(local_path, 'wb') as f:
                                    for chunk in r.iter_content(chunk_size=8192):
                                        f.write(chunk)
                        except Exception as e:
                            log(f"⚠️ Failed to download {file_url}: {e}")
                    else:
                        log(f"↪️ Already exists: {file}")
                downloaded += 1
                log(f"✅ Downloaded {prefix} with PLETH")

        if downloaded == 0:
            log(f"❌ No PLETH waves found for {subject_id}")

    except Exception as e:
        log(f"❌ Error downloading for subject {subject_id}: {e}")

# === Load subjects from CSV ===
subject_ids = []
with open(input_csv, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if len(row) >= 2 and row[1].strip().lower() == 'true':
            try:
                subject_ids.append(int(row[0].strip()))
            except ValueError:
                log(f"⚠️ Skipping invalid subject ID row: {row}")

log(f"📋 Loaded {len(subject_ids)} subject IDs from {input_csv}")

# === Main loop ===
for subj in subject_ids:
    try:
        download_waveform_files(subj)
    except Exception as e:
        log(f"❌ Crash on subject {subj}: {e}")
    time.sleep(0.5)  # polite to PhysioNet
