import os
import requests
from bs4 import BeautifulSoup
import time

# === CONFIG ===
input_csv = 'Downloads/CHF_with_pleth_mimic4.csv'   # or Downloads/healthy_with_pleth_mimic4.csv
output_root = 'Downloads/CHF'                       # or Downloads/healthy

# === PHYSIONET SETUP ===
BASE_URL = 'https://physionet.org/files/mimic3wdb-matched/1.0'

def subject_to_path(subject_id):
    padded = str(subject_id).zfill(6)
    return f"p{padded[:2]}", f"p{padded}"

def download_waveform_files(subject_id):
    folder, subfolder = subject_to_path(subject_id)
    base_url = f'{BASE_URL}/{folder}/{subfolder}/'
    local_dir = os.path.join(output_root, folder, subfolder)
    os.makedirs(local_dir, exist_ok=True)

    print(f"🔍 Checking {subject_id} at {base_url}")
    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"⚠️  Failed to access: {base_url}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    hea_files = [l for l in links if l.endswith('.hea')]

    for hea_file in hea_files:
        hea_url = base_url + hea_file
        hea_response = requests.get(hea_url)
        if hea_response.status_code != 200:
            continue

        if 'PLETH' in hea_response.text:
            prefix = hea_file.replace('.hea', '')
            matched_files = [l for l in links if l.startswith(prefix)]

            for file in matched_files:
                file_url = base_url + file
                local_path = os.path.join(local_dir, file)
                if not os.path.exists(local_path):
                    with requests.get(file_url, stream=True) as r:
                        with open(local_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
            print(f"✅ Downloaded {prefix} with PLETH")
        time.sleep(0.25)

# === Load subject IDs from CSV ===
subject_ids = []
with open(input_csv, 'r') as f:
    next(f)  # skip header
    for line in f:
        subj_id, has_ppg = line.strip().split(',')
        if has_ppg.lower() == 'true':
            subject_ids.append(int(subj_id))

# === Process each subject ===
for subj in subject_ids:
    download_waveform_files(subj)
    time.sleep(1)  # polite to PhysioNet
