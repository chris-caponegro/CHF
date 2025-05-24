# This is a script that will look through patients ids from the mimic 3 big query 
# and determine if they have a pleth wave or not. Results will be saved in Downloads
# Note: you can search for CHF and healthy pateints just change the file paths (:
import requests
from bs4 import BeautifulSoup
import re
import time
import csv

input_file = 'Downloads/CHF_BQ_mimic3.csv'
input_file = 'Downloads/healthy_BQ_mimic3.csv'

output_file = 'Downloads/CHF_with_pleth_mimic4.csv'
output_file = 'Downloads/healthy_with_pleth_mimic4.csv'


def subject_to_path(subject_id):
    padded = str(subject_id).zfill(6)
    folder = f'p{padded[:2]}'
    subfolder = f'p{padded}'
    return folder, subfolder

def check_pleth_in_hea(subject_id):
    folder, subfolder = subject_to_path(subject_id)
    url = f'https://physionet.org/files/mimic3wdb-matched/1.0/{folder}/{subfolder}/'

    try:
        response = requests.get(url)
        if response.status_code != 200:
            return False

        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        hea_files = [l for l in links if l.endswith('.hea')]

        for hea_file in hea_files:
            hea_url = url + hea_file
            hea_response = requests.get(hea_url)
            if hea_response.status_code != 200:
                continue

            if 'PLETH' in hea_response.text:
                return True

        return False

    except Exception as e:
        print(f"Error checking subject {subject_id}: {e}")
        return False

# === Load your CHF SUBJECT_IDs from CSV ===
chf_subjects = []
with open(input_file, 'r') as f: #Downloads/healthy_BQ_mimic3.csv
    reader = csv.reader(f)
    next(reader)  # Skip header row
    for row in reader:
        chf_subjects.append(int(row[0]))

# === Check and log results ===
results = []
for subject_id in chf_subjects:
    print(f"Checking {subject_id}...")
    has_pleth = check_pleth_in_hea(subject_id)
    results.append((subject_id, has_pleth))
    time.sleep(0.5)  # be polite to PhysioNet servers

# === Save results ===
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['SUBJECT_ID', 'HAS_PPG'])
    writer.writerows(results)

print("Finished. Results saved to %d", output_file)
