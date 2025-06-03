import requests
from bs4 import BeautifulSoup
import time
import csv

# Construct MIMIC-IV waveform path using 8-digit subject_id with 'p' prefix
def subject_to_path(subject_id):
    padded = str(subject_id).zfill(8)  # Ensure 8 digits
    folder = f'p{padded[:3]}'          # p100, p101, etc.
    subfolder = f'p{padded}'           # p10014354, etc.
    return folder, subfolder

# Check if any .hea file for a subject contains "PLETH"
def check_pleth_in_hea(subject_id):
    folder, subfolder = subject_to_path(subject_id)
    base_url = f'https://physionet.org/content/mimic4wdb/0.1.0/waves/{folder}/{subfolder}/'
    print(base_url)

    try:
        response = requests.get(base_url)
        if response.status_code != 200:
            print(f"Subject folder {subfolder} not found.")
            return False

        soup = BeautifulSoup(response.text, 'html.parser')
        subdirs = [a['href'].strip('/') for a in soup.find_all('a', href=True) if a['href'].strip('/').isdigit()]

        # Loop through timestamp subfolders (like 84050536)
        for subdir in subdirs:
            subdir_url = base_url.replace('/content/', '/files/') + f'{subdir}/'
            index_url = f'{subdir_url}?download'
            index_response = requests.get(index_url)
            if index_response.status_code != 200:
                continue

            # Fetch .hea files from subdir
            subdir_page = requests.get(base_url + f'{subdir}/')
            sub_soup = BeautifulSoup(subdir_page.text, 'html.parser')
            hea_files = [a['href'] for a in sub_soup.find_all('a', href=True) if a['href'].endswith('.hea')]

            for hea_file in hea_files:
                hea_url = subdir_url + hea_file
                hea_response = requests.get(hea_url)
                if hea_response.status_code != 200:
                    continue

                if 'PLETH' in hea_response.text.upper():
                    return True

        return False

    except Exception as e:
        print(f"Error checking subject {subject_id}: {e}")
        return False

# === Load SUBJECT_IDs from CSV (BigQuery export) ===
chf_subjects = []
with open('Downloads/CHF_BQ_mimic4.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        chf_subjects.append(int(row[0]))

# === Check each subject and log result ===
results = []
for subject_id in chf_subjects:
    print(f"Checking subject {subject_id}...")
    has_pleth = check_pleth_in_hea(subject_id)
    results.append((subject_id, has_pleth))
    time.sleep(0.5)  # Rate limit requests

# === Save results ===
with open('Downloads/CHF_with_pleth_mimic4.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['SUBJECT_ID', 'HAS_PPG'])
    writer.writerows(results)

print("Done. Results saved to chf_subjects_with_pleth.csv")
