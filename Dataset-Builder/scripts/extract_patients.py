import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Initialize counters
chf_count = 0
healthy_count = 0

# Define the path to your CSV files
data_path = 'physionet.org/files/mimiciii/1.4'  # Replace with your actual path

# Load CSV files
print("Loading CSV files...")
patients = pd.read_csv(os.path.join(data_path, 'PATIENTS.csv'))
diagnoses = pd.read_csv(os.path.join(data_path, 'DIAGNOSES_ICD.csv'))
labevents = pd.read_csv(os.path.join(data_path, 'LABEVENTS.csv'))
d_labitems = pd.read_csv(os.path.join(data_path, 'D_LABITEMS.csv'))
chartevents = pd.read_csv(os.path.join(data_path, 'CHARTEVENTS.csv'))
d_items = pd.read_csv(os.path.join(data_path, 'D_ITEMS.csv'))

# Identify BNP ITEMID(s)
print("Identifying BNP ITEMIDs...")
bnp_itemids = d_labitems[d_labitems['LABEL'].str.contains('BNP', case=False, na=False)]['ITEMID'].tolist()
print(f"BNP ITEMIDs: {bnp_itemids}")

# Filter LABEVENTS for BNP measurements
print("Filtering LABEVENTS for BNP measurements...")
bnp_measurements = labevents[labevents['ITEMID'].isin(bnp_itemids)].copy()
bnp_measurements = bnp_measurements[pd.to_numeric(bnp_measurements['VALUENUM'], errors='coerce').notnull()]
bnp_measurements['VALUENUM'] = bnp_measurements['VALUENUM'].astype(float)

# Identify CHF patients (ICD-9 codes starting with '428')
print("Identifying CHF patients...")
chf_diagnoses = diagnoses[diagnoses['ICD9_CODE'].astype(str).str.startswith('428')]
chf_patient_ids = chf_diagnoses['SUBJECT_ID'].unique()

# Identify healthy patients (no CHF diagnosis)
print("Identifying healthy patients...")
all_patient_ids = patients['SUBJECT_ID'].unique()
healthy_patient_ids = np.setdiff1d(all_patient_ids, chf_patient_ids)

# Define ITEMIDs for required measurements
print("Defining ITEMIDs for required measurements...")
spo2_itemids = d_items[d_items['LABEL'].str.contains('O2 Saturation', case=False, na=False)]['ITEMID'].tolist()
heart_rate_itemids = d_items[d_items['LABEL'].str.contains('Heart Rate', case=False, na=False)]['ITEMID'].tolist()
perfusion_index_itemids = d_items[d_items['LABEL'].str.contains('Perfusion Index', case=False, na=False)]['ITEMID'].tolist()
pulse_amplitude_itemids = d_items[d_items['LABEL'].str.contains('Pulse Amplitude', case=False, na=False)]['ITEMID'].tolist()

required_itemids = spo2_itemids + heart_rate_itemids + perfusion_index_itemids + pulse_amplitude_itemids

# Filter CHARTEVENTS for required measurements
print("Filtering CHARTEVENTS for required measurements...")
oximeter_data = chartevents[chartevents['ITEMID'].isin(required_itemids)].copy()

# Create output directories
output_dir = 'Dataset'
chf_dir = os.path.join(output_dir, 'CHF')
healthy_dir = os.path.join(output_dir, 'healthy')
os.makedirs(chf_dir, exist_ok=True)
os.makedirs(healthy_dir, exist_ok=True)

# Function to process and save patient data
def process_patient(subject_id, category):
    # Get BNP measurements for the patient
    bnp_vals = bnp_measurements[bnp_measurements['SUBJECT_ID'] == subject_id]
    if bnp_vals.empty:
        return False
    # Get oximeter data for the patient
    patient_oximeter = oximeter_data[oximeter_data['SUBJECT_ID'] == subject_id]
    if patient_oximeter.empty:
        return False
    # Merge data
    merged = patient_oximeter.copy()
    merged['BNP'] = bnp_vals['VALUENUM'].values[0]  # Assuming the first BNP value
    # Map ITEMIDs to labels
    itemid_label_map = d_items.set_index('ITEMID')['LABEL'].to_dict()
    merged['LABEL'] = merged['ITEMID'].map(itemid_label_map)
    # Pivot data to have measurements as columns
    pivot = merged.pivot_table(index='CHARTTIME', columns='LABEL', values='VALUENUM', aggfunc='first')
    pivot.reset_index(inplace=True)
    pivot['BNP'] = bnp_vals['VALUENUM'].values[0]
    # Save to CSV
    filename = os.path.join(chf_dir if category == 'CHF' else healthy_dir, f'{subject_id}.csv')
    pivot.to_csv(filename, index=False)
    return True

# Process CHF patients
print("Processing CHF patients...")
for subject_id in tqdm(chf_patient_ids):
    if process_patient(subject_id, 'CHF'):
        chf_count += 1

# Process healthy patients
print("Processing healthy patients...")
for subject_id in tqdm(healthy_patient_ids):
    if process_patient(subject_id, 'healthy'):
        healthy_count += 1


print("Data extraction and organization complete.")
print(f"Total CHF patients recorded: {chf_count}")
print(f"Total healthy patients recorded: {healthy_count}")