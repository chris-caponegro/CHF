import os
import wfdb
import numpy as np
import traceback

# === CONFIG ===
input_root = 'Downloads/CHF/p00'
output_root = 'Downloads/PLETH_CSV'
output_suffix = '_pleth.csv'

os.makedirs(output_root, exist_ok=True)

# === Go through each patient folder ===
for patient_folder in sorted(os.listdir(input_root)):
    input_patient_path = os.path.join(input_root, patient_folder)
    if not os.path.isdir(input_patient_path):
        continue

    output_patient_path = os.path.join(output_root, patient_folder)
    os.makedirs(output_patient_path, exist_ok=True)

    print(f"🔍 Processing patient folder: {patient_folder}")

    for file in os.listdir(input_patient_path):
        if not file.endswith('.hea') or '_layout' in file:
            continue

        record_name = file.replace('.hea', '')
        input_record_path = os.path.join(input_patient_path, record_name)
        output_file = os.path.join(output_patient_path, record_name + output_suffix)

        if os.path.exists(output_file):
            print(f"⏩ Already extracted: {output_file}")
            continue

        try:
            # Read full record metadata to check channels
            record = wfdb.rdrecord(input_record_path)

            # Skip layout or zero-length signals
            if record.sig_len == 0:
                print(f"⚠️ Empty signal in {record_name}, skipping.")
                continue

            if 'PLETH' not in record.sig_name:
                print(f"⚠️ No PLETH in {record_name}, skipping.")
                continue

            pleth_index = record.sig_name.index('PLETH')
            pleth_signal = record.p_signal[:, pleth_index]

            np.savetxt(output_file, pleth_signal, delimiter=',')
            print(f"✅ Saved: {output_file}")

        except Exception as e:
            print(f"❌ Error processing {record_name}: {e}")
            traceback.print_exc()
