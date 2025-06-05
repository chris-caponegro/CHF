import wfdb
import numpy as np
import os
import json

def is_valid_waveform(wave, fs, min_duration_sec=10):
    if wave is None or len(wave) < min_duration_sec * fs:
        return False
    if np.std(wave) < 0.01:
        return False
    return True

def clean_and_verify_wave(record_path, output_dir, patient_id, record_file):
    try:
        record = wfdb.rdrecord(record_path)
        fs = record.fs
        signals = record.p_signal
        signal_names = record.sig_name

        for i, name in enumerate(signal_names):
            if 'pleth' in name.lower():
                pleth_wave = signals[:, i]
                if is_valid_waveform(pleth_wave, fs):
                    print(f"✅ Valid Pleth: {record_file} (Patient: {patient_id})")

                    out_fname = f"{patient_id}_{record_file}_pleth.npy"
                    np.save(os.path.join(output_dir, out_fname), pleth_wave)

                    return {
                        "patient_id": patient_id,
                        "record": record_file,
                        "fs": fs,
                        "length": len(pleth_wave),
                        "duration_sec": len(pleth_wave) / fs
                    }
                else:
                    print(f"⚠️ Invalid Pleth: {record_file}")
        return None
    except Exception as e:
        print(f"❌ Error in {record_path}: {e}")
        return None

def process_mimic_structure(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".hea") and "layout" not in file:
                record_file = os.path.splitext(file)[0]
                record_path = os.path.join(root, record_file)
                patient_id = os.path.basename(root)
                meta = clean_and_verify_wave(record_path, output_dir, patient_id, record_file)
                if meta:
                    metadata.append(meta)
    with open(os.path.join(output_dir, "pleth_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    import argparse
    #parser = argparse.ArgumentParser(description="Process MIMIC-III folder tree for Pleth waves")
    #parser.add_argument("--input", required=True, help="Top-level folder (e.g., p00)")
    #parser.add_argument("--output", required=True, help="Output folder for clean PPG files")
    #args = parser.parse_args()

    #process_mimic_structure(args.input, args.output)
    input = "/home/chris/Projects/CHF/Data-Download/Downloads/CHF/p00"
    output = "results/CHF_npy"
    process_mimic_structure(input,output)