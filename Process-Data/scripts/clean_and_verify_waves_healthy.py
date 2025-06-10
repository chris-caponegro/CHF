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

def clean_and_verify_wave(record_path, output_dir, record_file):
    try:
        record = wfdb.rdrecord(record_path)
        fs = record.fs
        signals = record.p_signal

        # Assume first (and only) channel is PPG
        ppg_wave = signals[:, 0]

        if is_valid_waveform(ppg_wave, fs):
            print(f"✅ Valid PPG: {record_file}")

            out_fname = f"{record_file}_ppg.npy"
            np.save(os.path.join(output_dir, out_fname), ppg_wave)

            return {
                "record": record_file,
                "fs": fs,
                "length": len(ppg_wave),
                "duration_sec": len(ppg_wave) / fs
            }
        else:
            print(f"⚠️ Invalid PPG: {record_file}")
            return None

    except Exception as e:
        print(f"❌ Error in {record_path}: {e}")
        return None

def process_openoxy_dataset(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    for file in os.listdir(base_dir):
        if file.endswith("_ppg.hea"):
            record_file = os.path.splitext(file)[0]
            record_path = os.path.join(base_dir, record_file)
            meta = clean_and_verify_wave(record_path, output_dir, record_file)
            if meta:
                metadata.append(meta)

    with open(os.path.join(output_dir, "ppg_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    input = "/home/chris/Projects/CHF/Data-Download/Downloads/physionet.org/files/openox-repo/1.1.1/waveforms/f"
    output = "results/healthy_npy"
    process_openoxy_dataset(input, output)