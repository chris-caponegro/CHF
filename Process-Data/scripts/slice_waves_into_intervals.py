import os
import numpy as np
import argparse

def slice_waveform(wave, fs, window_sec, step_sec):
    """
    Slices a waveform into overlapping windows.
    """
    window_len = int(window_sec * fs)
    step_len = int(step_sec * fs)
    slices = []
    for start in range(0, len(wave) - window_len + 1, step_len):
        end = start + window_len
        segment = wave[start:end]
        slices.append(segment)
    return slices

def process_directory(input_dir, output_dir, fs, window_sec, step_sec):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith(".npy"):
            wave_path = os.path.join(input_dir, file)
            wave = np.load(wave_path)

            slices = slice_waveform(wave, fs, window_sec, step_sec)

            base_name = os.path.splitext(file)[0]
            for idx, segment in enumerate(slices):
                slice_name = f"{base_name}_slice{idx}.npy"
                np.save(os.path.join(output_dir, slice_name), segment)

            print(f"✅ Sliced {file} into {len(slices)} segments")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice .npy pleth waves into overlapping windows")
    parser.add_argument("--input", required=True, help="Input directory with .npy files")
    parser.add_argument("--output", required=True, help="Output directory for sliced files")
    parser.add_argument("--fs", type=int, default=125, help="Sampling frequency (default: 125 Hz)")
    parser.add_argument("--window_sec", type=int, default=10, help="Window length in seconds (default: 10)")
    parser.add_argument("--step_sec", type=int, default=5, help="Step size in seconds (default: 5)")
    args = parser.parse_args()

    process_directory(args.input, args.output, args.fs, args.window_sec, args.step_sec)
