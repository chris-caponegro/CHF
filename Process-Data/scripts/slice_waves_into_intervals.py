import os
import numpy as np
import argparse
from scipy.signal import find_peaks, welch
from scipy.stats import entropy

def verify_ppg_waveform(wave, fs, min_peaks=3, min_std=0.01, max_entropy=3.5,
                        freq_range=(0.5, 2.5), verbose=False):
    if len(wave) < fs * 5:
        if verbose: print("❌ Too short")
        return False, 0.0

    std = np.std(wave)
    if std < min_std:
        if verbose: print("❌ Too flat")
        return False, 0.0

    try:
        peaks, _ = find_peaks(wave, distance=int(0.4 * fs))
    except:
        if verbose: print("❌ Peak detection error")
        return False, 0.0

    if len(peaks) < min_peaks:
        if verbose: print(f"❌ Too few peaks: {len(peaks)}")
        return False, 0.2

    f, Pxx = welch(wave, fs=fs, nperseg=min(256, len(wave)))
    Pxx_norm = Pxx / np.sum(Pxx)
    spec_entropy = entropy(Pxx_norm)

    if spec_entropy > max_entropy:
        if verbose: print(f"❌ Noisy (entropy = {spec_entropy:.2f})")
        return False, 0.3

    dominant_freq = f[np.argmax(Pxx)]
    if not (freq_range[0] <= dominant_freq <= freq_range[1]):
        if verbose: print(f"❌ Abnormal HR frequency = {dominant_freq:.2f} Hz")
        return False, 0.4

    return True, 1.0

def slice_waveform(wave, fs, window_sec, step_sec):
    window_len = int(window_sec * fs)
    step_len = int(step_sec * fs)
    return [wave[i:i+window_len] for i in range(0, len(wave) - window_len + 1, step_len)]

def process_directory(input_dir, output_dir, fs, window_sec, step_sec, verbose=False):
    os.makedirs(output_dir, exist_ok=True)
    accepted, rejected = 0, 0

    for file in os.listdir(input_dir):
        if file.endswith(".npy"):
            wave_path = os.path.join(input_dir, file)
            wave = np.load(wave_path)

            slices = slice_waveform(wave, fs, window_sec, step_sec)

            base_name = os.path.splitext(file)[0]
            for idx, segment in enumerate(slices):
                is_valid, score = verify_ppg_waveform(segment, fs, verbose=verbose)
                if is_valid:
                    slice_name = f"{base_name}_slice{idx}.npy"
                    np.save(os.path.join(output_dir, slice_name), segment)
                    accepted += 1
                else:
                    rejected += 1

            print(f"🧾 Processed {file}: {accepted} accepted, {rejected} rejected slices")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice and quality-check PPG .npy waves")
    parser.add_argument("--input", required=True, help="Input folder with .npy waveforms")
    parser.add_argument("--output", required=True, help="Output folder for valid slices")
    parser.add_argument("--fs", type=int, default=125, help="Sampling frequency (Hz)")
    parser.add_argument("--window_sec", type=int, default=10, help="Window size in seconds")
    parser.add_argument("--step_sec", type=int, default=5, help="Step size in seconds (overlap = window - step)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs for each slice")
    args = parser.parse_args()

    process_directory(args.input, args.output, args.fs, args.window_sec, args.step_sec, args.verbose)

    #python3 script/slice_waves_into_intervals.py --input ./results/CHF_npy --output ./results/sliced_valid_pleth --fs 125 --window_sec 10 --step_sec 5 --verbose
    #python3 script/slice_waves_into_intervals.py --input ./results/CHF_npy --output ./results/sliced_valid_pleth --fs 125 --window_sec 30 --step_sec 10 --verbose
