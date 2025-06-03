import matplotlib.pyplot as plt
import pandas as pd
import os

# === CONFIG ===
csv_file = 'Downloads/PLETH_CSV/p000107/3746356_0005_pleth.csv'  # Update path as needed
save_plot = True                    # Set to True to save as PNG
output_file = csv_file.replace('.csv', '_plot.png')

# === Load PLETH signal ===
data = pd.read_csv(csv_file, header=None)
pleth = data[0].values

# === Plot ===
plt.figure(figsize=(10, 4))
plt.plot(pleth, linewidth=1)
plt.title(f"PPG Waveform - {os.path.basename(csv_file)}")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()

if save_plot:
    plt.savefig(output_file)
    print(f"✅ Plot saved to {output_file}")
else:
    plt.show()
