#!/bin/bash

# Go to script directory
cd /home/user/CHF/chf_real_time/

# Activate the virtual environment
source venv/bin/activate

# Create logs directory if not exists
mkdir -p logs

# Generate a timestamped log filename
timestamp=$(date +"%Y%m%d_%H%M")
log_file="logs/chf_run_$timestamp.txt"

# Run the script and log output
python live_chf_pi_gui_stream.py > "$log_file" 2>&1
