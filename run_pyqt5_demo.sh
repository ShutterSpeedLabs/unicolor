#!/bin/bash
# Launcher script for the PyQt5-based demo GUI

echo "Starting UniColor PyQt5 Demo..."
echo "================================"

cd "$(dirname "$0")"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unicolor

# Set display for GUI
export QT_QPA_PLATFORM=xcb

# Run the demo
cd demo
python main.py
