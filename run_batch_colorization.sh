#!/bin/bash
# Run the Batch Keyframe Colorization GUI

echo "=========================================="
echo "Batch Keyframe Colorization GUI"
echo "=========================================="
echo ""
echo "This GUI processes folders of B&W keyframes"
echo "with object similarity-based color propagation."
echo ""
echo "Perfect for preparing keyframes for ColorMNet!"
echo ""

# Activate conda environment
if command -v conda &> /dev/null; then
    echo "Activating unicolor conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate unicolor
fi

# Run the GUI
python batch_colorization_gui.py
