# Quick Start Guide

## Launch the GUI

```bash
cd /media/kisna/docker_d/unicolor
conda activate unicolor
python colorization_gui.py
```

The GUI will open in your browser at: **http://localhost:7860**

## Features

### 🎲 Tab 1: Unconditional
- Upload grayscale image
- Click "Colorize"
- Get automatic colorization

### ✏️ Tab 2: Manual Color Hints
- Upload grayscale image
- Draw color strokes on specific regions
- Click "Colorize with Hints"

### 📝 Tab 3: Text-Based
- Upload grayscale image
- Enter text like "green jacket" or "blue sky"
- Click "Colorize with Text"

### 🖼️ Tab 4: Exemplar-Based
- Upload grayscale image
- Upload a color reference image
- Click "Colorize with Exemplar"

### 🔀 Tab 5: Hybrid Mode
- Combine any/all of the above methods
- Enable checkboxes for desired modes
- Click "Colorize (Hybrid)"

## Parameters

- **Top-k**: Higher = more creative (default: 100)
- **Samples**: Number of variations (1-5)

## Tips

- First run takes ~10-30 seconds to load model
- GPU recommended for faster processing
- Draw broader strokes for better manual hints
- Be specific with text prompts ("green jacket" not just "green")

For detailed documentation, see [GUI_README.md](file:///media/kisna/docker_d/unicolor/GUI_README.md)
