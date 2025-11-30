# Interactive Colorization GUI

A user-friendly Gradio-based interface for the UniColor colorization system.

## Quick Start

### Prerequisites

Make sure you have the unicolor conda environment activated and the model checkpoint downloaded:

```bash
conda activate unicolor
```

The GUI expects the checkpoint at:
```
./framework/checkpoints/unicolor_mscoco/mscoco_step259999.ckpt
```

### Installation

Install Gradio (if not already installed):
```bash
pip install gradio
```

### Running the GUI

```bash
cd /media/kisna/docker_d/unicolor
conda activate unicolor
python colorization_gui.py
```

The GUI will launch at `http://localhost:7860` and should open automatically in your browser.

## Features

### 🎲 Unconditional Colorization
- Upload a grayscale image
- Automatic colorization without any hints
- Adjustable diversity (top-k parameter)
- Generate multiple samples

### ✏️ Manual Color Hints
- Draw color strokes on specific regions
- Interactive color picker built into the interface
- Visual feedback showing extracted hint points
- Perfect for precise color control

### 📝 Text-Based Colorization
- Describe colors using natural language
- Examples: "green jacket", "blue sky", "red dress"
- CLIP-based semantic understanding
- Displays attention heatmaps

### 🖼️ Exemplar-Based Colorization
- Upload a reference color image
- Automatically extracts color hints from similar regions
- Shows warped exemplar and hint points
- Great for matching existing color schemes

### 🔀 Hybrid Mode
- Combine multiple input methods
- Enable/disable each mode as needed
- Merges hints from all enabled sources
- Maximum flexibility and control

## Parameters

- **Top-k (diversity)**: Controls output diversity
  - Higher values (80-100): More creative, diverse colors
  - Lower values (1-20): More conservative, realistic colors
  - Default: 100

- **Number of samples**: Generate multiple variations
  - Range: 1-5
  - Default: 1
  - Higher values take longer but show more options

## Tips for Best Results

1. **Image Quality**: Use high-quality grayscale images (256px - 1024px)

2. **Manual Hints**: 
   - Draw strokes in key regions (clothing, sky, objects)
   - Use broader strokes for better coverage
   - The system works with 16x16 grid cells

3. **Text Prompts**:
   - Be specific: "green jacket" not just "green"
   - Mention the object and its color
   - Works best with common objects

4. **Exemplar Images**:
   - Choose references with similar content/composition
   - Better matches produce better results

5. **Hybrid Mode**:
   - Combine methods for best control
   - Example: Manual hints for main objects + Text for background

## Troubleshooting

### Model Not Loading
- Check that the checkpoint file exists at the expected path
- Verify the conda environment is activated
- Check CUDA availability if using GPU

### Out of Memory
- Reduce image size (keep under 1024px)
- Use lower number of samples
- Close other GPU-intensive applications

### Slow Performance
- First run is slower due to model loading
- Subsequent colorizations are faster
- CPU mode is significantly slower than GPU

## File Structure

```
colorization_gui.py          # Main GUI application
sample/
├── colorizer.py            # Colorization model
├── utils_func.py           # Utility functions
└── ImageMatch/             # Exemplar matching
framework/
└── checkpoints/            # Model checkpoints
```

## Citation

If you use this tool, please cite the original UniColor paper:

```bibtex
@article{huang2022unicolor,
  title={UniColor: A Unified Framework for Multi-Modal Colorization with Transformer},
  author={Huang, Zhitong and Zhao, Nanxuan and Liao, Jing},
  journal={ACM Transactions on Graphics},
  year={2022}
}
```

## License

This GUI is built on top of the UniColor framework. Please refer to the original repository for licensing information.
