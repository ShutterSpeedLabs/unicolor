# PyQt5 Demo - Linux Version

The original PyQt5-based demo GUI has been adapted to run on Linux.

## What Was Changed

### Removed Windows Dependencies
- **win32clipboard** → Replaced with Qt's built-in clipboard (QGuiApplication.clipboard())
- **Windows paths** → Updated to use os.path.join() for cross-platform compatibility
- **Hardcoded paths** → Changed to use relative paths from project directory

### Updated Features
- Auto-detects CUDA availability (falls back to CPU if not available)
- Linux-compatible clipboard copy functionality
- Uses sample/images as default directory for loading images
- Automatically looks for checkpoint at: `framework/checkpoints/unicolor_mscoco/mscoco_step259999.ckpt`

## Requirements

All requirements are now installed in the `unicolor` conda environment:
- PyQt5 (5.15.10)
- qimage2ndarray (1.10.0)
- All other dependencies from environment.yaml

## Running the Demo

### Option 1: Using the launcher script
```bash
cd /media/kisna/docker_d/unicolor
./run_pyqt5_demo.sh
```

### Option 2: Manual launch
```bash
cd /media/kisna/docker_d/unicolor
conda activate unicolor
cd demo
python main.py
```

## Features

The PyQt5 demo provides a desktop application with:

- **Image Loading**: Load grayscale or color images
- **Manual Strokes**: Draw color strokes directly on the canvas
- **Color Picker**: Interactive color selection
- **Text Prompts**: Enter text descriptions for colorization
- **Exemplar Loading**: Use reference images for color matching
- **Result Gallery**: View multiple colorization results
- **Copy to Clipboard**: Copy results (uses Qt clipboard on Linux)
- **Save Results**: Automatically saves to demo/results/

## Key Shortcuts

- **Left Click + Drag**: Draw color strokes
- **Right Click + Drag**: Select region for recolorization (on output)
- **Color Picker**: Click color buttons on the left panel

## Differences from Windows Version

1. **Clipboard**: Uses Qt's native clipboard instead of win32clipboard
   - If clipboard copy fails, image is saved to `/tmp/unicolor_output.png`
   
2. **File Paths**: Uses forward slashes and os.path.join()

3. **Model Loading**: More flexible checkpoint path detection

## Troubleshooting

### GUI doesn't launch
- Make sure you're in a graphical environment (not SSH without X11 forwarding)
- Check that DISPLAY is set: `echo $DISPLAY`
- Try: `export QT_QPA_PLATFORM=xcb`

### Model not loading
- Verify checkpoint exists at: `framework/checkpoints/unicolor_mscoco/mscoco_step259999.ckpt`
- You can load it manually via the GUI interface

### Clipboard copy doesn't work
- Check terminal output - image will be saved to `/tmp/` as fallback
- Install xclip for better clipboard support: `sudo apt install xclip`

## Architecture

- **main.py**: Main application window (now Linux-compatible)
- **stroke.py**: Stroke and region selection logic
- **thread.py**: Background threads for model loading and sampling
- **ui/**: Qt Designer UI files
- **colorpicker/**: Custom color picker widget

## Comparison: PyQt5 vs Gradio

| Feature | PyQt5 Demo | Gradio GUI |
|---------|-----------|------------|
| Type | Desktop App | Web App |
| Launch | Native window | Browser-based |
| Drawing | Direct canvas | Color sketch tool |
| Performance | Faster (native) | Slightly slower |
| Deployment | Local only | Can share via link |
| UI Updates | Immediate | Auto-refresh |
| Best For | Power users | Quick testing |

Choose the GUI that fits your workflow!
