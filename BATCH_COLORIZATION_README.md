# Batch Keyframe Colorization GUI

A specialized GUI for colorizing batches of B&W video keyframes with object similarity-based color propagation. Designed for video colorization workflows, particularly for preparing keyframes for tools like [ColorMNet](https://github.com/yyang181/colormnet).

## Features

- **Batch Processing**: Process entire folder structures of keyframes automatically
- **Object Similarity Matching**: Uses exemplar-based colorization to propagate colors between frames
- **Color Consistency**: Each frame uses the previous colorized frame as reference for temporal coherence
- **Region Selection**: Target specific areas (faces, clothing) for focused colorization
- **Text Prompts**: Use natural language to guide colors (e.g., "red saree, blue sky")
- **Hybrid Mode**: Combine exemplar + text hints for best results
- **Resume Support**: Skip already processed files to resume interrupted jobs
- **Progress Tracking**: Real-time progress updates and processing log
- **Folder Structure Preservation**: Output maintains the same folder hierarchy as input

## Expected Folder Structure

```
root_folder/
├── video_1_key_1/
│   ├── keyframe_0000_00001.png
│   ├── keyframe_0001_00285.png
│   ├── keyframe_0002_00297.png
│   └── ...
├── video_1_key_2/
│   ├── keyframe_0000_00001.png
│   └── ...
├── video_2_key_1/
│   └── ...
└── ...
```

## Usage

### 1. Start the GUI

```bash
./run_batch_colorization.sh
```

Or directly:
```bash
python batch_colorization_gui.py
```

The GUI will launch at `http://localhost:7861`

### 2. Configure Settings

1. **Root Folder Path**: Set the path to your keyframes folder
2. **Output Folder**: Leave empty for auto-generated path (`{root}_colorized`)
3. **Initial Reference Image** (optional): Provide a color reference for the first frame
4. **Top-k**: Adjust diversity (lower = more consistent, higher = more varied)
5. **Use previous frame as reference**: Enable for color propagation (recommended)
6. **Skip existing files**: Enable to resume interrupted processing

### 2b. Region Selection & Text Hints (Optional)

Expand the "Region Selection & Text Hints" accordion for advanced options:

- **Enable region selection**: Only re-colorize specific areas
- **Regions**: Define areas using normalized coordinates (0-1)
  - Format: `x0,y0,x1,y1; x0,y0,x1,y1; ...`
  - Example face region: `0.3,0.1,0.7,0.45`
  - Example upper body: `0.2,0.2,0.8,0.6`
- **Text Prompt**: Describe colors for objects
  - Example: `red saree, golden jewelry, blue sky`
  - Uses CLIP to locate objects and apply colors

### 3. Process

1. Click "Scan Folder Structure" to preview the batches
2. Click "Start Batch Processing" to begin
3. Monitor progress in the log and gallery
4. Use "Stop" to interrupt if needed

## Workflow for ColorMNet

1. Extract keyframes from your B&W video using your preferred method
2. Organize keyframes into batch folders (as shown above)
3. Run this batch colorization GUI to colorize all keyframes
4. Use the colorized keyframes as reference for ColorMNet video propagation

## Tips

- **Initial Reference**: Providing a good color reference image for the first frame significantly improves results
- **Top-k Setting**: 
  - Use lower values (30-50) for more consistent, conservative colors
  - Use higher values (80-100) for more varied, creative colors
- **Batch Organization**: Group related scenes together in batches for better color consistency
- **Resume Processing**: If interrupted, just run again with "Skip existing files" enabled
- **Region Selection**: Use for Bollywood videos to focus on:
  - Face/skin tones: `0.3,0.1,0.7,0.45`
  - Clothing/saree: `0.2,0.4,0.8,0.9`
- **Text Prompts for Indian Cinema**:
  - `red saree, golden jewelry`
  - `blue kurta, white dhoti`
  - `brown skin, black hair`
  - `green trees, blue sky`

## Technical Details

- Uses UniColor's exemplar-based colorization with image warping
- Object similarity is computed using feature matching between frames
- Color hints are extracted from high-similarity regions and propagated
- Falls back to unconditional colorization if no good matches found

## Requirements

Same as the main UniColor project:
- PyTorch with CUDA support
- Gradio
- PIL/Pillow
- NumPy
- OpenCV

## Output

The colorized frames are saved in the output folder with the same structure:

```
root_folder_colorized/
├── video_1_key_1/
│   ├── keyframe_0000_00001.png  (colorized)
│   ├── keyframe_0001_00285.png  (colorized)
│   └── ...
├── video_1_key_2/
│   └── ...
└── ...
```
