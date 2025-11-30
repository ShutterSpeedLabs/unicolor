#!/usr/bin/env python3
"""
Batch Keyframe Colorization GUI using Gradio
Processes folders of keyframes with object similarity-based color propagation.
Designed for video colorization workflows (e.g., feeding to ColorMNet).

Features:
- Batch processing of keyframe folders
- Object similarity-based color propagation
- Region selection for targeted colorization
- Multiple hint modes: exemplar, text, manual strokes
"""

import os
import sys
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch
from pathlib import Path
import time
from datetime import datetime
import math

# Add sample directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sample'))

from colorizer import Colorizer
from utils_func import draw_strokes, color_resize, limit_size

# Global variables
COLORIZER = None
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = [256, 256]  # Default processing size
CHECKPOINT_PATH = './framework/checkpoints/unicolor_mscoco/mscoco_step259999.ckpt'

# Native resolution support
NATIVE_RES_COLORIZER = None  # Separate colorizer for native resolution

# Processing state
PROCESSING_STATE = {
    'is_running': False,
    'should_stop': False,
    'current_batch': '',
    'current_frame': '',
    'progress': 0,
    'log': []
}

# Region selection state
REGION_STATE = {
    'regions': [],  # List of [x0, y0, x1, y1] normalized coordinates
    'hint_mode': 'exemplar',  # 'exemplar', 'text', 'stroke', 'hybrid'
    'text_prompt': '',
    'strokes': []  # Manual color strokes
}


def load_model():
    """Load the colorization model"""
    global COLORIZER
    
    if not os.path.exists(CHECKPOINT_PATH):
        return f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}"
    
    try:
        COLORIZER = Colorizer(
            CHECKPOINT_PATH, 
            DEVICE, 
            IMG_SIZE, 
            load_clip=True, 
            load_warper=True
        )
        return f"✅ Model loaded successfully on {DEVICE} (processing size: {IMG_SIZE[0]}x{IMG_SIZE[1]})"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"


def get_colorizer_for_size(width, height):
    """Get or create a colorizer for the specified image size"""
    global COLORIZER, NATIVE_RES_COLORIZER
    
    # Round to nearest multiple of 16
    proc_width = (width // 16) * 16
    proc_height = (height // 16) * 16
    
    # If matches default size, use default colorizer
    if proc_width == IMG_SIZE[1] and proc_height == IMG_SIZE[0]:
        return COLORIZER, [proc_height, proc_width]
    
    # Check if we already have a native res colorizer with matching size
    if NATIVE_RES_COLORIZER is not None:
        if NATIVE_RES_COLORIZER.img_size == [proc_height, proc_width]:
            return NATIVE_RES_COLORIZER, [proc_height, proc_width]
    
    # Create new colorizer for this size
    try:
        NATIVE_RES_COLORIZER = Colorizer(
            CHECKPOINT_PATH,
            DEVICE,
            [proc_height, proc_width],
            load_clip=True,
            load_warper=True
        )
        return NATIVE_RES_COLORIZER, [proc_height, proc_width]
    except Exception as e:
        # Fallback to default colorizer
        print(f"Failed to create native res colorizer: {e}, using default")
        return COLORIZER, IMG_SIZE


def scan_folder_structure(root_folder):
    """Scan the root folder and return batch structure info"""
    if not root_folder or not os.path.exists(root_folder):
        return "❌ Invalid folder path", None, None
    
    root_path = Path(root_folder)
    batches = []
    total_frames = 0
    
    # Get all subdirectories (batch folders)
    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    if not subdirs:
        # Check if frames are directly in root folder
        frames = get_image_files(root_path)
        if frames:
            batches.append({
                'name': root_path.name,
                'path': str(root_path),
                'frames': frames,
                'count': len(frames)
            })
            total_frames = len(frames)
    else:
        for subdir in subdirs:
            frames = get_image_files(subdir)
            if frames:
                batches.append({
                    'name': subdir.name,
                    'path': str(subdir),
                    'frames': frames,
                    'count': len(frames)
                })
                total_frames += len(frames)
    
    if not batches:
        return "❌ No image files found in folder structure", None, None
    
    # Build summary
    summary_lines = [
        f"📁 Root: {root_folder}",
        f"📊 Found {len(batches)} batch(es) with {total_frames} total frames",
        "",
        "Batch Details:"
    ]
    
    for batch in batches:
        summary_lines.append(f"  • {batch['name']}: {batch['count']} frames")
    
    summary = "\n".join(summary_lines)
    
    return summary, batches, total_frames


def get_image_files(folder_path):
    """Get sorted list of image files in a folder"""
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    folder = Path(folder_path)
    
    files = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in extensions:
            files.append(str(f))
    
    return sorted(files)


def prepare_image(image, preserve_size=False, max_size=1024):
    """
    Prepare image for colorization.
    If preserve_size=True, keeps original dimensions (must be divisible by 16).
    Otherwise limits to 256-1024 range.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    if preserve_size:
        # Keep original size but ensure divisible by 16
        w, h = image.size
        new_w = (w // 16) * 16
        new_h = (h // 16) * 16
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.LANCZOS)
        return image
    else:
        return limit_size(image, minsize=256, maxsize=max_size)


def colorize_frame_with_exemplar(gray_image, exemplar_image, topk=100, native_resolution=False):
    """Colorize a single frame using an exemplar image for color hints"""
    global COLORIZER
    
    if COLORIZER is None:
        return None, "Model not loaded"
    
    try:
        # Store original size for final resize
        if isinstance(gray_image, np.ndarray):
            gray_image = Image.fromarray(gray_image)
        original_size = gray_image.size  # (width, height)
        
        # Prepare exemplar
        if isinstance(exemplar_image, np.ndarray):
            exemplar_image = Image.fromarray(exemplar_image)
        
        # Choose colorizer based on resolution mode
        if native_resolution:
            colorizer, proc_size = get_colorizer_for_size(original_size[0], original_size[1])
            # Resize to processing size (divisible by 16)
            gray_prepared = gray_image.resize((proc_size[1], proc_size[0]), Image.LANCZOS)
            exemplar_prepared = exemplar_image.convert('RGB').resize((proc_size[1], proc_size[0]), Image.LANCZOS)
        else:
            colorizer = COLORIZER
            proc_size = IMG_SIZE
            gray_prepared = limit_size(gray_image, minsize=256, maxsize=1024)
            exemplar_prepared = limit_size(exemplar_image.convert('RGB'), minsize=256, maxsize=1024)
        
        # Get strokes from exemplar using object similarity
        strokes, warped = colorizer.get_strokes_from_exemplar(gray_prepared, exemplar_prepared)
        
        if len(strokes) == 0:
            # Fallback to unconditional if no strokes found
            colorized = colorizer.sample(gray_prepared, [], topk=topk)
        else:
            colorized = colorizer.sample(gray_prepared, strokes, topk=topk)
        
        # Resize back to original size
        if colorized.size != original_size:
            colorized = color_resize(gray_image, colorized)
        
        size_info = f"{proc_size[1]}x{proc_size[0]}" if native_resolution else "256x256"
        return colorized, f"OK ({len(strokes)} hints, {size_info})"
    except Exception as e:
        return None, str(e)


def colorize_frame_unconditional(gray_image, topk=100, native_resolution=False):
    """Colorize a single frame without any hints"""
    global COLORIZER
    
    if COLORIZER is None:
        return None, "Model not loaded"
    
    try:
        if isinstance(gray_image, np.ndarray):
            gray_image = Image.fromarray(gray_image)
        original_size = gray_image.size  # (width, height)
        
        # Choose colorizer based on resolution mode
        if native_resolution:
            colorizer, proc_size = get_colorizer_for_size(original_size[0], original_size[1])
            gray_prepared = gray_image.resize((proc_size[1], proc_size[0]), Image.LANCZOS)
        else:
            colorizer = COLORIZER
            proc_size = IMG_SIZE
            gray_prepared = limit_size(gray_image, minsize=256, maxsize=1024)
        
        colorized = colorizer.sample(gray_prepared, [], topk=topk)
        
        # Resize back to original size
        if colorized.size != original_size:
            colorized = color_resize(gray_image, colorized)
        
        size_info = f"{proc_size[1]}x{proc_size[0]}" if native_resolution else "256x256"
        return colorized, f"OK (unconditional, {size_info})"
    except Exception as e:
        return None, str(e)


def get_region_sample_indices(regions, img_size=[256, 256]):
    """Convert normalized region coordinates to sample indices for selective colorization"""
    indices = []
    sample_size = [img_size[0] // 16, img_size[1] // 16]  # 16x16 grid
    
    for region in regions:
        x0, y0, x1, y1 = region
        # Convert normalized coords to grid indices
        gx0 = int(x0 * sample_size[1])
        gy0 = int(y0 * sample_size[0])
        gx1 = int(math.ceil(x1 * sample_size[1]))
        gy1 = int(math.ceil(y1 * sample_size[0]))
        
        # Clamp to valid range
        gx0 = max(0, min(gx0, sample_size[1] - 1))
        gy0 = max(0, min(gy0, sample_size[0] - 1))
        gx1 = max(0, min(gx1, sample_size[1]))
        gy1 = max(0, min(gy1, sample_size[0]))
        
        for y in range(gy0, gy1):
            for x in range(gx0, gx1):
                if [y, x] not in indices:
                    indices.append([y, x])
    
    return sorted(indices)


def colorize_frame_with_regions(gray_image, exemplar_image, regions, text_prompt="", topk=100, native_resolution=False):
    """
    Colorize specific regions of a frame using exemplar/text hints.
    Regions outside selection keep colors from exemplar warping.
    """
    global COLORIZER
    
    if COLORIZER is None:
        return None, "Model not loaded"
    
    try:
        # Prepare images
        if isinstance(gray_image, np.ndarray):
            gray_image = Image.fromarray(gray_image)
        original_size = gray_image.size  # (width, height)
        
        if isinstance(exemplar_image, np.ndarray):
            exemplar_image = Image.fromarray(exemplar_image)
        
        # Choose colorizer based on resolution mode
        if native_resolution:
            colorizer, proc_size = get_colorizer_for_size(original_size[0], original_size[1])
            gray_prepared = gray_image.resize((proc_size[1], proc_size[0]), Image.LANCZOS)
            exemplar_prepared = exemplar_image.convert('RGB').resize((proc_size[1], proc_size[0]), Image.LANCZOS)
        else:
            colorizer = COLORIZER
            proc_size = IMG_SIZE
            gray_prepared = limit_size(gray_image, minsize=256, maxsize=1024)
            exemplar_prepared = limit_size(exemplar_image.convert('RGB'), minsize=256, maxsize=1024)
        
        # Collect all strokes
        all_strokes = []
        
        # Get strokes from exemplar
        exemplar_strokes, warped = colorizer.get_strokes_from_exemplar(gray_prepared, exemplar_prepared)
        all_strokes.extend(exemplar_strokes)
        
        # Get strokes from text if provided
        if text_prompt and text_prompt.strip():
            try:
                text_strokes, _ = colorizer.get_strokes_from_clip(gray_prepared, text_prompt)
                all_strokes.extend(text_strokes)
            except:
                pass
        
        # Remove duplicates
        unique_strokes = []
        seen = set()
        for stk in all_strokes:
            key = tuple(stk['index'])
            if key not in seen:
                unique_strokes.append(stk)
                seen.add(key)
        
        size_info = f"{proc_size[1]}x{proc_size[0]}" if native_resolution else "256x256"
        
        if not regions:
            # No region selection - colorize entire frame
            colorized = colorizer.sample(gray_prepared, unique_strokes, topk=topk)
            if colorized.size != original_size:
                colorized = color_resize(gray_image, colorized)
            return colorized, f"OK ({len(unique_strokes)} hints, full frame, {size_info})"
        
        # Region-based colorization
        sample_indices = get_region_sample_indices(regions, proc_size)
        
        if not sample_indices:
            # Fallback to full frame
            colorized = colorizer.sample(gray_prepared, unique_strokes, topk=topk)
            if colorized.size != original_size:
                colorized = color_resize(gray_image, colorized)
            return colorized, f"OK ({len(unique_strokes)} hints, full frame - no valid regions, {size_info})"
        
        # Use exemplar as prior, re-colorize only selected regions
        colorized = colorizer.sample(
            gray_prepared, 
            unique_strokes, 
            topk=topk,
            prior_image=exemplar_prepared,
            mask_indices=sample_indices,
            sample_indices=sample_indices
        )
        
        if colorized.size != original_size:
            colorized = color_resize(gray_image, colorized)
        
        return colorized, f"OK ({len(unique_strokes)} hints, {len(sample_indices)} region cells, {size_info})"
        
    except Exception as e:
        return None, str(e)


def draw_regions_on_image(image, regions):
    """Draw region rectangles on an image for visualization"""
    if image is None or not regions:
        return image
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    img_copy = image.copy().convert('RGB')
    draw = ImageDraw.Draw(img_copy, 'RGBA')
    
    w, h = img_copy.size
    
    for i, region in enumerate(regions):
        x0, y0, x1, y1 = region
        # Convert normalized to pixel coords
        px0, py0 = int(x0 * w), int(y0 * h)
        px1, py1 = int(x1 * w), int(y1 * h)
        
        # Draw semi-transparent rectangle
        draw.rectangle([px0, py0, px1, py1], outline='red', width=3)
        draw.rectangle([px0, py0, px1, py1], fill=(255, 0, 0, 50))
        
        # Label
        draw.text((px0 + 5, py0 + 5), f"R{i+1}", fill='white')
    
    return img_copy


def parse_regions_from_boxes(box_data, img_width, img_height):
    """Parse region boxes from Gradio image annotation format"""
    regions = []
    
    if not box_data:
        return regions
    
    # Handle different Gradio annotation formats
    if isinstance(box_data, dict) and 'boxes' in box_data:
        boxes = box_data['boxes']
    elif isinstance(box_data, list):
        boxes = box_data
    else:
        return regions
    
    for box in boxes:
        if isinstance(box, dict):
            # Format: {'xmin': x0, 'ymin': y0, 'xmax': x1, 'ymax': y1}
            x0 = box.get('xmin', 0) / img_width
            y0 = box.get('ymin', 0) / img_height
            x1 = box.get('xmax', img_width) / img_width
            y1 = box.get('ymax', img_height) / img_height
        elif isinstance(box, (list, tuple)) and len(box) >= 4:
            # Format: [x0, y0, x1, y1]
            x0, y0, x1, y1 = box[:4]
            x0, y0 = x0 / img_width, y0 / img_height
            x1, y1 = x1 / img_width, y1 / img_height
        else:
            continue
        
        # Normalize and validate
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(1, x1), min(1, y1)
        
        if x1 > x0 and y1 > y0:
            regions.append([x0, y0, x1, y1])
    
    return regions


def process_batch_folder(
    root_folder, 
    output_folder, 
    reference_image,
    topk,
    use_previous_as_ref,
    skip_existing,
    use_regions,
    regions_str,
    text_prompt,
    native_resolution,
    progress=gr.Progress()
):
    """Process all batches in the root folder with optional region selection"""
    global PROCESSING_STATE
    
    if COLORIZER is None:
        yield "❌ Model not loaded", None, "Model not loaded"
        return
    
    if not root_folder or not os.path.exists(root_folder):
        yield "❌ Invalid root folder", None, "Invalid folder"
        return
    
    # Scan folder structure
    summary, batches, total_frames = scan_folder_structure(root_folder)
    if batches is None:
        yield summary, None, "No batches found"
        return
    
    # Parse regions if enabled
    regions = []
    if use_regions and regions_str:
        try:
            # Parse regions from string format: "x0,y0,x1,y1;x0,y0,x1,y1;..."
            for region_str in regions_str.strip().split(';'):
                if region_str.strip():
                    coords = [float(x.strip()) for x in region_str.split(',')]
                    if len(coords) == 4:
                        regions.append(coords)
        except:
            pass
    
    # Setup output folder
    if not output_folder:
        output_folder = str(Path(root_folder).parent / (Path(root_folder).name + "_colorized"))
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize state
    PROCESSING_STATE['is_running'] = True
    PROCESSING_STATE['should_stop'] = False
    PROCESSING_STATE['log'] = []
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    gallery_images = []
    
    # Load initial reference image if provided
    current_reference = None
    if reference_image is not None:
        if isinstance(reference_image, np.ndarray):
            current_reference = Image.fromarray(reference_image)
        else:
            current_reference = reference_image
        log_msg = f"Using provided reference image as initial exemplar"
        PROCESSING_STATE['log'].append(log_msg)
    
    if native_resolution:
        PROCESSING_STATE['log'].append(f"Native resolution mode enabled (no downscaling)")
    if regions:
        PROCESSING_STATE['log'].append(f"Region selection enabled: {len(regions)} region(s)")
    if text_prompt:
        PROCESSING_STATE['log'].append(f"Text prompt: {text_prompt}")
    
    try:
        for batch_idx, batch in enumerate(batches):
            if PROCESSING_STATE['should_stop']:
                break
            
            batch_name = batch['name']
            batch_path = batch['path']
            frames = batch['frames']
            
            PROCESSING_STATE['current_batch'] = batch_name
            log_msg = f"\n📁 Processing batch: {batch_name} ({len(frames)} frames)"
            PROCESSING_STATE['log'].append(log_msg)
            
            # Create output subfolder
            if len(batches) > 1 or batch_path != root_folder:
                batch_output = os.path.join(output_folder, batch_name)
            else:
                batch_output = output_folder
            os.makedirs(batch_output, exist_ok=True)
            
            # Reset reference for new batch if not using previous
            batch_reference = current_reference
            
            for frame_idx, frame_path in enumerate(frames):
                if PROCESSING_STATE['should_stop']:
                    break
                
                frame_name = os.path.basename(frame_path)
                output_path = os.path.join(batch_output, frame_name)
                
                PROCESSING_STATE['current_frame'] = frame_name
                overall_progress = (processed_count + skipped_count + error_count) / total_frames
                progress(overall_progress, desc=f"Batch {batch_idx+1}/{len(batches)}: {frame_name}")
                
                # Skip if exists
                if skip_existing and os.path.exists(output_path):
                    skipped_count += 1
                    PROCESSING_STATE['log'].append(f"  ⏭️ Skipped (exists): {frame_name}")
                    continue
                
                try:
                    # Load grayscale frame
                    gray_image = Image.open(frame_path)
                    
                    # Colorize based on mode
                    if batch_reference is not None:
                        if use_regions and regions:
                            # Region-based colorization with exemplar + optional text
                            colorized, status = colorize_frame_with_regions(
                                gray_image, batch_reference, regions, 
                                text_prompt=text_prompt, topk=topk,
                                native_resolution=native_resolution
                            )
                        else:
                            # Standard exemplar-based colorization
                            colorized, status = colorize_frame_with_exemplar(
                                gray_image, batch_reference, topk=topk,
                                native_resolution=native_resolution
                            )
                    else:
                        colorized, status = colorize_frame_unconditional(
                            gray_image, topk=topk, 
                            native_resolution=native_resolution
                        )
                    
                    if colorized is None:
                        error_count += 1
                        PROCESSING_STATE['log'].append(f"  ❌ Error: {frame_name} - {status}")
                        continue
                    
                    # Save colorized frame
                    colorized.save(output_path)
                    processed_count += 1
                    
                    # Update reference for next frame if enabled
                    if use_previous_as_ref:
                        batch_reference = colorized
                        # Also update global reference for next batch
                        current_reference = colorized
                    
                    PROCESSING_STATE['log'].append(f"  ✅ {frame_name} - {status}")
                    
                    # Add to gallery (limit to last 10)
                    gallery_images.append(np.array(colorized))
                    if len(gallery_images) > 10:
                        gallery_images.pop(0)
                    
                except Exception as e:
                    error_count += 1
                    PROCESSING_STATE['log'].append(f"  ❌ Error: {frame_name} - {str(e)}")
                
                # Yield progress update
                status_text = f"Processed: {processed_count} | Skipped: {skipped_count} | Errors: {error_count}"
                log_text = "\n".join(PROCESSING_STATE['log'][-50:])  # Last 50 lines
                yield status_text, gallery_images.copy() if gallery_images else None, log_text
        
        # Final summary
        PROCESSING_STATE['is_running'] = False
        final_status = f"✅ Complete! Processed: {processed_count} | Skipped: {skipped_count} | Errors: {error_count}"
        PROCESSING_STATE['log'].append(f"\n{final_status}")
        PROCESSING_STATE['log'].append(f"Output saved to: {output_folder}")
        
        log_text = "\n".join(PROCESSING_STATE['log'])
        yield final_status, gallery_images.copy() if gallery_images else None, log_text
        
    except Exception as e:
        PROCESSING_STATE['is_running'] = False
        error_msg = f"❌ Processing failed: {str(e)}"
        PROCESSING_STATE['log'].append(error_msg)
        log_text = "\n".join(PROCESSING_STATE['log'])
        yield error_msg, gallery_images.copy() if gallery_images else None, log_text


def stop_processing():
    """Signal to stop processing"""
    global PROCESSING_STATE
    PROCESSING_STATE['should_stop'] = True
    return "⏹️ Stop requested... finishing current frame"


def preview_first_frames(root_folder):
    """Preview first frame from each batch"""
    if not root_folder or not os.path.exists(root_folder):
        return None, "Invalid folder"
    
    summary, batches, _ = scan_folder_structure(root_folder)
    if batches is None:
        return None, summary
    
    preview_images = []
    for batch in batches[:6]:  # Limit to 6 previews
        if batch['frames']:
            try:
                img = Image.open(batch['frames'][0])
                preview_images.append(np.array(img))
            except:
                pass
    
    return preview_images if preview_images else None, summary


def build_interface():
    """Build the Gradio interface for batch processing"""
    
    with gr.Blocks(title="Batch Keyframe Colorization", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 Batch Keyframe Colorization for Video
        
        Process folders of B&W keyframes with object similarity-based color propagation.
        Perfect for preparing keyframes for video colorization tools like ColorMNet.
        
        **Workflow:**
        1. Set your root folder containing batch subfolders (e.g., `video_1_key_1`, `video_1_key_2`, ...)
        2. Optionally provide a reference color image for the first frame
        3. Each subsequent frame uses the previous colorized frame as reference
        4. Optionally define regions and text prompts for targeted colorization
        5. Output maintains the same folder structure
        """)
        
        # Model status
        status_text = gr.Textbox(label="Model Status", value="Loading model...", interactive=False)
        
        with gr.Row():
            # Left column - Settings
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input/Output Settings")
                
                root_folder = gr.Textbox(
                    label="Root Folder Path",
                    placeholder="/path/to/keyframes_folder",
                    value="/media/kisna/bkp_data/DeOldify/vid_data_colorize/JaneTeriNazronN/JaneTeriNazronN_key",
                    lines=1
                )
                
                output_folder = gr.Textbox(
                    label="Output Folder (leave empty for auto)",
                    placeholder="Auto: {root_folder}_colorized",
                    lines=1
                )
                
                scan_btn = gr.Button("🔍 Scan Folder Structure", variant="secondary")
                folder_info = gr.Textbox(label="Folder Structure", lines=6, interactive=False)
                
                gr.Markdown("### 🎨 Colorization Settings")
                
                reference_image = gr.Image(
                    label="Initial Reference Image (optional)",
                    type="pil",
                    height=180
                )
                gr.Markdown("*Provide a color reference for the first frame.*")
                
                topk = gr.Slider(
                    minimum=1, maximum=100, value=100, step=1,
                    label="Top-k (diversity)",
                    info="Higher = more diverse, Lower = more conservative"
                )
                
                use_previous_as_ref = gr.Checkbox(
                    label="Use previous frame as reference",
                    value=True,
                    info="Propagate colors through frames"
                )
                
                skip_existing = gr.Checkbox(
                    label="Skip existing output files",
                    value=True,
                    info="Resume interrupted processing"
                )
                
                native_resolution = gr.Checkbox(
                    label="Native resolution (no downscaling)",
                    value=False,
                    info="Process at original size (e.g., 640x368). Uses more VRAM but better quality."
                )
                
                # Region & Text Settings
                with gr.Accordion("🎯 Region Selection & Text Hints", open=False):
                    gr.Markdown("""
                    Define specific regions to colorize and/or add text prompts for color guidance.
                    Regions use normalized coordinates (0-1).
                    """)
                    
                    use_regions = gr.Checkbox(
                        label="Enable region selection",
                        value=False,
                        info="Only re-colorize specific regions"
                    )
                    
                    regions_str = gr.Textbox(
                        label="Regions (normalized: x0,y0,x1,y1; ...)",
                        placeholder="0.2,0.1,0.8,0.5; 0.3,0.6,0.7,0.9",
                        lines=2,
                        info="Semicolon-separated regions. E.g., face: 0.3,0.1,0.7,0.4"
                    )
                    
                    text_prompt = gr.Textbox(
                        label="Text Prompt (optional)",
                        placeholder="e.g., red saree, blue sky, green trees",
                        lines=2,
                        info="Describe colors for objects (uses CLIP)"
                    )
                    
                    gr.Markdown("""
                    **Region Examples:**
                    - Face area: `0.3,0.1,0.7,0.45`
                    - Upper body: `0.2,0.2,0.8,0.6`
                    - Full frame: `0,0,1,1`
                    
                    **Text Prompt Examples:**
                    - `red saree, golden jewelry`
                    - `blue sky, green trees`
                    - `brown skin, black hair`
                    """)
            
            # Right column - Processing & Results
            with gr.Column(scale=2):
                gr.Markdown("### 🚀 Processing")
                
                with gr.Row():
                    start_btn = gr.Button("▶️ Start Batch Processing", variant="primary", scale=3)
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)
                
                processing_status = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("### 📸 Recent Results")
                result_gallery = gr.Gallery(
                    label="Recently Colorized Frames",
                    columns=5,
                    rows=2,
                    height=300
                )
                
                gr.Markdown("### 📋 Processing Log")
                log_output = gr.Textbox(
                    label="Log",
                    lines=12,
                    max_lines=25,
                    interactive=False,
                    autoscroll=True
                )
        
        # Preview section
        with gr.Accordion("🖼️ Preview First Frames", open=False):
            preview_gallery = gr.Gallery(
                label="First frame from each batch",
                columns=3,
                rows=2,
                height=250
            )
        
        gr.Markdown("""
        ---
        ### 📖 Usage Tips
        
        **Folder Structure Expected:**
        ```
        root_folder/
        ├── video_1_key_1/
        │   ├── keyframe_0000_00001.png
        │   └── ...
        ├── video_1_key_2/
        │   └── ...
        └── ...
        ```
        
        **For ColorMNet Integration:**
        - Output folder structure matches input
        - Use these colorized keyframes as reference for ColorMNet video propagation
        
        **Region Selection:**
        - Use regions to focus colorization on specific areas (faces, clothing)
        - Combine with text prompts for better color control
        - Format: `x0,y0,x1,y1` where values are 0-1 (normalized)
        """)
        
        # Event handlers
        scan_btn.click(
            fn=preview_first_frames,
            inputs=[root_folder],
            outputs=[preview_gallery, folder_info]
        )
        
        start_btn.click(
            fn=process_batch_folder,
            inputs=[
                root_folder,
                output_folder,
                reference_image,
                topk,
                use_previous_as_ref,
                skip_existing,
                use_regions,
                regions_str,
                text_prompt,
                native_resolution
            ],
            outputs=[processing_status, result_gallery, log_output]
        )
        
        stop_btn.click(
            fn=stop_processing,
            outputs=[processing_status]
        )
        
        # Load model on startup
        demo.load(fn=load_model, outputs=[status_text])
    
    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("Batch Keyframe Colorization GUI")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print()
    print("Starting Gradio interface...")
    
    demo = build_interface()
    demo.queue()  # Enable queuing for progress tracking
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from main GUI
        share=False,
        show_error=True
    )
