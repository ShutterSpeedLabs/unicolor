#!/usr/bin/env python3
"""
Interactive Colorization GUI using Gradio
Supports multiple colorization modes: unconditional, manual hints, text-based, exemplar-based, and hybrid
"""

import os
import sys
import gradio as gr
import numpy as np
from PIL import Image
import torch

# Add sample directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sample'))

from colorizer import Colorizer
from utils_func import draw_strokes, color_resize, limit_size

# Global variables
COLORIZER = None
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = [256, 256]
CHECKPOINT_PATH = './framework/checkpoints/unicolor_mscoco/mscoco_step259999.ckpt'

# Color palette for manual hints (RGB values)
COLOR_PALETTE = {
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'yellow': [255, 255, 0],
    'orange': [255, 165, 0],
    'purple': [128, 0, 128],
    'pink': [255, 192, 203],
    'brown': [165, 42, 42],
    'gray': [128, 128, 128],
    'white': [255, 255, 255],
    'black': [0, 0, 0],
    'cyan': [0, 255, 255],
    'magenta': [255, 0, 255],
    'lime': [0, 128, 0],
    'navy': [0, 0, 128],
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
        return f"✅ Model loaded successfully on {DEVICE}"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"


def prepare_grayscale_image(image):
    """Convert image to grayscale if needed and resize"""
    if image is None:
        return None, "❌ No image provided"
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Limit size
        image = limit_size(image, minsize=256, maxsize=1024)
        
        # Convert to grayscale
        gray_image = image.convert('L')
        
        return gray_image, f"✅ Image prepared ({gray_image.size[0]}x{gray_image.size[1]})"
    except Exception as e:
        return None, f"❌ Error preparing image: {str(e)}"


def extract_strokes_from_annotated_image(gray_image, annotated_image, grid_size=16):
    """
    Extract color strokes from an annotated image
    Compares grayscale with annotated version to find colored regions
    """
    if annotated_image is None:
        return []
    
    try:
        # Convert to numpy arrays
        gray_np = np.array(gray_image.convert('RGB'))
        annotated_np = np.array(annotated_image)
        
        # Resize to match if needed
        if gray_np.shape != annotated_np.shape:
            annotated_image_pil = Image.fromarray(annotated_np)
            annotated_image_pil = annotated_image_pil.resize(gray_image.size)
            annotated_np = np.array(annotated_image_pil)
        
        # Find pixels that differ from grayscale (i.e., have been colored)
        diff = np.abs(gray_np.astype(float) - annotated_np.astype(float)).sum(axis=2)
        threshold = 10  # Threshold to detect colored regions
        
        strokes = []
        h, w = diff.shape
        
        # Divide image into grid cells
        for r in range(0, h // grid_size):
            for c in range(0, w // grid_size):
                y0, y1 = r * grid_size, (r + 1) * grid_size
                x0, x1 = c * grid_size, (c + 1) * grid_size
                
                # Check if this cell has been colored
                cell_diff = diff[y0:y1, x0:x1]
                if cell_diff.mean() > threshold:
                    # Extract the average color in this cell
                    color = annotated_np[y0:y1, x0:x1, :].mean(axis=(0, 1))
                    
                    # Scale coordinates to IMG_SIZE
                    index = [
                        int(r * grid_size / h * IMG_SIZE[0]),
                        int(c * grid_size / w * IMG_SIZE[1])
                    ]
                    
                    strokes.append({
                        'index': index,
                        'color': color.tolist()
                    })
        
        return strokes
    except Exception as e:
        print(f"Error extracting strokes: {e}")
        return []


def process_unconditional(image, topk, num_samples, progress=gr.Progress()):
    """Unconditional colorization - no hints"""
    if COLORIZER is None:
        return None, "❌ Model not loaded. Please wait for initialization."
    
    gray_image, msg = prepare_grayscale_image(image)
    if gray_image is None:
        return None, msg
    
    try:
        results = []
        for i in range(num_samples):
            progress((i + 1) / num_samples, desc=f"Generating sample {i+1}/{num_samples}...")
            colorized = COLORIZER.sample(gray_image, [], topk=topk)
            results.append(np.array(colorized))
        
        return results, f"✅ Generated {num_samples} sample(s)"
    except Exception as e:
        return None, f"❌ Error during colorization: {str(e)}"


def process_manual_hints(image, annotated_image, topk, num_samples, progress=gr.Progress()):
    """Colorization with manual color hints"""
    if COLORIZER is None:
        return None, None, "❌ Model not loaded. Please wait for initialization."
    
    gray_image, msg = prepare_grayscale_image(image)
    if gray_image is None:
        return None, None, msg
    
    # Extract strokes from annotated image
    strokes = extract_strokes_from_annotated_image(gray_image, annotated_image)
    
    if len(strokes) == 0:
        return None, None, "❌ No color hints detected. Please draw some color strokes on the image."
    
    try:
        # Draw strokes visualization
        strokes_img = draw_strokes(gray_image.convert('RGB'), IMG_SIZE, strokes)
        
        # Generate colorized images
        results = []
        for i in range(num_samples):
            progress((i + 1) / num_samples, desc=f"Generating sample {i+1}/{num_samples}...")
            colorized = COLORIZER.sample(gray_image, strokes, topk=topk)
            results.append(np.array(colorized))
        
        return results, np.array(strokes_img), f"✅ Generated {num_samples} sample(s) with {len(strokes)} hint points"
    except Exception as e:
        return None, None, f"❌ Error during colorization: {str(e)}"


def process_text(image, text_prompt, topk, num_samples, progress=gr.Progress()):
    """Text-based colorization using CLIP"""
    if COLORIZER is None:
        return None, None, "❌ Model not loaded. Please wait for initialization."
    
    if not text_prompt or text_prompt.strip() == "":
        return None, None, "❌ Please enter a text prompt (e.g., 'green jacket', 'blue sky')"
    
    gray_image, msg = prepare_grayscale_image(image)
    if gray_image is None:
        return None, None, msg
    
    try:
        progress(0.3, desc="Extracting hints from text prompt...")
        
        # Get strokes from CLIP
        strokes, heatmaps = COLORIZER.get_strokes_from_clip(gray_image, text_prompt)
        
        if len(strokes) == 0:
            return None, None, f"❌ Could not extract color hints from prompt: '{text_prompt}'"
        
        # Draw strokes visualization
        strokes_img = draw_strokes(gray_image.convert('RGB'), IMG_SIZE, strokes)
        
        # Generate colorized images
        results = []
        for i in range(num_samples):
            progress(0.3 + (0.7 * (i + 1) / num_samples), desc=f"Generating sample {i+1}/{num_samples}...")
            colorized = COLORIZER.sample(gray_image, strokes, topk=topk)
            results.append(np.array(colorized))
        
        # Combine heatmaps with strokes image for visualization
        viz_images = [np.array(strokes_img)]
        if heatmaps:
            viz_images.extend([np.array(h) for h in heatmaps])
        
        return results, viz_images, f"✅ Generated {num_samples} sample(s) using text: '{text_prompt}'"
    except Exception as e:
        return None, None, f"❌ Error during text-based colorization: {str(e)}"


def process_exemplar(image, exemplar_image, topk, num_samples, progress=gr.Progress()):
    """Exemplar-based colorization using reference image"""
    if COLORIZER is None:
        return None, None, "❌ Model not loaded. Please wait for initialization."
    
    if exemplar_image is None:
        return None, None, "❌ Please upload an exemplar (reference) image"
    
    gray_image, msg = prepare_grayscale_image(image)
    if gray_image is None:
        return None, None, msg
    
    try:
        # Prepare exemplar
        if isinstance(exemplar_image, np.ndarray):
            exemplar_image = Image.fromarray(exemplar_image)
        exemplar_image = limit_size(exemplar_image.convert('RGB'), minsize=256, maxsize=1024)
        
        progress(0.3, desc="Extracting colors from exemplar...")
        
        # Get strokes from exemplar
        strokes, warped = COLORIZER.get_strokes_from_exemplar(gray_image, exemplar_image)
        
        if len(strokes) == 0:
            return None, None, "❌ Could not extract color hints from exemplar image"
        
        # Resize warped to match gray image
        warped = color_resize(gray_image, warped)
        
        # Draw strokes visualization
        strokes_img = draw_strokes(gray_image.convert('RGB'), IMG_SIZE, strokes)
        
        # Generate colorized images
        results = []
        for i in range(num_samples):
            progress(0.3 + (0.7 * (i + 1) / num_samples), desc=f"Generating sample {i+1}/{num_samples}...")
            colorized = COLORIZER.sample(gray_image, strokes, topk=topk)
            results.append(np.array(colorized))
        
        # Visualization: warped exemplar and strokes
        viz_images = [np.array(warped), np.array(strokes_img)]
        
        return results, viz_images, f"✅ Generated {num_samples} sample(s) using exemplar with {len(strokes)} hint points"
    except Exception as e:
        return None, None, f"❌ Error during exemplar-based colorization: {str(e)}"


def process_hybrid(image, annotated_image, text_prompt, exemplar_image, 
                   use_manual, use_text, use_exemplar, topk, num_samples, progress=gr.Progress()):
    """Hybrid colorization combining multiple input modes"""
    if COLORIZER is None:
        return None, None, "❌ Model not loaded. Please wait for initialization."
    
    gray_image, msg = prepare_grayscale_image(image)
    if gray_image is None:
        return None, None, msg
    
    all_strokes = []
    modes_used = []
    
    try:
        # Collect strokes from manual hints
        if use_manual and annotated_image is not None:
            progress(0.1, desc="Extracting manual hints...")
            manual_strokes = extract_strokes_from_annotated_image(gray_image, annotated_image)
            all_strokes.extend(manual_strokes)
            if len(manual_strokes) > 0:
                modes_used.append(f"manual ({len(manual_strokes)} hints)")
        
        # Collect strokes from text
        if use_text and text_prompt and text_prompt.strip():
            progress(0.2, desc="Extracting hints from text...")
            text_strokes, _ = COLORIZER.get_strokes_from_clip(gray_image, text_prompt)
            all_strokes.extend(text_strokes)
            if len(text_strokes) > 0:
                modes_used.append(f"text ({len(text_strokes)} hints)")
        
        # Collect strokes from exemplar
        if use_exemplar and exemplar_image is not None:
            progress(0.3, desc="Extracting hints from exemplar...")
            if isinstance(exemplar_image, np.ndarray):
                exemplar_image = Image.fromarray(exemplar_image)
            exemplar_image = limit_size(exemplar_image.convert('RGB'), minsize=256, maxsize=1024)
            
            exemplar_strokes, _ = COLORIZER.get_strokes_from_exemplar(gray_image, exemplar_image)
            all_strokes.extend(exemplar_strokes)
            if len(exemplar_strokes) > 0:
                modes_used.append(f"exemplar ({len(exemplar_strokes)} hints)")
        
        if len(all_strokes) == 0:
            return None, None, "❌ No color hints found. Please enable at least one mode and provide inputs."
        
        # Remove duplicate strokes (same position)
        unique_strokes = []
        seen_indices = set()
        for stk in all_strokes:
            idx_tuple = tuple(stk['index'])
            if idx_tuple not in seen_indices:
                unique_strokes.append(stk)
                seen_indices.add(idx_tuple)
        
        # Draw strokes visualization
        strokes_img = draw_strokes(gray_image.convert('RGB'), IMG_SIZE, unique_strokes)
        
        # Generate colorized images
        results = []
        for i in range(num_samples):
            progress(0.3 + (0.7 * (i + 1) / num_samples), desc=f"Generating sample {i+1}/{num_samples}...")
            colorized = COLORIZER.sample(gray_image, unique_strokes, topk=topk)
            results.append(np.array(colorized))
        
        modes_str = " + ".join(modes_used)
        return results, [np.array(strokes_img)], f"✅ Generated {num_samples} sample(s) using {modes_str}"
    except Exception as e:
        return None, None, f"❌ Error during hybrid colorization: {str(e)}"


# Build Gradio interface
def build_interface():
    """Build the Gradio interface"""
    
    with gr.Blocks(title="UniColor Interactive Colorization", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 UniColor Interactive Colorization
        
        Transform grayscale images into vibrant color images using AI! Choose from multiple colorization modes:
        - **Unconditional**: Automatic colorization without hints
        - **Manual Hints**: Draw color strokes on specific regions
        - **Text-Based**: Describe colors using natural language (e.g., "green jacket")
        - **Exemplar-Based**: Use a reference image for color guidance
        - **Hybrid**: Combine multiple methods for best results
        """)
        
        # Model status
        status_text = gr.Textbox(label="Model Status", value="Loading model...", interactive=False)
        
        with gr.Tabs():
            # Tab 1: Unconditional
            with gr.Tab("🎲 Unconditional"):
                gr.Markdown("Upload a grayscale image and let the AI colorize it automatically.")
                with gr.Row():
                    with gr.Column():
                        uncond_input = gr.Image(label="Input Image", type="pil")
                        with gr.Row():
                            uncond_topk = gr.Slider(1, 100, value=100, step=1, label="Top-k (diversity)")
                            uncond_samples = gr.Slider(1, 5, value=1, step=1, label="Number of samples")
                        uncond_btn = gr.Button("🎨 Colorize", variant="primary")
                    with gr.Column():
                        uncond_output = gr.Gallery(label="Colorized Results")
                        uncond_status = gr.Textbox(label="Status", interactive=False)
                
                uncond_btn.click(
                    fn=process_unconditional,
                    inputs=[uncond_input, uncond_topk, uncond_samples],
                    outputs=[uncond_output, uncond_status]
                )
            
            # Tab 2: Manual Hints
            with gr.Tab("✏️ Manual Color Hints"):
                gr.Markdown("""
                Draw color strokes on specific regions of the image. 
                - Use the drawing tools to add color hints
                - The AI will use these hints to guide colorization
                """)
                with gr.Row():
                    with gr.Column():
                        manual_input = gr.Image(label="Input Image", type="pil")
                        manual_annotated = gr.Image(label="Draw Color Hints Here", type="pil", 
                                                   tool="color-sketch", brush_radius=20)
                        gr.Markdown("💡 **Tip**: Draw strokes in the regions you want to colorize with specific colors")
                        with gr.Row():
                            manual_topk = gr.Slider(1, 100, value=100, step=1, label="Top-k (diversity)")
                            manual_samples = gr.Slider(1, 5, value=1, step=1, label="Number of samples")
                        manual_btn = gr.Button("🎨 Colorize with Hints", variant="primary")
                    with gr.Column():
                        manual_viz = gr.Image(label="Extracted Hint Points")
                        manual_output = gr.Gallery(label="Colorized Results")
                        manual_status = gr.Textbox(label="Status", interactive=False)
                
                # Auto-populate annotated image when input is loaded
                manual_input.change(
                    fn=lambda x: x,
                    inputs=[manual_input],
                    outputs=[manual_annotated]
                )
                
                manual_btn.click(
                    fn=process_manual_hints,
                    inputs=[manual_input, manual_annotated, manual_topk, manual_samples],
                    outputs=[manual_output, manual_viz, manual_status]
                )
            
            # Tab 3: Text-Based
            with gr.Tab("📝 Text-Based"):
                gr.Markdown("""
                Describe colors using natural language. Examples:
                - "green jacket"
                - "blue sky"
                - "red dress"
                - "brown hair"
                """)
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Image(label="Input Image", type="pil")
                        text_prompt = gr.Textbox(label="Text Prompt", 
                                               placeholder="e.g., green jacket, blue sky",
                                               lines=2)
                        with gr.Row():
                            text_topk = gr.Slider(1, 100, value=100, step=1, label="Top-k (diversity)")
                            text_samples = gr.Slider(1, 5, value=1, step=1, label="Number of samples")
                        text_btn = gr.Button("🎨 Colorize with Text", variant="primary")
                    with gr.Column():
                        text_viz = gr.Gallery(label="Heatmaps & Hint Points")
                        text_output = gr.Gallery(label="Colorized Results")
                        text_status = gr.Textbox(label="Status", interactive=False)
                
                text_btn.click(
                    fn=process_text,
                    inputs=[text_input, text_prompt, text_topk, text_samples],
                    outputs=[text_output, text_viz, text_status]
                )
            
            # Tab 4: Exemplar-Based
            with gr.Tab("🖼️ Exemplar-Based"):
                gr.Markdown("Upload a reference (exemplar) color image to guide the colorization.")
                with gr.Row():
                    with gr.Column():
                        exemplar_input = gr.Image(label="Input Grayscale Image", type="pil")
                        exemplar_ref = gr.Image(label="Reference Color Image", type="pil")
                        with gr.Row():
                            exemplar_topk = gr.Slider(1, 100, value=100, step=1, label="Top-k (diversity)")
                            exemplar_samples = gr.Slider(1, 5, value=1, step=1, label="Number of samples")
                        exemplar_btn = gr.Button("🎨 Colorize with Exemplar", variant="primary")
                    with gr.Column():
                        exemplar_viz = gr.Gallery(label="Warped Exemplar & Hint Points")
                        exemplar_output = gr.Gallery(label="Colorized Results")
                        exemplar_status = gr.Textbox(label="Status", interactive=False)
                
                exemplar_btn.click(
                    fn=process_exemplar,
                    inputs=[exemplar_input, exemplar_ref, exemplar_topk, exemplar_samples],
                    outputs=[exemplar_output, exemplar_viz, exemplar_status]
                )
            
            # Tab 5: Hybrid
            with gr.Tab("🔀 Hybrid Mode"):
                gr.Markdown("Combine multiple input modes for the best colorization results!")
                with gr.Row():
                    with gr.Column():
                        hybrid_input = gr.Image(label="Input Image", type="pil")
                        
                        with gr.Accordion("Manual Hints", open=False):
                            hybrid_use_manual = gr.Checkbox(label="Enable Manual Hints", value=False)
                            hybrid_annotated = gr.Image(label="Draw Color Hints", type="pil", 
                                                       tool="color-sketch", brush_radius=20)
                        
                        with gr.Accordion("Text Prompt", open=False):
                            hybrid_use_text = gr.Checkbox(label="Enable Text Prompt", value=False)
                            hybrid_text = gr.Textbox(label="Text Prompt", 
                                                   placeholder="e.g., green jacket")
                        
                        with gr.Accordion("Exemplar Image", open=False):
                            hybrid_use_exemplar = gr.Checkbox(label="Enable Exemplar", value=False)
                            hybrid_exemplar = gr.Image(label="Reference Image", type="pil")
                        
                        with gr.Row():
                            hybrid_topk = gr.Slider(1, 100, value=100, step=1, label="Top-k (diversity)")
                            hybrid_samples = gr.Slider(1, 5, value=1, step=1, label="Number of samples")
                        hybrid_btn = gr.Button("🎨 Colorize (Hybrid)", variant="primary")
                    
                    with gr.Column():
                        hybrid_viz = gr.Gallery(label="Combined Hint Points")
                        hybrid_output = gr.Gallery(label="Colorized Results")
                        hybrid_status = gr.Textbox(label="Status", interactive=False)
                
                # Auto-populate annotated image
                hybrid_input.change(
                    fn=lambda x: x,
                    inputs=[hybrid_input],
                    outputs=[hybrid_annotated]
                )
                
                hybrid_btn.click(
                    fn=process_hybrid,
                    inputs=[
                        hybrid_input, hybrid_annotated, hybrid_text, hybrid_exemplar,
                        hybrid_use_manual, hybrid_use_text, hybrid_use_exemplar,
                        hybrid_topk, hybrid_samples
                    ],
                    outputs=[hybrid_output, hybrid_viz, hybrid_status]
                )
        
        gr.Markdown("""
        ---
        ### 📖 About
        This is an interactive interface for **UniColor**: A Unified Framework for Multi-Modal Colorization with Transformer.
        
        **Tips for best results:**
        - Use high-quality grayscale images
        - For manual hints: draw strokes in key regions (clothing, sky, objects)
        - For text prompts: be specific about color and object (e.g., "green jacket" not just "green")
        - For exemplar: choose reference images with similar content
        - Adjust top-k for diversity (higher = more diverse/creative, lower = more conservative)
        """)
        
        # Load model on startup
        demo.load(fn=load_model, outputs=[status_text])
    
    return demo


if __name__ == "__main__":
    print("Starting UniColor Interactive Colorization GUI...")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    
    demo = build_interface()
    demo.queue()  # Enable queuing for progress tracking
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
