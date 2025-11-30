import os
from PIL import Image
import numpy as np
import torch
import nltk

from colorizer import Colorizer
from utils_func import *

# Fix device
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
print(f"Using device: {device}")

# Mock display function
def display(img):
    if isinstance(img, Image.Image):
        # Create output directory if not exists
        output_dir = '/media/kisna/docker_d/unicolor/sample/output'
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
            except OSError as e:
                print(f"Failed to create directory {output_dir}: {e}")
                return
        
        filename = f"{output_dir}/output_{len([f for f in os.listdir(output_dir) if f.startswith('output_')])}.png"
        try:
            img.save(filename)
            print(f"Saved image to {filename}")
            print(f"File exists check: {os.path.exists(filename)}")
        except Exception as e:
            print(f"Failed to save image: {e}")
    else:
        print(f"Display called with {type(img)}")

print(f"Current working directory: {os.getcwd()}")
print(f"User: {os.getlogin() if hasattr(os, 'getlogin') else 'unknown'}")


ckpt_file = '/media/kisna/docker_d/unicolor/framework/checkpoints/unicolor_mscoco/mscoco_step259999.ckpt'

# Load CLIP and ImageWarper
print("Loading model...")
try:
    colorizer = Colorizer(ckpt_file, device, [256, 256], load_clip=True, load_warper=True)
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Default directory is in /framework/
print("Processing grayscale image...")
try:
    I_gray = Image.open('../sample/images/1.jpg').convert('L')
    display(I_gray)
except Exception as e:
    print(f"Error loading image: {e}")
    # Try local path if relative fails
    try:
        I_gray = Image.open('images/1.jpg').convert('L')
        display(I_gray)
    except Exception as e2:
        print(f"Error loading image from local path: {e2}")
        raise e2

print("Unconditional sampling...")
try:
    I_uncond = colorizer.sample(I_gray, [], topk=100)
    display(I_uncond)
except Exception as e:
    print(f"Error in unconditional sampling: {e}")
    raise e

# Hint points
print("Hint points sampling...")
try:
    points = [
        {'index': [6*16, 2*16], 'color': [171, 209, 247]}
    ]
    point_img = draw_strokes(I_gray, [256, 256], points)
    I_stk = colorizer.sample(I_gray, points, topk=100)
    display(Image.fromarray( np.concatenate([np.array(point_img), np.array(I_stk)], axis=1) ))
except Exception as e:
    print(f"Error in hint points sampling: {e}")
    raise e

# Exemplar
print("Exemplar sampling...")
try:
    I_exp = Image.open('../sample/images/1_exp.jpg').convert('RGB')
except:
    I_exp = Image.open('images/1_exp.jpg').convert('RGB')
display(I_exp)

try:
    points, warped = colorizer.get_strokes_from_exemplar(I_gray, I_exp)
    warped = color_resize(I_gray, warped)
    display(warped)
    point_img = draw_strokes(I_gray, [256, 256], points)
    I_exp = colorizer.sample(I_gray, points, topk=100)
    display(Image.fromarray( np.concatenate([np.array(point_img), np.array(I_exp)], axis=1) ))
except Exception as e:
    print(f"Error in exemplar sampling: {e}")
    raise e

# Text prompt
print("Text prompt sampling...")
try:
    nltk.download('punkt', quiet=True)
except:
    pass

try:
    text_prompt = 'green jacket'
    points, heatmaps = colorizer.get_strokes_from_clip(I_gray, text_prompt)
    point_img = draw_strokes(I_gray, [256, 256], points)
    display(heatmaps[0])
    I_text = colorizer.sample(I_gray, points, topk=100)
    display(Image.fromarray( np.concatenate([np.array(point_img), np.array(I_text)], axis=1) ))
except Exception as e:
    print(f"Error in text prompt sampling: {e}")
    raise e

print("Done.")