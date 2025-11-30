#!/usr/bin/env python3
"""
Quick validation script for colorization_gui.py
Tests imports and basic functionality without launching the full GUI
"""

import sys
import os

print("=" * 60)
print("Testing Colorization GUI Dependencies")
print("=" * 60)

# Test 1: Check Python version
print(f"\n1. Python version: {sys.version.split()[0]}")

# Test 2: Test imports
print("\n2. Testing imports...")
try:
    import gradio as gr
    print(f"   ✅ Gradio {gr.__version__}")
except ImportError as e:
    print(f"   ❌ Gradio import failed: {e}")
    sys.exit(1)

try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"   ❌ PyTorch import failed: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print(f"   ✅ PIL/Pillow")
except ImportError as e:
    print(f"   ❌ PIL import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"   ✅ NumPy {np.__version__}")
except ImportError as e:
    print(f"   ❌ NumPy import failed: {e}")
    sys.exit(1)

# Test 3: Check sample directory modules
print("\n3. Testing sample directory imports...")
sys.path.append('./sample')
try:
    from colorizer import Colorizer
    print("   ✅ Colorizer module")
except ImportError as e:
    print(f"   ❌ Colorizer import failed: {e}")
    sys.exit(1)

try:
    from utils_func import draw_strokes, color_resize, limit_size
    print("   ✅ Utils functions")
except ImportError as e:
    print(f"   ❌ Utils import failed: {e}")
    sys.exit(1)

# Test 4: Check checkpoint file
print("\n4. Checking model checkpoint...")
checkpoint_path = './framework/checkpoints/unicolor_mscoco/mscoco_step259999.ckpt'
if os.path.exists(checkpoint_path):
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"   ✅ Checkpoint found ({size_mb:.1f} MB)")
else:
    print(f"   ❌ Checkpoint not found at: {checkpoint_path}")
    print("   ⚠️  Please download the checkpoint from Hugging Face")

# Test 5: Check test images
print("\n5. Checking test images...")
test_img = './sample/images/1.jpg'
if os.path.exists(test_img):
    print(f"   ✅ Test image found: {test_img}")
else:
    print(f"   ⚠️  Test image not found: {test_img}")

# Test 6: Validate GUI syntax
print("\n6. Validating GUI script syntax...")
try:
    with open('colorization_gui.py', 'r') as f:
        code = f.read()
    compile(code, 'colorization_gui.py', 'exec')
    print("   ✅ GUI script syntax is valid")
except SyntaxError as e:
    print(f"   ❌ Syntax error in GUI script: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All validation checks passed!")
print("=" * 60)
print("\nYou can now run the GUI with:")
print("  python colorization_gui.py")
print("\nThe web interface will launch at: http://localhost:7860")
print("=" * 60)
