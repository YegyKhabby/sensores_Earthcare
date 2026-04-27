#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stack_monthly_images.py
=======================
Stack monthly comparison images vertically (one above the other) for all months.

This script:
  1. For each month, loads the two monthly comparison plots:
     - Ze_vs_rr_<month>.png
     - Ze_fwd_vs_cloudnet_<month>.png
  2. Stacks them vertically (Ze_vs_rr on top, Ze_fwd_vs_cloudnet below)
  3. Saves the combined image

Usage
-----
  python3 stack_monthly_images.py <input_dir> <output_dir> [month1 month2 ...]

Examples:
  python3 stack_monthly_images.py \
    output/comparison/cloudnet_vs_fwd_masked_shift30/monthly \
    output/comparison/cloudnet_vs_fwd_masked_shift30/stacked
  
  python3 stack_monthly_images.py \
    output/comparison/cloudnet_vs_fwd_masked_shift30/monthly \
    output/comparison/cloudnet_vs_fwd_masked_shift30/stacked \
    202502 202503 202504 202505 202506 202507
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def stack_images(img1_path, img2_path, output_path):
    """
    Stack two images vertically (img1 on top, img2 on bottom).
    
    Parameters
    ----------
    img1_path : Path
        Path to the top image
    img2_path : Path
        Path to the bottom image
    output_path : Path
        Path where the combined image will be saved
    """
    # Load images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Get dimensions
    w1, h1 = img1.size
    w2, h2 = img2.size
    
    # Make images the same width (use the maximum width)
    max_width = max(w1, w2)
    
    # If widths differ, resize to same width
    if w1 != max_width:
        img1 = img1.resize((max_width, int(h1 * max_width / w1)), Image.Resampling.LANCZOS)
    if w2 != max_width:
        img2 = img2.resize((max_width, int(h2 * max_width / w2)), Image.Resampling.LANCZOS)
    
    # Create new image with combined height
    _, h1_new = img1.size
    _, h2_new = img2.size
    total_height = h1_new + h2_new
    
    # Create combined image (white background)
    combined = Image.new('RGB', (max_width, total_height), color='white')
    
    # Paste images
    combined.paste(img1, (0, 0))
    combined.paste(img2, (0, h1_new))
    
    # Save
    combined.save(output_path)
    print(f'  Stacked → {output_path.name}')


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    # Get months to process
    if len(sys.argv) > 3:
        months = sys.argv[3:]
    else:
        # Find all months in input directory
        months = sorted([
            f.stem.split('_')[-1]
            for f in input_dir.glob('*Ze_vs_rr_*.png')
        ])
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Stacking monthly images from {input_dir}')
    print(f'Output directory: {output_dir}\n')
    
    for month in months:
        img1_path = input_dir / f'monthly_Ze_vs_rr_{month}.png'
        img2_path = input_dir / f'monthly_Ze_fwd_vs_cloudnet_{month}.png'
        output_path = output_dir / f'stacked_{month}.png'
        
        if not img1_path.exists():
            print(f'  ⚠ {img1_path.name} not found')
            continue
        if not img2_path.exists():
            print(f'  ⚠ {img2_path.name} not found')
            continue
        
        stack_images(img1_path, img2_path, output_path)
    
    print(f'\nDone! Stacked images saved to {output_dir}')


if __name__ == '__main__':
    main()
