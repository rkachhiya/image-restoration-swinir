"""
KLA Hackathon 2026 — Baseline Restoration Script

Simple baseline: bicubic upsampling + median filter denoising.
No neural network. Used for comparison against SwinIR model.

Usage:
    python baseline.py --input_dir data/val/NoisyLR --output_dir baseline_out
"""

import os
import argparse
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Image Restoration (Bicubic + Median)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with degraded LR .npy images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save baseline outputs")
    return parser.parse_args()


def baseline_restore(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if len(files) == 0:
        files = sorted(glob.glob(os.path.join(input_dir, "**", "*.npy"), recursive=True))

    print(f"[Baseline] Found {len(files)} input images")

    for fname in files:
        # Load LR image
        img = np.load(fname).astype(np.float32)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze()  # (H, W, 1) -> (H, W)
        elif img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze()  # (1, H, W) -> (H, W)

        h, w = img.shape

        # 1. Bicubic upsample 2x
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode="L")
        img_up_pil = img_pil.resize((w * 2, h * 2), Image.BICUBIC)
        img_up = np.array(img_up_pil).astype(np.float32) / 255.0

        # 2. Median filter for denoising (3x3 kernel)
        img_denoised = median_filter(img_up, size=3)

        # 3. Clip to [0, 1]
        img_denoised = np.clip(img_denoised, 0.0, 1.0)

        # Save
        out_name = os.path.basename(fname)
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, img_denoised.astype(np.float32))

    print(f"[Baseline] Restored {len(files)} images -> {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    baseline_restore(args.input_dir, args.output_dir)

