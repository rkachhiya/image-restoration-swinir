# KLA Hackathon 2026 — AI-Based Image Restoration for Semiconductor Inspection

## Problem
Restore degraded semiconductor inspection images affected by three simultaneous degradations applied in random order:
- **Speckle noise** — multiplicative grain, pushes pixel values beyond true range
- **Additive Gaussian noise** — soft, hazy loss of edge sharpness  
- **2× bicubic downsampling** — spatial resolution reduction, fine detail lost

**Input:** 64×64 grayscale NoisyLR (values may intentionally exceed [0,1])  
**Output:** 128×128 grayscale restored, clipped to [0,1]

## Solution Overview
- **Architecture:** Lightweight SwinIR transformer (1.1M parameters)
- **Task:** Joint single-step denoising + 2× super-resolution
- **Key design:** Windowed self-attention (window_size=8) captures local speckle patterns better than CNNs
- **Size:** 26.8 MB checkpoint — deployable on edge hardware

## Repository Structure
```
.
├── evaluate.py          # Standalone inference script (KLA evaluator compatible)
├── train.py             # Reproducible training script
├── baseline.py          # Bicubic + Median baseline for comparison
├── configs/
│   └── config.yaml      # Model & training configuration
├── models/
│   ├── swinir.py        # SwinIR architecture
│   ├── losses.py        # Combined loss functions
│   └── __init__.py
├── utils/
│   ├── dataset.py       # Data loader & augmentation
│   ├── augmentations.py # Augmentation pipeline
│   └── metrics.py       # PSNR, SSIM
├── weights/
│   └── best_model.pt    # Final checkpoint (28.08 dB)
└── requirements.txt     # Environment specification
```

## Results — Validated on 320-Image Hold-Out Set

| Metric | Value |
|--------|-------|
| PSNR   | **28.08 dB** |
| SSIM   | **0.7725** |
| LPIPS  | **0.2688** |

### Baseline Comparison

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Parameters |
|--------|--------|--------|---------|------------|
| Bicubic + Median Filter | 24.94 dB | 0.6227 | — | 0 |
| **Our SwinIR** | **28.08 dB** | **0.7725** | **0.2688** | **1.1M** |
| **Improvement** | **+3.14 dB** | **+14.98%** | — | — |

## Inference

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Evaluation (KLA Format)
```bash
python evaluate.py \
  --input_dir <path/to/degraded/images> \
  --output_dir <path/to/save/restored> \
  --model_path weights/best_model.pt \
  --batch_size 16
```

### Optional: Compute Metrics Against Ground Truth
```bash
python evaluate.py \
  --input_dir data/val/NoisyLR \
  --output_dir val_pred \
  --model_path weights/best_model.pt \
  --gt_dir data/val/GT \
  --batch_size 16
```

### Optional: Benchmark Throughput
```bash
python evaluate.py \
  --input_dir data/test/NoisyLR \
  --output_dir bench_out \
  --model_path weights/best_model.pt \
  --batch_size 16 \
  --benchmark
```

## Training

```bash
python train.py --config configs/config.yaml --work_dir work_dir
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=5e-4, weight_decay=0.01) |
| Scheduler | Cosine Annealing Warm Restarts (T_0=50, T_mult=2) |
| Loss | Charbonnier (1.0) + SSIM (0.05) + Edge (0.05) |
| Augmentation | H/V flip, 90° rotation, intensity scaling, CutMix, noise injection |
| Epochs | 45 |
| Batch size | 4 (CPU) / 16 (GPU) |
| EMA | Enabled (decay=0.999) |
| Gradient clipping | max_norm=1.0 |

## Runtime Performance

| Platform | Throughput | Time/Image |
|----------|-----------|------------|
| CPU (laptop) | 1.33 img/s | 753 ms |
| **NVIDIA H100 (estimated)** | **~180 img/s** | **~5.5 ms** |

- Lightweight 1.1M parameter model minimizes memory bandwidth
- `torch.compile()` ready for H100 optimization
- Batch processing with automatic GPU memory handling

## Reproducibility
- **Random seed:** 42 (set in `train.py` via `torch.manual_seed`, `np.random.seed`, `random.seed`)
- **Validation split:** 10% (320 images), deterministic via `torch.Generator().manual_seed(42)`
- **Config:** `configs/config.yaml` — `embed_dim=60`, `depths=[6,6,6,6]`, `num_heads=[6,6,6,6]`
- **No source code edits required** by evaluators to run inference

## External Resources
- **No external pre-trained weights** used in the final submitted model.
- Swin2SR pre-trained weights were evaluated during development but showed domain mismatch (27.16 dB vs 28.08 dB trained from scratch); not used in final submission.
- **SwinIR reference:** Liang et al., "SwinIR: Image Restoration Using Swin Transformer", ICCVW 2021.

## Limitations & Failure Cases
- Model occasionally struggles with **very high speckle variance** — fine periodic structures can be partially smoothed
- Performance bounded by **3,200 training images**; larger or more diverse datasets may improve generalization
- LPIPS score (0.2688) reflects grayscale-domain evaluation on small (128×128) images; perceptual quality may differ from natural-image benchmarks

## Submission Checklist
- [x] Standalone inference script (`evaluate.py`) with `--input_dir` and `--output_dir`
- [x] Model weights included (`weights/best_model.pt`, 26.8 MB)
- [x] Training code reproducible (`train.py` + `configs/config.yaml`)
- [x] Environment specification (`requirements.txt`)
- [x] PSNR, SSIM, LPIPS reported on validation hold-out
- [x] Baseline comparison (Bicubic + Median)
- [x] End-to-end runtime documented
- [x] Random seeds tracked
- [x] External resources disclosed
- [x] Output clipped to [0,1] before saving
- [x] Filename preservation implemented
