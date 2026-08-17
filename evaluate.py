"""
KLA HACKATHON 2026 — STANDALONE EVALUATION SCRIPT

Accepts:
  --input_dir  : Path to directory containing degraded input images
  --output_dir : Path to directory where restored output images will be saved
  --model_path : Path to trained model checkpoint (.pt)
  --batch_size : Inference batch size (default: 16)
  --gt_dir     : [Optional] Path to ground-truth images for metric computation
  --benchmark  : [Optional] Run warm-up + timed benchmark and report throughput
  --save_fmt   : Output format: 'npy' or 'png' (default: npy)

This script will be executed AS-IS by KLA's benchmarking team on NVIDIA H100.
"""

import os
import sys
import glob
import time
import json
import shutil
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Optional LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("[WARNING] lpips not installed. LPIPS will be skipped.")


def parse_args():
    parser = argparse.ArgumentParser(description="KLA Image Restoration — Evaluation")
    parser.add_argument("--input_dir",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="weights/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gt_dir",     type=str, default=None)
    parser.add_argument("--benchmark",  action="store_true")
    parser.add_argument("--save_fmt",   type=str, default="npy", choices=["npy", "png"])
    parser.add_argument("--device",     type=str, default="auto")
    return parser.parse_args()


def load_model(model_path, device):
    """Load SwinIR model. Falls back to configs/config.yaml if needed."""
    from models.swinir import build_swinir
    import yaml

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[Inference] Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Try to get model config from checkpoint
    config = checkpoint.get("config", None)
    if config is not None and "model" in config:
        print("[Inference] Using model config from checkpoint")
        model = build_swinir(config["model"])
    else:
        # Fallback: load from configs/config.yaml
        print("[Inference] No config in checkpoint — loading from configs/config.yaml")
        config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            model = build_swinir(yaml_config["model"])
        else:
            raise FileNotFoundError("No config found in checkpoint or configs/config.yaml")

    # Load state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Inference] Model parameters: {total_params:,}")
    return model


def load_image(path):
    """Load image as (1, H, W) float32 in [0, 1] or raw values."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        img = np.load(path).astype(np.float32)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3 and img.shape[2] == 1:
            img = img.transpose(2, 0, 1)
    else:
        img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
        img = img[np.newaxis, ...]
    return img


def save_image(path, img, fmt="npy"):
    """Save image. img is (H, W) float32."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # CRITICAL: Clip to [0, 1] as required by KLA
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    if fmt == "npy":
        np.save(path, img)
    else:
        img_uint8 = (img * 255.0).astype(np.uint8)
        Image.fromarray(img_uint8, mode="L").save(path)


def compute_psnr(pred, target, data_range=1.0):
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def compute_ssim(pred, target, data_range=1.0):
    import math
    window_size = 11
    sigma = 1.5
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1D = gauss.unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D.expand(1, 1, window_size, window_size).contiguous()

    p = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
    t = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
    window = window.to(p.device)

    mu1 = F.conv2d(p, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(t, window, padding=window_size // 2, groups=1)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(p * p, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(t * t, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(p * t, window, padding=window_size // 2, groups=1) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def compute_lpips(pred, target, loss_fn, device):
    p = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0).to(device)
    t = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0).to(device)

    if p.shape[2] < 256 or p.shape[3] < 256:
        p = F.interpolate(p, size=(256, 256), mode="bilinear", align_corners=False)
        t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)

    p = p.repeat(1, 3, 1, 1)
    t = t.repeat(1, 3, 1, 1)
    p = p * 2.0 - 1.0
    t = t * 2.0 - 1.0

    with torch.no_grad():
        dist = loss_fn(p, t)
    return dist.item()


def run_inference(model, input_files, output_dir, batch_size, device, save_fmt="npy"):
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    batch_imgs, batch_names = [], []

    for i, file_path in enumerate(input_files):
        img = load_image(file_path)
        batch_imgs.append(img)
        batch_names.append(os.path.basename(file_path))

        if len(batch_imgs) == batch_size or i == len(input_files) - 1:
            batch_tensor = torch.from_numpy(np.array(batch_imgs)).float().to(device)

            with torch.no_grad():
                # FIXED: Use torch.autocast for PyTorch 2.x compatibility
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    pred_tensor = model(batch_tensor)

            pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
            pred_np = pred_tensor.float().cpu().numpy()

            for j in range(pred_np.shape[0]):
                out_img = np.squeeze(pred_np[j])
                base_name = os.path.splitext(batch_names[j])[0]
                out_name = base_name + (".npy" if save_fmt == "npy" else ".png")
                out_path = os.path.join(output_dir, out_name)
                save_image(out_path, out_img, fmt=save_fmt)
                output_paths.append(out_path)

            batch_imgs, batch_names = [], []

    return output_paths, len(input_files)


def evaluate_metrics(output_dir, gt_dir, device, lpips_fn=None):
    out_files = sorted(glob.glob(os.path.join(output_dir, "*.npy")) + glob.glob(os.path.join(output_dir, "*.png")))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")) + glob.glob(os.path.join(gt_dir, "*.png")))

    out_map = {os.path.splitext(os.path.basename(f))[0]: f for f in out_files}
    gt_map = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
    common = sorted(set(out_map.keys()) & set(gt_map.keys()))

    if len(common) == 0:
        print("[Metrics] No matching files found.")
        return None

    print(f"[Metrics] Evaluating {len(common)} images...")
    psnr_list, ssim_list, lpips_list = [], [], []
    per_image = {}

    for name in common:
        pred_path = out_map[name]
        pred = np.load(pred_path).astype(np.float32) if pred_path.endswith(".npy") else np.array(Image.open(pred_path).convert("L")).astype(np.float32) / 255.0

        gt_path = gt_map[name]
        gt = np.load(gt_path).astype(np.float32) if gt_path.endswith(".npy") else np.array(Image.open(gt_path).convert("L")).astype(np.float32) / 255.0

        if pred.shape != gt.shape:
            gt = np.array(Image.fromarray((gt * 255).astype(np.uint8)).resize((pred.shape[1], pred.shape[0]), Image.BICUBIC)) / 255.0

        psnr = compute_psnr(pred, gt)
        ssim = compute_ssim(pred, gt)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        entry = {"psnr": round(psnr, 4), "ssim": round(ssim, 4)}

        if lpips_fn is not None:
            lp = compute_lpips(pred, gt, lpips_fn, device)
            lpips_list.append(lp)
            entry["lpips"] = round(lp, 4)

        per_image[name] = entry

    results = {
        "num_images": len(common),
        "mean_psnr": round(float(np.mean(psnr_list)), 4),
        "mean_ssim": round(float(np.mean(ssim_list)), 4),
        "std_psnr": round(float(np.std(psnr_list)), 4),
        "std_ssim": round(float(np.std(ssim_list)), 4),
        "per_image": per_image,
    }

    if lpips_list:
        results["mean_lpips"] = round(float(np.mean(lpips_list)), 4)
        results["std_lpips"] = round(float(np.std(lpips_list)), 4)

    return results


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Inference] Device: {device}")

    # Load model
    model = load_model(args.model_path, device)

    # Collect input files
    input_files = sorted(
        glob.glob(os.path.join(args.input_dir, "*.npy")) +
        glob.glob(os.path.join(args.input_dir, "*.png")) +
        glob.glob(os.path.join(args.input_dir, "*.jpg")) +
        glob.glob(os.path.join(args.input_dir, "*.tif"))
    )
    if len(input_files) == 0:
        input_files = sorted(
            glob.glob(os.path.join(args.input_dir, "**", "*.npy"), recursive=True) +
            glob.glob(os.path.join(args.input_dir, "**", "*.png"), recursive=True)
        )

    if len(input_files) == 0:
        print(f"[ERROR] No input images found in {args.input_dir}")
        sys.exit(1)

    print(f"[Inference] Found {len(input_files)} input images")

    # LPIPS
    lpips_fn = None
    if LPIPS_AVAILABLE and args.gt_dir is not None:
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        print("[Metrics] LPIPS initialized (AlexNet backbone)")

    # Benchmark or normal inference
    if args.benchmark:
        print("\n" + "=" * 60)
        print("BENCHMARK MODE")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"PyTorch: {torch.__version__}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Total images: {len(input_files)}")

        # Warm-up
        warmup_files = input_files[:min(args.batch_size, len(input_files))]
        _ = run_inference(model, warmup_files, args.output_dir + "_warmup", args.batch_size, device, args.save_fmt)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Clean output dir
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

        # Timed run
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        output_paths, total_processed = run_inference(model, input_files, args.output_dir, args.batch_size, device, args.save_fmt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_time = end - start
        throughput = total_processed / total_time if total_time > 0 else 0
        ms_per_img = (total_time / total_processed) * 1000 if total_processed > 0 else 0

        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total images: {total_processed}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Throughput: {throughput:.2f} img/s")
        print(f"Time per image: {ms_per_img:.2f} ms")
        print("=" * 60)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_paths, total_processed = run_inference(model, input_files, args.output_dir, args.batch_size, device, args.save_fmt)
        print(f"[Inference] Restored {total_processed} images -> {args.output_dir}")

    # Metrics
    if args.gt_dir is not None and os.path.isdir(args.gt_dir):
        metrics_results = evaluate_metrics(args.output_dir, args.gt_dir, device, lpips_fn)
        if metrics_results is not None:
            print("\n" + "=" * 60)
            print("QUALITY METRICS")
            print("=" * 60)
            print(f"Images evaluated: {metrics_results['num_images']}")
            print(f"Mean PSNR: {metrics_results['mean_psnr']:.4f} dB")
            print(f"Mean SSIM: {metrics_results['mean_ssim']:.4f}")
            if "mean_lpips" in metrics_results:
                print(f"Mean LPIPS: {metrics_results['mean_lpips']:.4f}")
            print("=" * 60)

    # Save report
    report = {
        "model_path": args.model_path,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "device": str(device),
        "num_images": len(input_files),
    }
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Report] Saved: {report_path}")
    print("\n[Done] Inference complete.")


if __name__ == "__main__":
    main()
