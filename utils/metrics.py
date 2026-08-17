"""
Evaluation metrics for image restoration quality.

Metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity) — optional
"""

import numpy as np
import torch
import torch.nn.functional as F
import math


def compute_psnr(pred, target, data_range=1.0):
    """Compute PSNR between prediction and target.
    
    Args:
        pred: numpy array or torch tensor
        target: numpy array or torch tensor
        data_range: maximum pixel value (1.0 for normalized images)
    
    Returns:
        PSNR value in dB.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 10.0 * math.log10(data_range ** 2 / mse)
    return psnr


def compute_ssim(pred, target, data_range=1.0, window_size=11, channel=1):
    """Compute SSIM between prediction and target.
    
    Uses sliding window approach with Gaussian weighting.
    
    Args:
        pred: numpy array (H, W) or (C, H, W) or torch tensor
        target: numpy array (H, W) or (C, H, W) or torch tensor
        data_range: dynamic range of pixel values
        window_size: size of the Gaussian window
        channel: number of channels
    
    Returns:
        SSIM value (higher is better, max 1.0).
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    # Ensure 4D: (B, C, H, W)
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    channel = pred.size(1)
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([
        math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_lpips(pred, target, net='alex'):
    """Compute LPIPS perceptual distance (lower is better).
    
    Requires lpips package. Falls back gracefully if not available.
    
    Args:
        pred: torch tensor (B, C, H, W) or (C, H, W)
        target: torch tensor (B, C, H, W) or (C, H, W)
        net: backbone network ('alex', 'vgg', 'squeeze')
    
    Returns:
        LPIPS distance (lower is better).
    """
    try:
        import lpips
    except ImportError:
        print("WARNING: lpips not installed. Skipping LPIPS computation.")
        return float('nan')
    
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    # Ensure 4D
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # LPIPS expects 3-channel input; replicate grayscale to RGB
    if pred.size(1) == 1:
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
    
    # Normalize to [-1, 1] range for LPIPS
    pred = pred * 2.0 - 1.0
    target = target * 2.0 - 1.0
    
    loss_fn = lpips.LPIPS(net=net, verbose=False)
    loss_fn.eval()
    
    with torch.no_grad():
        distance = loss_fn(pred, target)
    
    return distance.mean().item()


def compute_all_metrics(pred, target, data_range=1.0):
    """Compute all metrics at once.
    
    Args:
        pred: numpy array or torch tensor
        target: numpy array or torch tensor
    
    Returns:
        dict with PSNR, SSIM, and LPIPS values.
    """
    psnr = compute_psnr(pred, target, data_range)
    ssim = compute_ssim(pred, target, data_range)
    
    return {
        "psnr": psnr,
        "ssim": ssim,
    }


def compute_metrics_batch(pred_dir, gt_dir, data_range=1.0):
    """Compute metrics for a batch of images.
    
    Args:
        pred_dir: Directory containing predicted .npy files.
        gt_dir: Directory containing ground truth .npy files.
    
    Returns:
        dict with average metrics and per-image results.
    """
    import os
    import glob
    
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    
    # Match by filename
    pred_names = {os.path.basename(f): f for f in pred_files}
    gt_names = {os.path.basename(f): f for f in gt_files}
    common = sorted(set(pred_names.keys()) & set(gt_names.keys()))
    
    all_psnr = []
    all_ssim = []
    per_image = {}
    
    for name in common:
        pred = np.load(pred_names[name]).astype(np.float32)
        gt = np.load(gt_names[name]).astype(np.float32)
        
        # Clip prediction to [0, 1] for fair comparison
        pred = np.clip(pred, 0.0, 1.0)
        
        metrics = compute_all_metrics(pred, gt, data_range)
        all_psnr.append(metrics["psnr"])
        all_ssim.append(metrics["ssim"])
        per_image[name] = metrics
    
    avg_metrics = {
        "avg_psnr": np.mean(all_psnr),
        "avg_ssim": np.mean(all_ssim),
        "num_images": len(common),
    }
    
    return avg_metrics, per_image


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        pred_dir, gt_dir = sys.argv[1], sys.argv[2]
        avg, per_img = compute_metrics_batch(pred_dir, gt_dir)
        print(f"Average PSNR: {avg['avg_psnr']:.2f} dB")
        print(f"Average SSIM: {avg['avg_ssim']:.4f}")
        print(f"Evaluated on {avg['num_images']} images")
    else:
        # Quick test with random data
        pred = np.random.rand(256, 256).astype(np.float32)
        target = pred + np.random.randn(256, 256).astype(np.float32) * 0.1
        target = np.clip(target, 0, 1)
        
        metrics = compute_all_metrics(pred, target)
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
