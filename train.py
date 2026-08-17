"""
Standalone Training Script for SwinIR Image Restoration.

Features:
- Configurable via YAML configuration file
- Multi-objective loss (Charbonnier, FFT, SSIM, Edge)
- Mixed Precision Training (Automatic Mixed Precision - AMP)
- Exponential Moving Average (EMA) of model weights
- Cosine Annealing learning rate schedule with warm restarts
- Validation logging with PSNR and SSIM tracking
- Model checkpoint saving (best model & periodic saves)
"""

import os
import sys
import copy
import time
import yaml
import argparse
from tqdm import tqdm

import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from models.swinir import build_swinir
from models.losses import CombinedLoss
from utils.dataset import create_dataloaders
from utils.augmentations import build_augmentation
from utils.metrics import compute_psnr, compute_ssim


class EMAModel:
    """Exponential Moving Average of model weights for smoother convergence."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def get_model(self):
        return self.ema_model


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                    device, use_amp=True, ema=None):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {}

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for lr, gt in pbar:
        lr, gt = lr.to(device), gt.to(device)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            pred = model(lr)
            loss, loss_dict = criterion(pred, gt)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    return avg_loss, avg_components


def validate(model, val_loader, criterion, device, use_amp=True):
    """Validate model performance using PSNR and SSIM metrics."""
    model.eval()
    total_loss = 0.0
    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for lr, gt in tqdm(val_loader, desc="Validation", leave=False):
            lr, gt = lr.to(device), gt.to(device)

            with autocast(enabled=use_amp):
                pred = model(lr)
                loss, _ = criterion(pred, gt)

            total_loss += loss.item()

            # Compute PSNR & SSIM per sample in batch
            pred_np = torch.clamp(pred, 0.0, 1.0).cpu().numpy()
            gt_np = gt.cpu().numpy()

            for i in range(pred_np.shape[0]):
                psnr_scores.append(compute_psnr(pred_np[i], gt_np[i]))
                ssim_scores.append(compute_ssim(pred_np[i], gt_np[i]))

    avg_loss = total_loss / len(val_loader)
    mean_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0.0
    mean_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0.0

    return avg_loss, mean_psnr, mean_ssim


def main():
    parser = argparse.ArgumentParser(description="Train SwinIR for Image Restoration")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--work_dir", type=str, default="work_dir", help="Directory to save logs and weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    os.makedirs(args.work_dir, exist_ok=True)
    weights_dir = os.path.join(args.work_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # Create DataLoaders
    aug_fn = build_augmentation(config)
    train_loader, val_loader = create_dataloaders(config, augmentation_fn=aug_fn)

    # Build Model
    model = build_swinir(config["model"]).to(device)
    print(f"[Train] Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss, Optimizer, Scheduler, AMP Scaler
    loss_cfg = config["training"]["loss"]
    criterion = CombinedLoss(
        charbonnier_weight=loss_cfg.get("charbonnier_weight", 1.0),
        fft_weight=loss_cfg.get("fft_weight", 0.1),
        ssim_weight=loss_cfg.get("ssim_weight", 0.05),
        charbonnier_eps=loss_cfg.get("charbonnier_eps", 1e-6)
    ).to(device)

    opt_cfg = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01))
    )

    sched_cfg = config["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_cfg.get("T_0", 50),
        T_mult=sched_cfg.get("T_mult", 2),
        eta_min=float(sched_cfg.get("eta_min", 1e-6))
    )

    use_amp = config["training"].get("use_amp", True) and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    ema = EMAModel(model, decay=config["training"].get("ema_decay", 0.999)) if config["training"].get("use_ema", True) else None

    # TensorBoard logging
    writer = SummaryWriter(log_dir=os.path.join(args.work_dir, "logs"))

    start_epoch = 0
    best_psnr = 0.0

    if args.resume and os.path.isfile(args.resume):
        print(f"[Train] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint.get("best_psnr", 0.0)

    epochs = config["training"]["epochs"]
    print(f"[Train] Starting training from epoch {start_epoch} to {epochs}...")

    for epoch in range(start_epoch, epochs):
        start_time = time.time()

        train_loss, loss_comps = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp=use_amp, ema=ema
        )
        scheduler.step()

        # Validate using EMA model if available
        eval_model = ema.get_model() if ema is not None else model
        val_loss, val_psnr, val_ssim = validate(eval_model, val_loader, criterion, device, use_amp=use_amp)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch+1}/{epochs}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | LR: {current_lr:.6f}")

        # Tensorboard logging
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/PSNR", val_psnr, epoch)
        writer.add_scalar("Val/SSIM", val_ssim, epoch)
        writer.add_scalar("Train/LR", current_lr, epoch)

        for k, v in loss_comps.items():
            writer.add_scalar(f"LossComponents/{k}", v, epoch)

        # Save best model checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_path = os.path.join(weights_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model": eval_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_psnr": best_psnr,
                "config": config,
            }, best_path)
            print(f"  --> Saved new best model checkpoint to {best_path} (PSNR: {best_psnr:.2f} dB)")

        # Periodic checkpoint save
        save_every = config["training"].get("save_every", 10)
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(weights_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_psnr": val_psnr,
                "config": config,
            }, ckpt_path)

    writer.close()
    print(f"[Train] Training complete. Best Validation PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
