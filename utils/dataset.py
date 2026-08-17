"""
PyTorch Dataset for KLA Hackathon image restoration.

Handles:
- Loading .npy grayscale image pairs (NoisyLR → GT)
- Proper handling of out-of-range NoisyLR values (speckle noise artifact)
- Random patch cropping for training
- Integration with augmentation pipeline
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KLADataset(Dataset):
    """Dataset for paired NoisyLR ↔ GT image restoration training.
    
    Args:
        gt_dir: Path to ground truth images (.npy files).
        lr_dir: Path to noisy low-resolution images (.npy files).
        patch_size_lr: Crop size for LR images (default: 64).
        upscale: Upscaling factor (default: 2).
        augment: Whether to apply augmentations (default: True).
        augmentation_fn: Augmentation function to apply.
        is_train: Whether this is training mode.
    """

    def __init__(self, gt_dir, lr_dir, patch_size_lr=64, upscale=2,
                 augment=True, augmentation_fn=None, is_train=True):
        super().__init__()
        self.gt_dir = gt_dir
        self.lr_dir = lr_dir
        self.patch_size_lr = patch_size_lr
        self.patch_size_gt = patch_size_lr * upscale
        self.upscale = upscale
        self.augment = augment and is_train
        self.augmentation_fn = augmentation_fn
        self.is_train = is_train

        # Collect file paths
        self.gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))

        # Match pairs by filename
        gt_basenames = {os.path.basename(f): f for f in self.gt_files}
        lr_basenames = {os.path.basename(f): f for f in self.lr_files}
        
        common_names = sorted(set(gt_basenames.keys()) & set(lr_basenames.keys()))
        self.gt_files = [gt_basenames[n] for n in common_names]
        self.lr_files = [lr_basenames[n] for n in common_names]

        assert len(self.gt_files) == len(self.lr_files), \
            f"Mismatch: {len(self.gt_files)} GT vs {len(self.lr_files)} LR files"
        assert len(self.gt_files) > 0, \
            f"No matching pairs found in {gt_dir} and {lr_dir}"

        print(f"[KLADataset] Loaded {len(self.gt_files)} image pairs "
              f"(train={is_train}, augment={self.augment})")

    def __len__(self):
        return len(self.gt_files)

    def _load_npy(self, path):
        """Load .npy file and convert to float32 tensor."""
        img = np.load(path).astype(np.float32)
        if img.ndim == 2:
            img = img[np.newaxis, ...]  # (H, W) → (1, H, W)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = img.transpose(2, 0, 1)  # (H, W, 1) → (1, H, W)
        return img

    def _random_crop(self, lr_img, gt_img):
        """Extract random corresponding patches from LR and GT images."""
        _, h_lr, w_lr = lr_img.shape
        _, h_gt, w_gt = gt_img.shape
        
        ps_lr = self.patch_size_lr
        ps_gt = self.patch_size_gt

        # Ensure patch size doesn't exceed image size
        ps_lr = min(ps_lr, h_lr, w_lr)
        ps_gt = min(ps_gt, h_gt, w_gt)

        # Random top-left corner for LR
        top_lr = random.randint(0, h_lr - ps_lr)
        left_lr = random.randint(0, w_lr - ps_lr)

        # Corresponding GT coordinates
        top_gt = top_lr * self.upscale
        left_gt = left_lr * self.upscale

        lr_patch = lr_img[:, top_lr:top_lr + ps_lr, left_lr:left_lr + ps_lr]
        gt_patch = gt_img[:, top_gt:top_gt + ps_gt, left_gt:left_gt + ps_gt]

        return lr_patch, gt_patch

    def __getitem__(self, idx):
        # Load images
        gt_img = self._load_npy(self.gt_files[idx])  # (1, 256, 256) range [0, 1]
        lr_img = self._load_npy(self.lr_files[idx])  # (1, 128, 128) range may exceed [0, 1]

        # NOTE: Do NOT clip LR values to [0, 1]!
        # The out-of-range values from speckle noise are INFORMATIVE.
        # The model should learn to use this information.

        if self.is_train:
            # Random crop
            lr_img, gt_img = self._random_crop(lr_img, gt_img)

            # Apply augmentations
            if self.augment and self.augmentation_fn is not None:
                lr_img, gt_img = self.augmentation_fn(lr_img, gt_img)

        # Convert to tensors
        lr_tensor = torch.from_numpy(lr_img.copy())
        gt_tensor = torch.from_numpy(gt_img.copy())

        return lr_tensor, gt_tensor


class KLATestDataset(Dataset):
    """Dataset for test-time inference (no GT available).
    
    Args:
        lr_dir: Path to noisy low-resolution test images.
    """

    def __init__(self, lr_dir):
        super().__init__()
        self.lr_dir = lr_dir
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))
        assert len(self.lr_files) > 0, f"No .npy files found in {lr_dir}"
        print(f"[KLATestDataset] Loaded {len(self.lr_files)} test images")

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_img = np.load(self.lr_files[idx]).astype(np.float32)
        if lr_img.ndim == 2:
            lr_img = lr_img[np.newaxis, ...]  # (H, W) → (1, H, W)
        elif lr_img.ndim == 3 and lr_img.shape[2] == 1:
            lr_img = lr_img.transpose(2, 0, 1)

        lr_tensor = torch.from_numpy(lr_img.copy())
        filename = os.path.basename(self.lr_files[idx])
        return lr_tensor, filename


def create_dataloaders(config, augmentation_fn=None):
    """Create train and validation DataLoaders from config.
    
    Args:
        config: dict with data configuration.
        augmentation_fn: Optional augmentation function.
    
    Returns:
        train_loader, val_loader
    """
    gt_dir = config["data"]["train_gt_dir"]
    lr_dir = config["data"]["train_lr_dir"]
    val_split = config["data"].get("val_split", 0.1)
    batch_size = config["training"]["batch_size"]
    patch_size_lr = config["training"]["patch_size_lr"]
    num_workers = config["data"].get("num_workers", 4)
    pin_memory = config["data"].get("pin_memory", True)
    upscale = config["model"]["upscale"]

    # Create full dataset
    full_dataset = KLADataset(
        gt_dir=gt_dir, lr_dir=lr_dir,
        patch_size_lr=patch_size_lr, upscale=upscale,
        augment=True, augmentation_fn=augmentation_fn,
        is_train=True
    )

    # Split into train and validation
    total_len = len(full_dataset)
    val_len = max(1, int(total_len * val_split))
    train_len = total_len - val_len

    # Use deterministic split for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_len, val_len], generator=generator
    )

    # Validation dataset should not use augmentation
    # We create a separate dataset for validation
    val_full_dataset = KLADataset(
        gt_dir=gt_dir, lr_dir=lr_dir,
        patch_size_lr=patch_size_lr, upscale=upscale,
        augment=False, augmentation_fn=None,
        is_train=False  # No augmentation, but still crop for consistent eval
    )
    
    # Use same indices as the split
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"[DataLoader] Train: {train_len} samples, Val: {val_len} samples")
    print(f"[DataLoader] Batch size: {batch_size}, Workers: {num_workers}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test
    import sys
    gt_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train/GT"
    lr_dir = sys.argv[2] if len(sys.argv) > 2 else "data/train/NoisyLR"
    
    ds = KLADataset(gt_dir, lr_dir, patch_size_lr=64, upscale=2, augment=False)
    lr, gt = ds[0]
    print(f"LR: {lr.shape}, range [{lr.min():.4f}, {lr.max():.4f}]")
    print(f"GT: {gt.shape}, range [{gt.min():.4f}, {gt.max():.4f}]")
