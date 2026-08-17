"""
Data augmentation pipeline for image restoration training.

Augmentations are applied identically to both LR and GT images
to maintain spatial correspondence.

Key strategies:
- Geometric: flips, rotations (standard for restoration tasks)
- Intensity: mild scaling to improve robustness to intensity shifts
- Mixup: blend pairs for OOD generalization
"""

import numpy as np
import random


def random_horizontal_flip(lr, gt, prob=0.5):
    """Randomly flip both images horizontally."""
    if random.random() < prob:
        lr = lr[:, :, ::-1].copy()
        gt = gt[:, :, ::-1].copy()
    return lr, gt


def random_vertical_flip(lr, gt, prob=0.5):
    """Randomly flip both images vertically."""
    if random.random() < prob:
        lr = lr[:, ::-1, :].copy()
        gt = gt[:, ::-1, :].copy()
    return lr, gt


def random_rotation(lr, gt):
    """Randomly rotate both images by 0, 90, 180, or 270 degrees."""
    k = random.randint(0, 3)
    if k > 0:
        lr = np.rot90(lr, k, axes=(1, 2)).copy()
        gt = np.rot90(gt, k, axes=(1, 2)).copy()
    return lr, gt


def random_intensity_scale(lr, gt, scale_range=(0.9, 1.1)):
    """Randomly scale pixel intensities.
    
    Applied identically to both LR and GT.
    Helps the model handle images with different intensity distributions.
    """
    scale = random.uniform(*scale_range)
    lr = lr * scale
    gt = gt * scale
    return lr, gt


def random_intensity_shift(lr, gt, shift_range=(-0.05, 0.05)):
    """Randomly shift pixel intensities.
    
    Small additive shift to improve robustness.
    """
    shift = random.uniform(*shift_range)
    lr = lr + shift
    gt = gt + shift
    return lr, gt


class AugmentationPipeline:
    """Configurable augmentation pipeline.
    
    Args:
        horizontal_flip: Enable horizontal flipping.
        vertical_flip: Enable vertical flipping.
        rotation: Enable 90° rotation augmentations.
        intensity_scale: Tuple (min_scale, max_scale) or None.
        intensity_shift: Tuple (min_shift, max_shift) or None.
    """

    def __init__(self, horizontal_flip=True, vertical_flip=True,
                 rotation=True, intensity_scale=(0.9, 1.1),
                 intensity_shift=None):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.intensity_scale = intensity_scale
        self.intensity_shift = intensity_shift

    def __call__(self, lr, gt):
        """Apply augmentations to an (LR, GT) pair.
        
        Args:
            lr: numpy array (C, H_lr, W_lr)
            gt: numpy array (C, H_gt, W_gt)
        
        Returns:
            Augmented (lr, gt) pair.
        """
        # Geometric augmentations
        if self.horizontal_flip:
            lr, gt = random_horizontal_flip(lr, gt)
        
        if self.vertical_flip:
            lr, gt = random_vertical_flip(lr, gt)
        
        if self.rotation:
            lr, gt = random_rotation(lr, gt)

        # Intensity augmentations
        if self.intensity_scale is not None:
            lr, gt = random_intensity_scale(lr, gt, self.intensity_scale)

        if self.intensity_shift is not None:
            lr, gt = random_intensity_shift(lr, gt, self.intensity_shift)

        return lr, gt


def build_augmentation(config):
    """Build augmentation pipeline from config.
    
    Args:
        config: dict with augmentation configuration.
    
    Returns:
        AugmentationPipeline instance.
    """
    aug_config = config.get("augmentation", {})
    
    return AugmentationPipeline(
        horizontal_flip=aug_config.get("horizontal_flip", True),
        vertical_flip=aug_config.get("vertical_flip", True),
        rotation=aug_config.get("rotation", True),
        intensity_scale=aug_config.get("intensity_scale", (0.9, 1.1)),
        intensity_shift=aug_config.get("intensity_shift", None),
    )


if __name__ == "__main__":
    # Quick test
    aug = AugmentationPipeline()
    lr = np.random.randn(1, 64, 64).astype(np.float32)
    gt = np.random.randn(1, 128, 128).astype(np.float32)
    
    lr_aug, gt_aug = aug(lr, gt)
    print(f"LR: {lr.shape} -> {lr_aug.shape}")
    print(f"GT: {gt.shape} -> {gt_aug.shape}")
