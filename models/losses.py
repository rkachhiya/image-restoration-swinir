"""
Custom loss functions for image restoration.

Includes:
- Charbonnier Loss (robust L1)
- FFT Frequency Loss (frequency domain supervision)
- SSIM Loss (structural similarity)
- Combined Loss (weighted sum of all losses)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant) — more robust to outliers than MSE.
    
    L_charb = sqrt((pred - target)^2 + eps^2)
    
    This is preferred over MSE because:
    1. Less sensitive to outlier pixels (e.g., from speckle noise)
    2. Doesn't over-penalize large differences
    3. Produces sharper results than MSE (avoids over-smoothing)
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return loss.mean()


class FFTLoss(nn.Module):
    """FFT-based Frequency Loss.
    
    Computes L1 loss in the frequency domain using FFT.
    This helps the model:
    1. Recover high-frequency details lost during downsampling
    2. Remove noise patterns that have specific frequency signatures
    3. Preserve edge sharpness (edges = high frequency content)
    
    The speaker emphasized: frequency-based losses help counter degradation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Compute 2D FFT in FP32 to avoid ComplexHalf issues with AMP
        pred_fft = torch.fft.fft2(pred.float(), norm="ortho")
        target_fft = torch.fft.fft2(target.float(), norm="ortho")
        
        # Compute magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Also compare phase
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # L1 loss on magnitude and phase
        mag_loss = F.l1_loss(pred_mag, target_mag)
        phase_loss = F.l1_loss(pred_phase, target_phase)
        
        return mag_loss + 0.1 * phase_loss


class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) Loss.
    
    SSIM measures perceptual similarity between two images.
    Loss = 1 - SSIM (so minimizing loss maximizes SSIM).
    
    Uses a sliding window approach with Gaussian weighting.
    """

    def __init__(self, window_size=11, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def forward(self, pred, target):
        channel = pred.size(1)
        
        if self.window.device != pred.device or self.window.dtype != pred.dtype:
            self.window = self._create_window(self.window_size, channel).to(pred.device, pred.dtype)
        
        ssim_val = self._ssim(pred, target, self.window, self.window_size, channel)
        return 1.0 - ssim_val


class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel operator.
    
    Emphasizes reconstruction quality at edges and boundaries,
    which is critical for semiconductor inspection images where
    sharp edges define chip structures.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels
        k_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        k_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weight_x', k_x)
        self.register_buffer('weight_y', k_y)

    def forward(self, pred, target):
        # Compute edges for prediction
        pred_edge_x = F.conv2d(pred, self.weight_x, padding=1)
        pred_edge_y = F.conv2d(pred, self.weight_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)

        # Compute edges for target
        target_edge_x = F.conv2d(target, self.weight_x, padding=1)
        target_edge_y = F.conv2d(target, self.weight_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)

        return F.l1_loss(pred_edge, target_edge)


class CombinedLoss(nn.Module):
    """Combined loss function with configurable weights.
    
    Total Loss = w1 * Charbonnier + w2 * FFT + w3 * SSIM + w4 * Edge
    
    Design rationale:
    - Charbonnier: Primary reconstruction loss (robust to outliers)
    - FFT: Frequency-domain supervision for detail recovery
    - SSIM: Structural/perceptual quality
    - Edge: Sharp edge reconstruction
    """

    def __init__(self, charbonnier_weight=1.0, fft_weight=0.1, ssim_weight=0.05,
                 edge_weight=0.05, charbonnier_eps=1e-6):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)
        self.fft = FFTLoss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()

        self.w_charb = charbonnier_weight
        self.w_fft = fft_weight
        self.w_ssim = ssim_weight
        self.w_edge = edge_weight

    def forward(self, pred, target):
        loss_charb = self.charbonnier(pred, target)
        loss_fft = self.fft(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_edge = self.edge(pred, target)

        total = (self.w_charb * loss_charb +
                 self.w_fft * loss_fft +
                 self.w_ssim * loss_ssim +
                 self.w_edge * loss_edge)

        return total, {
            "charbonnier": loss_charb.item(),
            "fft": loss_fft.item(),
            "ssim": loss_ssim.item(),
            "edge": loss_edge.item(),
            "total": total.item(),
        }


if __name__ == "__main__":
    # Quick test
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randn(2, 1, 256, 256)
    
    loss_fn = CombinedLoss()
    loss, loss_dict = loss_fn(pred, target)
    print(f"Combined loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
