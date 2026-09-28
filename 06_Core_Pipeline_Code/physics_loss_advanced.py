"""
FIXED + UPGRADED Advanced Physics Loss Module for Multi-Modal Geophysical Super-Resolution.

Fixes applied:
- FIX 1: SpectralLaplaceLoss now computes wavenumber grid DYNAMICALLY inside forward()
          instead of hardcoding at init time. Works for any patch size.
- NEW: Added SSIMLoss for structural preservation of ridges and fracture zones.

Implements:
1. Charbonnier Reconstruction Loss (smooth L1, robust to outliers)
2. Gallardo-Meju Structural Cross-Gradient Loss
3. Spectral Laplace Frequency Loss (dynamic wavenumber, any patch size)
4. Total Variation Loss (gridding artifact suppression)
5. SSIM Loss (structural similarity for ridge/fracture zone morphology)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss - smooth L1, robust to outliers"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps))


class CrossGradientLoss(nn.Module):
    """
    Gallardo & Meju (2003, 2004) Structural Cross-Gradient Constraint.
    t(m, z) = (dm/dx * dz/dy - dm/dy * dz/dx)
    Enforces structural boundary alignment between predicted magnetics and bathymetry.
    Uses proper Sobel-style central difference kernels (registered buffers).
    """
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_x', kx)
        self.register_buffer('kernel_y', ky)

    def forward(self, m, z):
        dm_dx = F.conv2d(F.pad(m, (1, 1, 0, 0), mode='replicate'), self.kernel_x)
        dm_dy = F.conv2d(F.pad(m, (0, 0, 1, 1), mode='replicate'), self.kernel_y)
        dz_dx = F.conv2d(F.pad(z, (1, 1, 0, 0), mode='replicate'), self.kernel_x)
        dz_dy = F.conv2d(F.pad(z, (0, 0, 1, 1), mode='replicate'), self.kernel_y)
        cross_grad = dm_dx * dz_dy - dm_dy * dz_dx
        return torch.mean(torch.abs(cross_grad))


class SpectralLaplaceLoss(nn.Module):
    """
    FIXED: Potential Field Spectral Laplace Loss with DYNAMIC wavenumber grid.
    Previously hardcoded to patch_size=64 at __init__ time — now computes
    the wavenumber grid dynamically inside forward() for any input size.

    Physics: magnetic potential fields V_m satisfy Laplace's equation:
        nabla^2 V_m = 0
    In 2D wavenumber domain: k = sqrt(kx^2 + ky^2)
    High-frequency wavenumbers should decay naturally (upward continuation).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_m, target_m):
        B, C, H, W = pred_m.shape
        # Dynamically compute wavenumber grid for actual input spatial dimensions
        kx = torch.fft.fftfreq(H, device=pred_m.device)
        ky = torch.fft.rfftfreq(W, device=pred_m.device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        K = torch.sqrt(KX ** 2 + KY ** 2)
        # Weight: penalise high-freq amplitude mismatches more (Laplace harmonic decay)
        wn_weight = (1.0 + 2.0 * K).unsqueeze(0).unsqueeze(0)  # [1,1,H,W//2+1]

        fft_pred   = torch.fft.rfft2(pred_m,   norm='ortho')
        fft_target = torch.fft.rfft2(target_m, norm='ortho')

        amp_pred   = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)

        spec_diff = torch.abs(amp_pred - amp_target) * wn_weight
        return torch.mean(spec_diff)


class TotalVariationLoss(nn.Module):
    """Suppresses high-frequency gridding / ringing artifacts"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return torch.mean(diff_h) + torch.mean(diff_w)


class SSIMLoss(nn.Module):
    """
    NEW: Structural Similarity Index (SSIM) loss.
    Preserves ridge and fracture zone morphology that Charbonnier loss alone can blur.
    Returns 1 - SSIM(pred, target) so minimising this maximises structural similarity.
    """
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        # Gaussian window
        sigma = 1.5
        gauss = torch.Tensor(
            [torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
             for x in range(window_size)]
        )
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window_2d)

    def forward(self, pred, target):
        C = pred.shape[1]
        window = self.window.expand(C, 1, -1, -1)
        pad = self.window_size // 2

        mu1 = F.conv2d(pred,   window, padding=pad, groups=C)
        mu2 = F.conv2d(target, window, padding=pad, groups=C)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred   * pred,   window, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
        sigma12   = F.conv2d(pred   * target, window, padding=pad, groups=C) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        return 1.0 - ssim_map.mean()


class AdvancedGeophysicalPhysicsLoss(nn.Module):
    """
    Composite Physics-Informed Objective (UPGRADED):
        L_total = L_recon + λ_cg*L_cg + λ_spec*L_laplace + λ_tv*L_tv + λ_ssim*L_ssim

    Upgrades:
    - SpectralLaplaceLoss is now dynamic (any patch size)
    - Added SSIMLoss for structural ridge/fracture zone preservation
    """
    def __init__(self, patch_size=64, lambda_cg=0.08, lambda_spec=0.04,
                 lambda_tv=0.002, lambda_ssim=0.1):
        super().__init__()
        self.recon_loss   = CharbonnierLoss()
        self.cg_loss      = CrossGradientLoss()
        self.spectral_loss = SpectralLaplaceLoss()   # Dynamic — no patch_size needed
        self.tv_loss      = TotalVariationLoss()
        self.ssim_loss    = SSIMLoss()
        self.lambda_cg    = lambda_cg
        self.lambda_spec  = lambda_spec
        self.lambda_tv    = lambda_tv
        self.lambda_ssim  = lambda_ssim

    def forward(self, pred_m, target_m, guide_z):
        l_recon  = self.recon_loss(pred_m, target_m)
        l_cg     = self.cg_loss(pred_m, guide_z)
        l_spec   = self.spectral_loss(pred_m, target_m)
        l_tv     = self.tv_loss(pred_m)
        l_ssim   = self.ssim_loss(pred_m, target_m)

        l_total  = (l_recon
                    + self.lambda_cg   * l_cg
                    + self.lambda_spec * l_spec
                    + self.lambda_tv   * l_tv
                    + self.lambda_ssim * l_ssim)

        return l_total, l_recon, l_cg, l_spec, l_tv, l_ssim
