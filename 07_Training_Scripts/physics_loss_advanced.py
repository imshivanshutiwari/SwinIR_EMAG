"""
UPGRADED Physics Loss for 4x SR.

Fixes Applied:
  FIX 4: Increased physics loss weights — spec=0.20, cg=0.15
          Previously spec=0.04, cg=0.08 — too weak to drive feature synthesis
  FIX 2: Dynamic wavenumber grid for any HR patch size (256x256 supported)
  All inputs forced to float32 before FFT (prevents ComplexHalf on GPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        x, y = x.float(), y.float()
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps))


class CrossGradientLoss(nn.Module):
    """Gallardo & Meju (2003) structural cross-gradient constraint."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_x', kx)
        self.register_buffer('kernel_y', ky)

    def forward(self, m, z):
        m, z = m.float(), z.float()
        dm_dx = F.conv2d(F.pad(m, (1, 1, 0, 0), mode='replicate'), self.kernel_x)
        dm_dy = F.conv2d(F.pad(m, (0, 0, 1, 1), mode='replicate'), self.kernel_y)
        dz_dx = F.conv2d(F.pad(z, (1, 1, 0, 0), mode='replicate'), self.kernel_x)
        dz_dy = F.conv2d(F.pad(z, (0, 0, 1, 1), mode='replicate'), self.kernel_y)
        cross_grad = dm_dx * dz_dy - dm_dy * dz_dx
        return torch.mean(torch.abs(cross_grad))


class SpectralLaplaceLoss(nn.Module):
    """
    Dynamic 2D Spectral Laplace Loss — works for any patch size.
    FIX: Forced float32 to prevent ComplexHalf overflow on Kaggle GPU AMP.
    FIX 4: Higher spectral weight exposes model to low-frequency wavenumber decay.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_m, target_m):
        pred_m   = pred_m.float()
        target_m = target_m.float()
        B, C, H, W = pred_m.shape
        kx = torch.fft.fftfreq(H, device=pred_m.device)
        ky = torch.fft.rfftfreq(W, device=pred_m.device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        K = torch.sqrt(KX ** 2 + KY ** 2)
        wn_weight = (1.0 + 2.0 * K).unsqueeze(0).unsqueeze(0)

        fft_pred   = torch.fft.rfft2(pred_m,   norm='ortho')
        fft_target = torch.fft.rfft2(target_m, norm='ortho')

        amp_pred   = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)

        spec_diff = torch.abs(amp_pred - amp_target) * wn_weight
        return torch.mean(spec_diff)


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.float()
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return torch.mean(diff_h) + torch.mean(diff_w)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        sigma = 1.5
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        w1d = gauss.unsqueeze(1)
        w2d = w1d.mm(w1d.t()).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', w2d)

    def forward(self, pred, target):
        pred, target = pred.float(), target.float()
        C = pred.shape[1]
        window = self.window.expand(C, 1, -1, -1)
        pad = self.window_size // 2

        mu1 = F.conv2d(pred,   window, padding=pad, groups=C)
        mu2 = F.conv2d(target, window, padding=pad, groups=C)

        mu1_sq  = mu1 ** 2; mu2_sq  = mu2 ** 2; mu1_mu2 = mu1 * mu2
        s1_sq   = F.conv2d(pred   * pred,   window, padding=pad, groups=C) - mu1_sq
        s2_sq   = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
        s12     = F.conv2d(pred   * target, window, padding=pad, groups=C) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * s12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (s1_sq + s2_sq + self.C2))
        return 1.0 - ssim_map.mean()


class AdvancedGeophysicalPhysicsLoss(nn.Module):
    """
    Composite Physics-Informed Objective for 4x Super-Resolution.

    FIX 4: Upgraded weights:
        lambda_cg   = 0.15  (was 0.08) — stronger boundary alignment drive
        lambda_spec = 0.20  (was 0.04) — dominant physics term for potential-field synthesis
        lambda_tv   = 0.002 (unchanged)
        lambda_ssim = 0.10  (unchanged)
    """
    def __init__(self, patch_size=256, lambda_cg=0.15, lambda_spec=0.20,
                 lambda_tv=0.002, lambda_ssim=0.10):
        super().__init__()
        self.recon_loss    = CharbonnierLoss()
        self.cg_loss       = CrossGradientLoss()
        self.spectral_loss = SpectralLaplaceLoss()
        self.tv_loss       = TotalVariationLoss()
        self.ssim_loss     = SSIMLoss()
        self.lambda_cg     = lambda_cg
        self.lambda_spec   = lambda_spec
        self.lambda_tv     = lambda_tv
        self.lambda_ssim   = lambda_ssim

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
