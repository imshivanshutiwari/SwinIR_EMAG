"""
Fixed Physics Loss Functions for Ultra-Sharp 4x Model:
1. Anti-Quantization Filtered SobelEdgeLoss: Smooths 1-pixel raster quantization steps before gradient comparison.
2. Balanced Loss Weights: Prevents staircase amplification while recovering sharp lineations.
3. High-Wavenumber Laplace Spectral Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiQuantizedSobelEdgeLoss(nn.Module):
    """
    Computes Sobel gradient magnitude loss after anti-aliasing filter to prevent
    fitting to discrete integer raster quantization stair-steps.
    """
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_x', kx)
        self.register_buffer('kernel_y', ky)

        # 3x3 Gaussian smoothing kernel to remove integer-step artifacts
        g = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32) / 16.0
        self.register_buffer('gauss_kernel', g.unsqueeze(0).unsqueeze(0))

    def forward(self, pred, target):
        pred, target = pred.float(), target.float()

        # Step 1: Filter out 1-pixel integer quantization noise
        pred_smooth   = F.conv2d(F.pad(pred,   (1, 1, 1, 1), mode='replicate'), self.gauss_kernel)
        target_smooth = F.conv2d(F.pad(target, (1, 1, 1, 1), mode='replicate'), self.gauss_kernel)

        # Step 2: Compute true geological gradients
        pred_gx   = F.conv2d(F.pad(pred_smooth,   (1, 1, 1, 1), mode='replicate'), self.kernel_x)
        pred_gy   = F.conv2d(F.pad(pred_smooth,   (1, 1, 1, 1), mode='replicate'), self.kernel_y)
        target_gx = F.conv2d(F.pad(target_smooth, (1, 1, 1, 1), mode='replicate'), self.kernel_x)
        target_gy = F.conv2d(F.pad(target_smooth, (1, 1, 1, 1), mode='replicate'), self.kernel_y)

        pred_mag   = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-6)
        target_mag = torch.sqrt(target_gx**2 + target_gy**2 + 1e-6)

        return F.l1_loss(pred_mag, target_mag)


class CrossGradientLoss(nn.Module):
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


class HighWavenumberBoostedSpectralLoss(nn.Module):
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


class SharpPhysicsLoss(nn.Module):
    """
    Rebalanced Composite Loss:
      lambda_recon = 1.00 (L1 Charbonnier base)
      lambda_edge  = 0.08 (Anti-quantized geological edge loss)
      lambda_cg    = 0.20 (Cross-gradient with bathymetry)
      lambda_spec  = 0.20 (Harmonic Laplace decay)
    """
    def __init__(self, lambda_edge=0.08, lambda_cg=0.20, lambda_spec=0.20):
        super().__init__()
        self.recon_loss = nn.L1Loss()
        self.edge_loss  = AntiQuantizedSobelEdgeLoss()
        self.cg_loss    = CrossGradientLoss()
        self.spec_loss  = HighWavenumberBoostedSpectralLoss()

        self.lambda_edge = lambda_edge
        self.lambda_cg   = lambda_cg
        self.lambda_spec = lambda_spec

    def forward(self, pred_m, target_m, guide_z):
        l_recon = self.recon_loss(pred_m, target_m)
        l_edge  = self.edge_loss(pred_m, target_m)
        l_cg    = self.cg_loss(pred_m, guide_z)
        l_spec  = self.spec_loss(pred_m, target_m)

        l_total = (l_recon
                   + self.lambda_edge * l_edge
                   + self.lambda_cg   * l_cg
                   + self.lambda_spec * l_spec)

        return l_total, l_recon, l_edge, l_cg, l_spec
