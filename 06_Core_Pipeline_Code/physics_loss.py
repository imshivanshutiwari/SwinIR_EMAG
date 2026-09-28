"""
Official Physics Loss Module for Multi-Modal Geophysical Super-Resolution.
Implements:
1. Reconstruction Loss: Charbonnier Loss (smooth L1, robust to outliers)
2. Gallardo-Meju Structural Cross-Gradient Loss:
   t(m, z) = | (dm/dx * dz/dy - dm/dy * dz/dx) |
   Enforces structural parallelism/anti-parallelism without forcing positive or negative sign.
3. Total Variation (TV) Loss: suppresses noise artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (a robust L1 loss alternative)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

class CrossGradientLoss(nn.Module):
    """
    Gallardo & Meju (2003, 2004) Structural Cross-Gradient Constraint.
    Evaluated between predicted field (e.g. Magnetics) and reference structural field (e.g. Bathymetry).
    
    Formula:
        t(m, z) = grad(m) x grad(z) = (dm/dx * dz/dy - dm/dy * dz/dx)
    
    The loss minimizes the L1 norm of the cross-gradient:
        L_cg = mean(|t(m, z)|)
    
    This enforces that the directional boundaries of magnetics match the boundaries
    of seafloor bathymetry, regardless of whether the magnetic polarity is positive or negative.
    """
    def __init__(self):
        super(CrossGradientLoss, self).__init__()
        # Central difference kernels: [-0.5, 0, 0.5]
        kx = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_x', kx)
        self.register_buffer('kernel_y', ky)

    def forward(self, m, z):
        # m: [B, 1, H, W] (Magnetics)
        # z: [B, 1, H, W] (Bathymetry or Gravity)
        
        # Central differences with replicate padding
        dm_dx = F.conv2d(F.pad(m, (1, 1, 0, 0), mode='replicate'), self.kernel_x)
        dm_dy = F.conv2d(F.pad(m, (0, 0, 1, 1), mode='replicate'), self.kernel_y)
        
        dz_dx = F.conv2d(F.pad(z, (1, 1, 0, 0), mode='replicate'), self.kernel_x)
        dz_dy = F.conv2d(F.pad(z, (0, 0, 1, 1), mode='replicate'), self.kernel_y)
        
        # 2D cross gradient: dm/dx * dz/dy - dm/dy * dz/dx
        cross_grad = dm_dx * dz_dy - dm_dy * dz_dx
        
        return torch.mean(torch.abs(cross_grad))

class TotalVariationLoss(nn.Module):
    """Suppresses high-frequency gridding / ringing artifacts"""
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return torch.mean(diff_h) + torch.mean(diff_w)

class GeophysicalPhysicsLoss(nn.Module):
    """
    Composite Physics-Informed Objective:
        L_total = L_recon + lambda_cg * L_cross_grad(m, z) + lambda_tv * L_tv(m)
    """
    def __init__(self, lambda_cg=0.05, lambda_tv=0.001):
        super(GeophysicalPhysicsLoss, self).__init__()
        self.recon_loss = CharbonnierLoss()
        self.cg_loss = CrossGradientLoss()
        self.tv_loss = TotalVariationLoss()
        self.lambda_cg = lambda_cg
        self.lambda_tv = lambda_tv

    def forward(self, pred_m, target_m, guide_z):
        l_recon = self.recon_loss(pred_m, target_m)
        l_cg = self.cg_loss(pred_m, guide_z)
        l_tv = self.tv_loss(pred_m)
        l_total = l_recon + self.lambda_cg * l_cg + self.lambda_tv * l_tv
        return l_total, l_recon, l_cg, l_tv
