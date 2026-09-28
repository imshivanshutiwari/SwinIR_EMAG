# ==============================================================================
# SwinIR: Image Restoration Using Swin Transformer (Adapted for Geomagnetic SR)
# Stage 1: EMAG2 (32x32, 1-ch) -> Australia 4km (128x128, 1-ch) [4x Super-Resolution]
# ==============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) with relative position bias. """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Attention mask for shifted window
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualGroup(nn.Module):
    """ Residual Swin Transformer Block (RSTB) """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        res = x
        for blk in self.blocks:
            res = blk(res, x_size)
        
        B, L, C = res.shape
        H, W = x_size
        res = res.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        res = self.conv(res)
        res = res.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        return x + res


class Upsample(nn.Module):
    """
    Anti-Aliased Upsample module for geomagnetic potential field super-resolution.

    scale=4: Two 2x bilinear+conv stages (no PixelShuffle artifacts)
    scale=2: Single bilinear+conv stage (NO PixelShuffle — eliminates checkerboard)

    FIX: PixelShuffle(2) caused 10x Nyquist-corner energy spikes and visible
    ±2.14 nT checkerboard artifacts on zero-input response. Replaced with
    Bilinear interpolation + Conv2d which is physically correct for smooth
    potential field upsampling (band-limited, no phase discontinuities).
    """
    def __init__(self, scale, num_feat):
        super().__init__()
        self.scale = scale
        if scale == 4:
            # Two 2x stages: bilinear upsample → refinement conv → GELU
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.GELU()
            )
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.GELU()
            )
        elif scale == 2:
            # Single 2x stage: bilinear upsample → refinement conv → GELU
            # NO PixelShuffle → no checkerboard, no Nyquist resonance
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.GELU()
            )
            self.up2 = None
        else:
            raise ValueError(f"Upsample scale {scale} not supported. Use 2 or 4.")

    def forward(self, x):
        x = self.up1(x)
        if self.up2 is not None:
            x = self.up2(x)
        return x


class SwinIR(nn.Module):
    """
    SwinIR super-resolution network for 1-channel geomagnetic anomaly fields.
    """
    def __init__(self, img_size=32, in_chans=1, embed_dim=60, depths=[4, 4, 4, 4],
                 num_heads=[6, 6, 6, 6], window_size=8, mlp_ratio=2.,
                 upscale=4, resi_connection='1conv', upsample_mode='bilinear'):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.upscale = upscale
        self.upsample_mode = upsample_mode
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        # 1. Shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction (Residual Swin Transformer Blocks)
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=self.img_size,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 3. HQ reconstruction / Upsampler
        if upsample_mode == 'pixelshuffle':
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, (upscale ** 2) * embed_dim, 3, 1, 1),
                nn.PixelShuffle(upscale)
            )
        else:
            self.upsample = Upsample(upscale, embed_dim)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 1, H, W) e.g., (B, 1, 32, 32)
        H, W = x.shape[2], x.shape[3]
        
        # Shallow features
        x_first = self.conv_first(x)
        
        # Deep features
        feat = x_first.flatten(2).transpose(1, 2)  # B, H*W, C
        for layer in self.layers:
            feat = layer(feat, (H, W))
        feat = self.norm(feat)
        feat = feat.view(-1, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        
        # Residual connection
        feat = self.conv_after_body(feat) + x_first
        
        # Upsampling & Reconstruction
        out = self.upsample(feat)
        out = self.conv_last(out)
        
        # v3: Restore bicubic skip with DC-offset correction.
        # The satellite input has ~+180 nT DC offset while targets are mean-zero.
        # v1 had raw bicubic skip → model crushed amplitude to cancel offset.
        # v2 removed skip entirely → lost residual learning, structural collapse.
        # v3: Zero-center the skip (remove DC, keep spatial variation).
        res_input = F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        res_input = res_input - res_input.mean(dim=[2, 3], keepdim=True)  # per-sample DC removal
        out = out + res_input
        
        return out


def create_stage1_swinir(upscale=4, embed_dim=60, depths=[4, 4, 4, 4], num_heads=[6, 6, 6, 6]):
    """ Factory for Stage 1 SwinIR (EMAG2 32x32 -> AU 4km 128x128) """
    return SwinIR(
        img_size=32,
        in_chans=1,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=8,
        mlp_ratio=2.0,
        upscale=upscale
    )


def create_stage2_swinir(upscale=2, embed_dim=96, depths=[4, 4, 4, 4], num_heads=[6, 6, 6, 6], upsample_mode='bilinear'):
    """ Factory for Stage 2 / Level 2 / Level 3 SwinIR """
    return SwinIR(
        img_size=64,
        in_chans=1,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=8,
        mlp_ratio=2.0,
        upscale=upscale,
        upsample_mode=upsample_mode
    )


def create_stage2_swinir_pixelshuffle(upscale=2, embed_dim=96, depths=[4, 4, 4, 4], num_heads=[6, 6, 6, 6]):
    """ Factory for Level 1 & Level 2 SwinIR trained with PixelShuffle """
    return create_stage2_swinir(
        upscale=upscale,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        upsample_mode='pixelshuffle'
    )


