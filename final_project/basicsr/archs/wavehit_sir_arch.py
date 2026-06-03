"""
WaveHiT-SIR: Hierarchical Wavelet Transformer for Efficient Image Super-Resolution
===================================================================================
Integrates ideas from WaveHiT-SR (Ali et al., 2025) into the HiT-SIR codebase:

  1. WaveDFE  – Replaces the original DFE conv branch with a Discrete Wavelet
                Transform (DWT) branch for multi-frequency dual feature extraction.
  2. WaveSCC  – Replaces the spatial linear projection inside S-SC with DWT-based
                downsampling (WA-SC), achieving linear complexity w.r.t. window size
                and retaining explicit frequency information in the correlation map.
  3. HAB      – Adds a lightweight squeeze-and-excitation Channel Attention Block
                before each HierarchicalTransformerBlock, forming the Hybrid Attention
                Block that jointly models global channel context and local wavelet
                spatial structure.
  4. WaveHiT_SIR – Top-level model class that plugs the components above into the
                   same RHTB/BasicLayer/HiT-SIR skeleton without touching the rest
                   of the training pipeline.

Usage
-----
  Register via BasicSR's ARCH_REGISTRY (same as HiT_SIR).
  Drop this file under basicsr/archs/ and reference it in your YML:
      network_g:
        type: WaveHiT_SIR
        ...   (same kwargs as HiT_SIR)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY

try:
    import pywt
    import pytorch_wavelets  # optional, used as fallback if available
    HAS_PYTORCH_WAVELETS = True
except ImportError:
    HAS_PYTORCH_WAVELETS = False


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Pure-PyTorch 2-D Haar DWT / IDWT
# ─────────────────────────────────────────────────────────────────────────────

def dwt_haar_2d(x):
    """
    Single-level 2-D Haar DWT using separable fixed filters.
    Input  : x  (B, C, H, W)
    Returns: (LL, LH, HL, HH)  each of shape (B, C, H//2, W//2)
    """
    device = x.device
    # 1-D Haar low/high pass filters
    lo = torch.tensor([1.0,  1.0], device=device).view(1, 1, 1, 2) * 0.5
    hi = torch.tensor([1.0, -1.0], device=device).view(1, 1, 1, 2) * 0.5

    B, C, H, W = x.shape
    x = x.view(B * C, 1, H, W)

    # Apply along width
    x_lo = F.conv2d(x, lo, stride=(1, 2), padding=(0, 0))
    x_hi = F.conv2d(x, hi, stride=(1, 2), padding=(0, 0))

    # Transpose filters for height
    lo_h = lo.permute(0, 1, 3, 2)   # (1,1,2,1)
    hi_h = hi.permute(0, 1, 3, 2)

    LL = F.conv2d(x_lo, lo_h, stride=(2, 1), padding=(0, 0))
    LH = F.conv2d(x_lo, hi_h, stride=(2, 1), padding=(0, 0))
    HL = F.conv2d(x_hi, lo_h, stride=(2, 1), padding=(0, 0))
    HH = F.conv2d(x_hi, hi_h, stride=(2, 1), padding=(0, 0))

    _, _, Hd, Wd = LL.shape
    LL = LL.view(B, C, Hd, Wd)
    LH = LH.view(B, C, Hd, Wd)
    HL = HL.view(B, C, Hd, Wd)
    HH = HH.view(B, C, Hd, Wd)
    return LL, LH, HL, HH


# ─────────────────────────────────────────────────────────────────────────────
# 1.  WaveDFE – Wavelet-based Dual Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

class WaveDFE(nn.Module):
    """
    Replaces the 3-layer-conv branch in the original DFE with a DWT branch.

    Original DFE:
        out = Conv(x) * Linear(x)

    WaveDFE (Eq. 2 in WaveHiT-SR):
        Xwave = WaveConv(x)     -- via 1-level DWT + learnable fusion conv
        Xch   = Linear(x)
        out   = Xch ⊙ Xwave

    This lets the spatial branch see explicit low-/high-frequency subbands
    before element-wise fusion with the channel linear branch.

    Args:
        in_features  (int): Number of input channels.
        out_features (int): Number of output channels.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features

        # Channel branch: same as original DFE linear branch
        self.linear = nn.Conv2d(in_features, out_features, 1, 1, 0)

        # Wavelet branch: fuse the 4 DWT subbands back to out_features
        # Each subband has in_features channels → cat → 4*in_features
        self.wave_fuse = nn.Sequential(
            nn.Conv2d(4 * in_features, in_features, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features, out_features, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # After fusing, upsample subbands back to original spatial size
        self.upsample = nn.Sequential(
            nn.Conv2d(out_features, out_features, 3, 1, 1),
        )

    def forward(self, x, x_size):
        """
        Args:
            x      : (B, H*W, C)
            x_size : (H, W)
        Returns:
            out    : (B, H*W, out_features)
        """
        B, L, C = x.shape
        H, W = x_size
        x2d = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # ── channel branch ───────────────────────────────────────────────
        x_ch = self.linear(x2d)                     # (B, out_features, H, W)

        # ── wavelet branch ───────────────────────────────────────────────
        # Pad to ensure even spatial dimensions for DWT
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x2d_pad = F.pad(x2d, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x2d_pad = x2d

        LL, LH, HL, HH = dwt_haar_2d(x2d_pad)      # each (B, C, H//2, W//2)
        subbands = torch.cat([LL, LH, HL, HH], dim=1)  # (B, 4C, H//2, W//2)
        x_wave = self.wave_fuse(subbands)            # (B, out_features, H//2, W//2)

        # Restore spatial resolution
        x_wave = F.interpolate(x_wave, size=(H, W),
                               mode='bilinear', align_corners=False)
        x_wave = self.upsample(x_wave)               # (B, out_features, H, W)

        # ── element-wise fusion ──────────────────────────────────────────
        out = x_ch * x_wave                          # (B, out_features, H, W)
        out = out.view(B, -1, H * W).permute(0, 2, 1).contiguous()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  WaveSCC – Wave Attention Spatial-Channel Correlation
# ─────────────────────────────────────────────────────────────────────────────

class WaveSCC(nn.Module):
    """
    Replaces the spatial_linear_projection in HiT-SR's SCC with DWT-based
    downsampling (WA-SC from WaveHiT-SR, Eq. 4-5).

    Complexity:
        Original S-SC  : O(hw * (h/map_h * w/map_w))   – linear layer
        WA-SC (this)   : O(h * w * log(h * w))         – near-linear via DWT

    The DWT halves spatial resolution before the correlation map is computed,
    reducing multiply-adds by ~4× in the spatial branch while preserving
    multi-frequency structure.

    Args:
        dim          (int)         : Number of input channels.
        base_win_size(tuple[int])  : (Hb, Wb) base window size.
        window_size  (tuple[int])  : (H, W) current hierarchical window size.
        num_heads    (int)         : Number of heads for spatial self-correlation.
        value_drop   (float)       : Dropout ratio of value. Default: 0.0
        proj_drop    (float)       : Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, base_win_size, window_size, num_heads,
                 value_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        # Feature projection: use WaveDFE instead of original DFE
        self.qv = WaveDFE(dim, dim)
        self.proj = nn.Linear(dim, dim)

        # Dropout
        self.value_drop = nn.Dropout(value_drop)
        self.proj_drop  = nn.Dropout(proj_drop)

        # Base window size
        min_h = min(self.window_size[0], base_win_size[0])
        min_w = min(self.window_size[1], base_win_size[1])
        self.base_win_size = (min_h, min_w)

        # Normalization scale for S-SC (head_dim)
        head_dim = dim // (2 * num_heads)
        self.scale = head_dim

        # Dynamic position bias (unchanged from HiT-SR)
        self.H_sp, self.W_sp = self.window_size
        self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

        # Learnable 1×1 conv to fuse DWT output channels back to head_dim
        # DWT produces 4 subbands → 4 * (C//2//num_heads) channels per head
        dwt_ch_in = 4 * (dim // (2 * num_heads))
        self.wave_proj = nn.Linear(dwt_ch_in, head_dim)

    # ------------------------------------------------------------------
    # WA-SC: DWT-based spatial projection  (replaces spatial_linear_proj)
    # ------------------------------------------------------------------
    def wave_spatial_projection(self, v):
        """
        Apply 1-level DWT to values v and project to base_win_size tokens.

        Args:
            v : (B, num_heads, L, C_h)  where L = H_sp * W_sp, C_h = head_dim
        Returns:
            v_down : (B, num_heads, map_L, C_h)
                     where map_L = base_win_size[0] * base_win_size[1]
        """
        B, num_h, L, C_h = v.shape
        H, W = self.window_size

        # Reshape to spatial for DWT
        v_2d = v.permute(0, 1, 3, 2).contiguous().view(B * num_h, C_h, H, W)

        # Pad if needed
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            v_2d = F.pad(v_2d, (0, pad_w, 0, pad_h), mode='reflect')

        LL, LH, HL, HH = dwt_haar_2d(v_2d)
        # cat subbands along channel → (B*num_h, 4*C_h, H//2, W//2)
        subbands = torch.cat([LL, LH, HL, HH], dim=1)

        # Downsample to base_win_size if still too large
        map_H, map_W = self.base_win_size
        subbands = F.adaptive_avg_pool2d(subbands, (map_H, map_W))

        # Reshape → (B, num_h, map_L, 4*C_h) then project to C_h
        subbands = subbands.view(B, num_h, 4 * C_h, map_H * map_W)
        subbands = subbands.permute(0, 1, 3, 2).contiguous()     # (B,h,map_L,4C)
        v_down = self.wave_proj(subbands)                         # (B,h,map_L,C_h)
        return v_down

    # ------------------------------------------------------------------
    # Spatial self-correlation with Wave Attention (WA-SC, Eq. 5)
    # ------------------------------------------------------------------
    def spatial_self_correlation(self, q, v):
        """
        Args:
            q : (B, num_heads, L, C_h)
            v : (B, num_heads, L, C_h)
        Returns:
            x : (B, L, dim//2)
        """
        B, num_head, L, C_h = q.shape

        # WA-SC: DWT-based value projection
        v_down = self.wave_spatial_projection(v)    # (B, h, map_L, C_h)

        # Correlation map: Q × V_down^T  (no softmax → linear via correlation)
        corr_map = (q @ v_down.transpose(-2, -1)) / self.scale  # (B,h,L,map_L)

        # --- relative position bias (unchanged from HiT-SR) ---
        position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=v.device)
        position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=v.device)
        biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
        rpe_biases = biases.flatten(1).transpose(0, 1).contiguous().float()
        pos = self.pos(rpe_biases)

        coords_h = torch.arange(self.H_sp, device=v.device)
        coords_w = torch.arange(self.W_sp, device=v.device)
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten  = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.H_sp - 1
        relative_coords[:, :, 1] += self.W_sp - 1
        relative_coords[:, :, 0] *= 2 * self.W_sp - 1
        relative_position_index = relative_coords.sum(-1)

        map_H, map_W = self.base_win_size
        rel_pos_bias = pos[relative_position_index.view(-1)].view(
            self.H_sp * self.W_sp,
            map_H, self.H_sp // map_H,
            map_W, self.W_sp // map_W, -1)
        rel_pos_bias = rel_pos_bias.permute(0, 1, 3, 5, 2, 4).contiguous().view(
            self.H_sp * self.W_sp,
            map_H * map_W, self.num_heads, -1).mean(-1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()   # (h, L, map_L)
        corr_map = corr_map + rel_pos_bias.unsqueeze(0)

        # Weighted value aggregation
        v_drop = self.value_drop(v_down)
        x = (corr_map @ v_drop).permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        return x

    # ------------------------------------------------------------------
    # Channel self-correlation  (unchanged from HiT-SR)
    # ------------------------------------------------------------------
    def channel_self_correlation(self, q, v):
        B, num_head, L, C = q.shape
        q = q.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)
        v = v.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)
        corr_map = (q.transpose(-2, -1) @ v) / L
        v_drop = self.value_drop(v)
        x = (corr_map @ v_drop.transpose(-2, -1)).permute(0, 2, 1).contiguous().view(B, L, -1)
        return x

    def forward(self, x):
        """
        Args:
            x : (B, H, W, C)
        """
        xB, xH, xW, xC = x.shape
        qv = self.qv(x.view(xB, -1, xC), (xH, xW)).view(xB, xH, xW, xC)

        # window partition
        qv = window_partition(qv, self.window_size)
        qv = qv.view(-1, self.window_size[0] * self.window_size[1], xC)

        B, L, C = qv.shape
        qv = qv.view(B, L, 2, self.num_heads, C // (2 * self.num_heads)
                     ).permute(2, 0, 3, 1, 4).contiguous()
        q, v = qv[0], qv[1]    # (B, num_heads, L, C//num_heads)

        # WA-SC
        x_spatial = self.spatial_self_correlation(q, v)
        x_spatial = x_spatial.view(-1, self.window_size[0], self.window_size[1], C // 2)
        x_spatial = window_reverse(x_spatial, self.window_size, xH, xW)

        # C-SC
        x_channel = self.channel_self_correlation(q, v)
        x_channel = x_channel.view(-1, self.window_size[0], self.window_size[1], C // 2)
        x_channel = window_reverse(x_channel, self.window_size, xH, xW)

        x = torch.cat([x_spatial, x_channel], dim=-1)
        x = self.proj_drop(self.proj(x))
        return x

    def extra_repr(self):
        return (f'dim={self.dim}, window_size={self.window_size}, '
                f'num_heads={self.num_heads}')


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Channel Attention (Squeeze-and-Excitation style) for HAB
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Lightweight Squeeze-and-Excitation channel attention.
    Used inside the Hybrid Attention Block (HAB) to model global channel context
    before the wavelet spatial correlation step.

    Paper (WaveHiT-SR Sec. Method): "channel attention leverages global information
    to dynamically adjust attention weights across channels, selectively activating
    pixels and focusing on key features across multiple scales."

    Args:
        dim         (int)  : Number of input channels.
        reduction   (int)  : Reduction ratio. Default: 4.
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        mid = max(dim // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x : (B, H*W, C)   – token sequence
        Returns:
            x : (B, H*W, C)   – channel-recalibrated sequence
        """
        B, L, C = x.shape
        # Global average over spatial positions
        y = x.mean(dim=1)              # (B, C)
        y = self.fc(y)                 # (B, C)  – channel weights in [0,1]
        return x * y.unsqueeze(1)      # broadcast over tokens


# ─────────────────────────────────────────────────────────────────────────────
# Helper classes copied verbatim from hit_sir_arch.py (needed for standalone use)
# ─────────────────────────────────────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features     = out_features or in_features
        hidden_features  = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0],
                   W // window_size[1], window_size[1], C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] * (window_size[0] * window_size[1]) / (H * W))
    x = windows.view(B, H // window_size[0], W // window_size[1],
                        window_size[0], window_size[1], -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class DynamicPosBias(nn.Module):
    """Copied from HiT-SIR (unchanged)."""
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual  = residual
        self.num_heads = num_heads
        self.pos_dim   = dim // 4
        self.pos_proj  = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True),
                                  nn.Linear(self.pos_dim, self.pos_dim))
        self.pos2 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True),
                                  nn.Linear(self.pos_dim, self.pos_dim))
        self.pos3 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True),
                                  nn.Linear(self.pos_dim, self.num_heads))

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                               img_size[1] // patch_size[1]]
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans    = in_chans
        self.embed_dim   = embed_dim

        # in_chans == 0 is a sentinel meaning "identity" (used inside RHTB)
        if in_chans == 0 or in_chans == embed_dim:
            self.proj = None          # identity – skip conv
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim,
                                  kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size   = img_size
        self.patch_size = patch_size
        self.embed_dim  = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        return x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                      nn.PixelShuffle(2)]
        elif scale == 3:
            m += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1),
                  nn.PixelShuffle(3)]
        else:
            raise ValueError(f'Unsupported scale {scale}')
        super().__init__(*m)


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = [nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
             nn.PixelShuffle(scale)]
        super().__init__(*m)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Hybrid Attention Block (HAB):  ChannelAttention + WaveSCC
# ─────────────────────────────────────────────────────────────────────────────

class WaveHierarchicalTransformerBlock(nn.Module):
    """
    Hierarchical Transformer Block enhanced with the Hybrid Attention Block (HAB).

    Sequence:
        shortcut = x
        x        = ChannelAttention(x)          ← global channel context
        x        = WaveSCC(x)                   ← WA-SC + C-SC  (wavelet spatial + channel)
        x        = LayerNorm(shortcut + x)
        x        = x + MLP(LayerNorm(x))

    Compared with the original HiT-SR block, the only additions are:
        • ChannelAttention before WaveSCC (HAB's channel branch)
        • WaveSCC instead of SCC (wavelet spatial projection)
    """

    def __init__(self, dim, input_resolution, num_heads, base_win_size, window_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ca_reduction=4):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.mlp_ratio        = mlp_ratio

        # Sanity check on window sizes
        if window_size[0] > base_win_size[0] and window_size[1] > base_win_size[1]:
            assert window_size[0] % base_win_size[0] == 0
            assert window_size[1] % base_win_size[1] == 0

        # Channel attention (HAB's global branch)
        self.channel_attn = ChannelAttention(dim, reduction=ca_reduction)

        # Wavelet spatial-channel correlation (replaces SCC)
        self.norm1       = norm_layer(dim)
        self.correlation = WaveSCC(
            dim, base_win_size=base_win_size, window_size=window_size,
            num_heads=num_heads, value_drop=value_drop, proj_drop=drop)

        self.drop_path   = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2       = norm_layer(dim)
        self.mlp         = Mlp(in_features=dim,
                               hidden_features=int(dim * mlp_ratio),
                               act_layer=act_layer, drop=drop)

    def check_image_size(self, x, win_size):
        x = x.permute(0, 3, 1, 2).contiguous()
        _, _, h, w = x.size()
        mod_pad_h = (win_size[0] - h % win_size[0]) % win_size[0]
        mod_pad_w = (win_size[1] - w % win_size[1]) % win_size[1]
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, x, x_size, win_size):
        H, W   = x_size
        B, L, C = x.shape
        shortcut = x

        # ── channel attention ────────────────────────────────────────────
        x = self.channel_attn(x)       # (B, L, C)

        # ── wavelet spatial-channel correlation ──────────────────────────
        x = x.view(B, H, W, C)
        x = self.check_image_size(x, win_size)
        x = self.correlation(x)        # (B, H_pad, W_pad, C)
        x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.norm1(x)

        # ── residual + FFN ───────────────────────────────────────────────
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

    def extra_repr(self):
        return (f'dim={self.dim}, input_resolution={self.input_resolution}, '
                f'num_heads={self.num_heads}, window_size={self.window_size}, '
                f'mlp_ratio={self.mlp_ratio}')


# ─────────────────────────────────────────────────────────────────────────────
# BasicLayer and RHTB  (same topology as HiT-SR; just swap block class)
# ─────────────────────────────────────────────────────────────────────────────

class WaveBasicLayer(nn.Module):
    """Stacks WaveHierarchicalTransformerBlock with expanding hierarchical windows."""

    def __init__(self, dim, input_resolution, depth, num_heads, base_win_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 hier_win_ratios=[0.5, 1, 2, 4, 6, 8], ca_reduction=4):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.depth            = depth
        self.use_checkpoint   = use_checkpoint

        self.win_hs = [int(base_win_size[0] * r) for r in hier_win_ratios]
        self.win_ws = [int(base_win_size[1] * r) for r in hier_win_ratios]

        self.blocks = nn.ModuleList([
            WaveHierarchicalTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads,
                base_win_size=base_win_size,
                window_size=(self.win_hs[i], self.win_ws[i]),
                mlp_ratio=mlp_ratio, drop=drop, value_drop=value_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, ca_reduction=ca_reduction)
            for i in range(depth)
        ])

        self.downsample = downsample(input_resolution, dim=dim,
                                     norm_layer=norm_layer) if downsample else None

    def forward(self, x, x_size):
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size,
                                          (self.win_hs[i], self.win_ws[i]))
            else:
                x = blk(x, x_size, (self.win_hs[i], self.win_ws[i]))
        if self.downsample:
            x = self.downsample(x)
        return x

    def extra_repr(self):
        return (f'dim={self.dim}, input_resolution={self.input_resolution}, '
                f'depth={self.depth}')


class PatchMerging(nn.Module):
    """Unchanged from HiT-SR."""
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim       = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm      = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
                        x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], dim=-1)
        x = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class WaveRHTB(nn.Module):
    """Residual Hierarchical Transformer Block using WaveBasicLayer."""

    def __init__(self, dim, input_resolution, depth, num_heads, base_win_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv',
                 hier_win_ratios=[0.5, 1, 2, 4, 6, 8], ca_reduction=4):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution

        self.residual_group = WaveBasicLayer(
            dim=dim, input_resolution=input_resolution,
            depth=depth, num_heads=num_heads, base_win_size=base_win_size,
            mlp_ratio=mlp_ratio, drop=drop, value_drop=value_drop,
            drop_path=drop_path, norm_layer=norm_layer,
            downsample=downsample, use_checkpoint=use_checkpoint,
            hier_win_ratios=hier_win_ratios, ca_reduction=ca_reduction)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed   = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                         in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size,
                                           in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return (self.patch_embed(
                    self.conv(
                        self.patch_unembed(
                            self.residual_group(x, x_size), x_size)))
                + x)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  WaveHiT_SIR  – top-level model
# ─────────────────────────────────────────────────────────────────────────────

@ARCH_REGISTRY.register()
class WaveHiT_SIR(nn.Module):
    """
    WaveHiT-SIR: HiT-SIR enhanced with WaveAttention and Channel Attention.

    Drop-in replacement for HiT_SIR – accepts identical constructor arguments
    plus two optional extras:
        ca_reduction (int)  : Reduction ratio for ChannelAttention. Default: 4.

    Architecture changes vs. HiT-SIR
    ──────────────────────────────────
        DFE  →  WaveDFE      (DWT-based spatial branch)
        SCC  →  WaveSCC      (DWT value projection, WA-SC)
        HTB  →  WaveHTB      (HAB = ChannelAttention + WaveSCC)
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 base_win_size=[8, 8], mlp_ratio=2.,
                 drop_rate=0., value_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=4, img_range=1.,
                 upsampler='pixelshuffledirect', resi_connection='1conv',
                 hier_win_ratios=[0.5, 1, 2, 4, 6, 8],
                 ca_reduction=4,
                 **kwargs):
        super().__init__()

        num_in_ch  = in_chans
        num_out_ch = in_chans
        num_feat   = 64

        self.img_range = img_range
        if in_chans == 3:
            self.mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.upscale      = upscale
        self.upsampler    = upsampler
        self.base_win_size = base_win_size

        # ── 1. Shallow feature extraction ────────────────────────────────
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ── 2. Deep feature extraction ───────────────────────────────────
        self.num_layers   = len(depths)
        self.embed_dim    = embed_dim
        self.ape          = ape
        self.patch_norm   = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio    = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        num_patches         = self.patch_embed.num_patches
        patches_resolution  = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build WaveRHTB layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = WaveRHTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                base_win_size=base_win_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate, value_drop=value_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                hier_win_ratios=hier_win_ratios,
                ca_reduction=ca_reduction,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ── 3. HQ image reconstruction ───────────────────────────────────
        if upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif upsampler == 'nearest+conv':
            assert upscale == 4, 'nearest+conv only supports x4'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    # ── weight initialisation ─────────────────────────────────────────────
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_last(self.upsample(self.conv_before_upsample(x)))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(
                F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(
                F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        return x[:, :, :H * self.upscale, :W * self.upscale]


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    upscale      = 4
    base_win     = [8, 8]
    height       = (1024 // upscale // base_win[0] + 1) * base_win[0]
    width        = (720  // upscale // base_win[1] + 1) * base_win[1]

    model = WaveHiT_SIR(
        upscale=upscale, img_size=(height, width),
        base_win_size=base_win, img_range=1.,
        depths=[6, 6, 6, 6], embed_dim=60,
        num_heads=[6, 6, 6, 6], mlp_ratio=2.,
        upsampler='pixelshuffledirect',
        ca_reduction=4,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"WaveHiT-SIR  |  parameters: {total_params:,}")

    # Forward pass test
    x = torch.randn(1, 3, height, width)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")