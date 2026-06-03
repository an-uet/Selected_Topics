"""
WaveDAT: MSW-DAT + WaveDFE (wavelet local branch) + FGCA (freq-guided channel attention)
==========================================================================================
Two targeted changes to MSW-DAT:

  1. WaveDFE replaces DWConv in AIM's local branch:
       DWConv(x) → single-scale local features
       WaveDFE(x) → multi-freq (LL/LH/HL/HH) local features
     This gives the AIM cross-gating richer frequency information.

  2. FGCA added before each spatial attention block:
       SE avg-pool gate → FGCA: LL (low-freq) + HF energy (LH+HL+HH) gate
     Channels specialise in different freq bands — FGCA exploits that.

Warm-start: load from MSW-DAT net_g_475000.pth (strict_load=False).
  ~90% layers match. DWConv→WaveDFE and new FGCA are randomly init.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.dat_arch import (
    Spatial_Attention, Adaptive_Channel_Attention, SGFN,
    DynamicPosBias, img2windows, windows2img, Upsample
)


# ─── Haar DWT (same as wavehit_sir_arch) ─────────────────────────────────────

def _dwt_haar_2d(x):
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    x = torch.clamp(x, -1e4, 1e4)
    device, dtype = x.device, x.dtype
    lo = torch.tensor([1., 1.], device=device, dtype=dtype).view(1,1,1,2) * 0.5
    hi = torch.tensor([1.,-1.], device=device, dtype=dtype).view(1,1,1,2) * 0.5
    B, C, H, W = x.shape
    x = x.reshape(B*C, 1, H, W)
    x_lo = F.conv2d(x, lo, stride=(1,2))
    x_hi = F.conv2d(x, hi, stride=(1,2))
    lo_h, hi_h = lo.permute(0,1,3,2), hi.permute(0,1,3,2)
    LL = F.conv2d(x_lo, lo_h, stride=(2,1)).view(B, C, -1, x_lo.shape[-1]//2+x_lo.shape[-1]%2)
    LH = F.conv2d(x_lo, hi_h, stride=(2,1)).view(B, C, -1, x_lo.shape[-1]//2+x_lo.shape[-1]%2)
    HL = F.conv2d(x_hi, lo_h, stride=(2,1)).view(B, C, -1, x_hi.shape[-1]//2+x_hi.shape[-1]%2)
    HH = F.conv2d(x_hi, hi_h, stride=(2,1)).view(B, C, -1, x_hi.shape[-1]//2+x_hi.shape[-1]%2)
    _, _, Hd, Wd = LL.shape
    return (LL.view(B,C,Hd,Wd), LH.view(B,C,Hd,Wd),
            HL.view(B,C,Hd,Wd), HH.view(B,C,Hd,Wd))


# ─── WaveDFE ─────────────────────────────────────────────────────────────────

class WaveDFE(nn.Module):
    """Wavelet Dual Feature Extraction — replaces DWConv in AIM's local branch."""

    def __init__(self, dim):
        super().__init__()
        self.linear   = nn.Conv2d(dim, dim, 1, 1, 0)
        self.wave_fuse = nn.Sequential(
            nn.Conv2d(4*dim, dim, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsample  = nn.Conv2d(dim, dim, 3, 1, 1)
        self.gate      = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.wave_norm = nn.GroupNorm(1, dim)

    def forward(self, x2d):
        """x2d: (B, C, H, W) → (B, C, H, W)"""
        B, C, H, W = x2d.shape
        x_ch = self.linear(x2d)

        pad_h, pad_w = H % 2, W % 2
        xp = F.pad(x2d, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h or pad_w) else x2d
        LL, LH, HL, HH = _dwt_haar_2d(xp)
        x_wave = self.wave_fuse(torch.cat([LL, LH, HL, HH], dim=1))
        x_wave = F.interpolate(x_wave, size=(H, W), mode='bilinear', align_corners=False)
        x_wave = self.wave_norm(self.upsample(x_wave))

        return x_ch + self.gate(x_ch) * x_wave


# ─── FGCA ────────────────────────────────────────────────────────────────────

class FreqGuidedChannelAttention(nn.Module):
    """Frequency-Guided Channel Attention — replaces SE avg-pool with DWT descriptors."""

    def __init__(self, dim, reduction=4):
        super().__init__()
        mid = max(dim // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(2*dim, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, dim),  nn.Sigmoid(),
        )

    def forward(self, x, x_size):
        """x: (B, L, C), x_size: (H, W) → (B, L, C)"""
        B, L, C = x.shape
        H, W = x_size
        x2d = x.transpose(1,2).view(B, C, H, W)

        pad_h, pad_w = H%2, W%2
        xp = F.pad(x2d, (0, pad_w, 0, pad_h), 'reflect') if (pad_h or pad_w) else x2d
        LL, LH, HL, HH = _dwt_haar_2d(xp)

        ll_desc = LL.mean(dim=[2,3])
        hf_desc = (LH.abs() + HL.abs() + HH.abs()).mean(dim=[2,3]) / 3.0
        gate = self.fc(torch.cat([ll_desc, hf_desc], dim=1))
        return x * gate.unsqueeze(1)


# ─── Wave_Adaptive_Spatial_Attention ─────────────────────────────────────────

class Wave_Adaptive_Spatial_Attention(nn.Module):
    """Adaptive_Spatial_Attention with WaveDFE replacing DWConv in AIM."""

    def __init__(self, dim, num_heads, reso=64, split_size=[8,8], shift_size=[1,2],
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., rg_idx=0, b_idx=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx = b_idx
        self.rg_idx = rg_idx
        self.patches_resolution = reso

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.branch_num = 2
        self.attns = nn.ModuleList([
            Spatial_Attention(dim//2, idx=i, split_size=split_size,
                              num_heads=num_heads//2, dim_out=dim//2,
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, position_bias=True)
            for i in range(self.branch_num)])

        assert 0 <= shift_size[0] < split_size[0]
        assert 0 <= shift_size[1] < split_size[1]

        if (rg_idx%2==0 and b_idx>0 and (b_idx-2)%4==0) or (rg_idx%2!=0 and b_idx%4==0):
            attn_mask = self._calculate_mask(reso, reso)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        # WaveDFE replaces DWConv
        self.wavedfe = WaveDFE(dim)

        # No BN in channel_interaction: AdaptiveAvgPool2d(1) → 1×1 spatial → BN can't compute stats
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1), nn.GELU(),
            nn.Conv2d(dim//8, dim, 1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim//16, 1), nn.GELU(),
            nn.Conv2d(dim//16, 1, 1),
        )

    def _calculate_mask(self, H, W):
        img_mask_0 = torch.zeros(1, H, W, 1)
        img_mask_1 = torch.zeros(1, H, W, 1)
        h0 = (slice(0,-self.split_size[0]), slice(-self.split_size[0],-self.shift_size[0]), slice(-self.shift_size[0],None))
        w0 = (slice(0,-self.split_size[1]), slice(-self.split_size[1],-self.shift_size[1]), slice(-self.shift_size[1],None))
        h1 = (slice(0,-self.split_size[1]), slice(-self.split_size[1],-self.shift_size[1]), slice(-self.shift_size[1],None))
        w1 = (slice(0,-self.split_size[0]), slice(-self.split_size[0],-self.shift_size[0]), slice(-self.shift_size[0],None))
        cnt = 0
        for h in h0:
            for w in w0: img_mask_0[:,h,w,:] = cnt; cnt += 1
        cnt = 0
        for h in h1:
            for w in w1: img_mask_1[:,h,w,:] = cnt; cnt += 1

        def make_mask(img_mask, sh, sw):
            m = img_mask.view(1, H//sh, sh, W//sw, sw, 1).permute(0,1,3,2,4,5).contiguous().view(-1,sh*sw)
            a = m.unsqueeze(1) - m.unsqueeze(2)
            return a.masked_fill(a!=0, -100.).masked_fill(a==0, 0.)

        return (make_mask(img_mask_0, self.split_size[0], self.split_size[1]),
                make_mask(img_mask_1, self.split_size[1], self.split_size[0]))

    def forward(self, x, H, W):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)

        max_sp = max(self.split_size)
        pad_r = (max_sp - W % max_sp) % max_sp
        pad_b = (max_sp - H % max_sp) % max_sp
        qkv = qkv.reshape(3*B, H, W, C).permute(0,3,1,2)
        qkv = F.pad(qkv, (0,pad_r,0,pad_b)).reshape(3,B,C,-1).transpose(-2,-1)
        _H, _W, _L = pad_b+H, pad_r+W, (pad_b+H)*(pad_r+W)

        if (self.rg_idx%2==0 and self.b_idx>0 and (self.b_idx-2)%4==0) or (self.rg_idx%2!=0 and self.b_idx%4==0):
            qkv_ = qkv.view(3,B,_H,_W,C)
            q0 = torch.roll(qkv_[:,:,:,:,:C//2], shifts=(-self.shift_size[0],-self.shift_size[1]), dims=(2,3)).view(3,B,_L,C//2)
            q1 = torch.roll(qkv_[:,:,:,:,C//2:], shifts=(-self.shift_size[1],-self.shift_size[0]), dims=(2,3)).view(3,B,_L,C//2)
            if self.patches_resolution!=_H or self.patches_resolution!=_W:
                m = self._calculate_mask(_H,_W)
                x1s = self.attns[0](q0,_H,_W,mask=m[0].to(x.device))
                x2s = self.attns[1](q1,_H,_W,mask=m[1].to(x.device))
            else:
                x1s = self.attns[0](q0,_H,_W,mask=self.attn_mask_0)
                x2s = self.attns[1](q1,_H,_W,mask=self.attn_mask_1)
            x1 = torch.roll(x1s, shifts=(self.shift_size[0],self.shift_size[1]), dims=(1,2))[:,:H,:W,:].reshape(B,L,C//2)
            x2 = torch.roll(x2s, shifts=(self.shift_size[1],self.shift_size[0]), dims=(1,2))[:,:H,:W,:].reshape(B,L,C//2)
        else:
            x1 = self.attns[0](qkv[:,:,:,:C//2],_H,_W)[:,:H,:W,:].reshape(B,L,C//2)
            x2 = self.attns[1](qkv[:,:,:,C//2:],_H,_W)[:,:H,:W,:].reshape(B,L,C//2)

        attened_x = torch.cat([x1, x2], dim=2)

        # WaveDFE local branch (replaces DWConv)
        conv_x = self.wavedfe(v)                                         # (B, C, H, W)

        # AIM cross-gating (unchanged logic)
        channel_map = self.channel_interaction(conv_x).permute(0,2,3,1).contiguous().view(B,1,C)
        spatial_map = self.spatial_interaction(attened_x.transpose(-2,-1).view(B,C,H,W))

        attened_x = attened_x * torch.sigmoid(channel_map)
        conv_x    = torch.sigmoid(spatial_map) * conv_x
        conv_x    = conv_x.permute(0,2,3,1).contiguous().view(B,L,C)

        x = self.proj_drop(self.proj(attened_x + conv_x))
        return x


# ─── WaveDATB ────────────────────────────────────────────────────────────────

class WaveDATB(nn.Module):
    """DATB with Wave_Adaptive_Spatial_Attention + FGCA before spatial blocks."""

    def __init__(self, dim, num_heads, reso=64, split_size=[2,4], shift_size=[1,2],
                 expansion_factor=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 rg_idx=0, b_idx=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.window_scale = [1, 2, 4]
        self.b_idx = b_idx

        if b_idx % 2 == 0:
            # Spatial block: Wave_Adaptive_Spatial_Attention + FGCA
            scaled_split = [s * self.window_scale[b_idx//2] for s in split_size]
            self.attn = Wave_Adaptive_Spatial_Attention(
                dim, num_heads=num_heads, reso=reso, split_size=scaled_split,
                shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, rg_idx=rg_idx, b_idx=b_idx)
            self.fgca = FreqGuidedChannelAttention(dim)
        else:
            # Channel block: unchanged
            self.attn = Adaptive_Channel_Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop)
            self.fgca = None

        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn   = SGFN(dim, int(dim*expansion_factor), dim, act_layer)
        self.norm2 = norm_layer(dim)

    def forward(self, x, x_size):
        H, W = x_size
        # FGCA before spatial attention
        if self.fgca is not None:
            x = self.fgca(x, x_size)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


# ─── WaveResidualGroup ────────────────────────────────────────────────────────

class WaveResidualGroup(nn.Module):

    def __init__(self, dim, reso, num_heads, split_size=[2,4], expansion_factor=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_paths=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=2,
                 use_chk=False, resi_connection='1conv', rg_idx=0):
        super().__init__()
        self.use_chk = use_chk
        self.blocks = nn.ModuleList([
            WaveDATB(dim=dim, num_heads=num_heads, reso=reso, split_size=split_size,
                     shift_size=[split_size[0]//2, split_size[1]//2],
                     expansion_factor=expansion_factor, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop, drop_path=drop_paths[i],
                     act_layer=act_layer, norm_layer=norm_layer, rg_idx=rg_idx, b_idx=i)
            for i in range(depth)])
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        H, W = x_size
        res = x
        for blk in self.blocks:
            x = blk(x, x_size)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H).contiguous()
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return res + x


# ─── WaveDAT ─────────────────────────────────────────────────────────────────

@ARCH_REGISTRY.register()
class WaveDAT(nn.Module):
    """
    WaveDAT = MSW-DAT + WaveDFE (in AIM) + FGCA (before spatial blocks).
    Identical constructor kwargs to DAT for easy config swap.
    """

    def __init__(self, img_size=64, in_chans=3, embed_dim=180, split_size=[2,4],
                 depth=[2,2,2,2], num_heads=[2,2,2,2], expansion_factor=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_chk=False, upscale=2, img_range=1., resi_connection='1conv',
                 upsampler='pixelshuffle', **kwargs):
        super().__init__()

        num_feat = 64
        self.img_range = img_range
        self.mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1,3,1,1) if in_chans==3 else torch.zeros(1,1,1,1)
        self.upscale   = upscale
        self.upsampler = upsampler

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.num_layers = len(depth)
        self.embed_dim  = embed_dim

        self.before_RG = nn.Sequential(Rearrange('b c h w -> b (h w) c'), nn.LayerNorm(embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, int(np.sum(depth)))]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(WaveResidualGroup(
                dim=embed_dim, num_heads=num_heads[i], reso=img_size,
                split_size=split_size, expansion_factor=expansion_factor,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]):sum(depth[:i+1])],
                act_layer=act_layer, norm_layer=norm_layer,
                depth=depth[i], use_chk=use_chk,
                resi_connection=resi_connection, rg_idx=i))

        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        if upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)

        self.apply(self._init_weights)
        # Identity init for new modules — must be AFTER self.apply() to avoid override
        self._init_identity()

    def _init_identity(self):
        """Init WaveDFE and FGCA to near-identity so warm-start is not disrupted."""
        for module in self.modules():
            if isinstance(module, WaveDFE):
                # gate ≈ 0 → wavelet branch off at start → out = x_ch (like DWConv)
                nn.init.constant_(module.gate[0].bias, -6.0)
                nn.init.zeros_(module.gate[0].weight)
            if isinstance(module, FreqGuidedChannelAttention):
                # gate ≈ 1 → identity → x * 1 = x (no disruption)
                nn.init.constant_(module.fc[2].bias, 5.0)
                nn.init.zeros_(module.fc[2].weight)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward_features(self, x):
        _, _, H, W = x.shape
        x_size = (H, W)
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_last(self.upsample(self.conv_before_upsample(x)))

        x = x / self.img_range + self.mean
        return x
