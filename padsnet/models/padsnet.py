import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


# 3D Swin Transformer backbone

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


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
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


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, D, H, W, C = x.shape
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=(1, 7, 7), mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        return x


class CleanSwinTransformer3D(nn.Module):
    def __init__(self, pretrained=None, patch_size=(2, 4, 4), in_chans=4,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=(8, 7, 7), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=False, frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                        norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Encode an RGB+mask clip with shape [B, 4, T, H, W]."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)

        # LayerNorm operates on a channel-last representation.
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)

        # Spatial pooling preserves the temporal sequence.
        x = x.mean(dim=[3, 4])

        return x.transpose(1, 2)


class HaarWaveletBlock(nn.Module):
    """Native PyTorch implementation of a 1D Haar transform."""

    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor(1.0 / np.sqrt(2.0)))

    def forward_dwt(self, x):
        """Map [B, T, C] to low/high bands with shape [B, T/2, C]."""
        B, T, C = x.shape
        if T % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='replicate')

        x = x.permute(0, 2, 1)  # [B, C, T]

        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]

        low = (x_even + x_odd) * self.scale
        high = (x_even - x_odd) * self.scale

        return low.permute(0, 2, 1), high.permute(0, 2, 1)  # [B, T/2, C]

    def forward_idwt(self, low, high):
        """Reconstruct [B, T, C] from low/high bands."""
        low = low.permute(0, 2, 1)  # [B, C, T/2]
        high = high.permute(0, 2, 1)

        x_even = (low + high) * self.scale
        x_odd = (low - high) * self.scale

        # Interleave even and odd time steps.
        B, C, T_half = low.shape
        x_out = torch.zeros((B, C, T_half * 2), device=low.device)
        x_out[:, :, 0::2] = x_even
        x_out[:, :, 1::2] = x_odd

        return x_out.permute(0, 2, 1)  # [B, T, C]


class WaveletCrossAttention(nn.Module):
    """Cross-attend visual and distance sequences in Haar frequency bands."""

    def __init__(self, vis_dim=768, dist_dim=128, embed_dim=256, num_heads=4):
        super().__init__()

        self.vis_proj = nn.Linear(vis_dim, embed_dim)
        self.dist_proj = nn.Linear(dist_dim, embed_dim)
        self.act = nn.GELU()

        self.dwt = HaarWaveletBlock()

        # Low and high bands learn motion and vibration alignment separately.
        self.attn_low = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_low = nn.LayerNorm(embed_dim)

        self.attn_high = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_high = nn.LayerNorm(embed_dim)

        self.out_norm_v = nn.LayerNorm(embed_dim)
        self.out_norm_d = nn.LayerNorm(embed_dim)

    def forward(self, vis_seq, dist_seq):
        """
        vis_seq:  [B, T, Vis_Dim]
        dist_seq: [B, T, Dist_Dim]
        """
        v = self.act(self.vis_proj(vis_seq))  # [B, T, E]
        d = self.act(self.dist_proj(dist_seq))  # [B, T, E]

        v_low, v_high = self.dwt.forward_dwt(v)
        d_low, d_high = self.dwt.forward_dwt(d)

        v_low_attn, _ = self.attn_low(v_low, d_low, d_low)
        d_low_attn, _ = self.attn_low(d_low, v_low, v_low)

        v_low_refined = self.norm_low(v_low + v_low_attn)
        d_low_refined = self.norm_low(d_low + d_low_attn)

        v_high_attn, _ = self.attn_high(v_high, d_high, d_high)
        d_high_attn, _ = self.attn_high(d_high, v_high, v_high)

        v_high_refined = self.norm_high(v_high + v_high_attn)
        d_high_refined = self.norm_high(d_high + d_high_attn)

        v_enhanced = self.dwt.forward_idwt(v_low_refined, v_high_refined)  # [B, T, E]
        d_enhanced = self.dwt.forward_idwt(d_low_refined, d_high_refined)  # [B, T, E]

        v_out = self.out_norm_v(v + v_enhanced)
        d_out = self.out_norm_d(d + d_enhanced)

        return v_out, d_out

class PhysicsFeatureExtractor(nn.Module):
    """Extract nine dynamics, geometry, symmetry, and frequency features."""

    def __init__(self, l_idx=0, r_idx=3, eps=1e-6):
        super().__init__()
        self.l_idx = l_idx
        self.r_idx = r_idx
        self.eps = eps

        # The nine physical features have different scales.
        self.norm = nn.BatchNorm1d(9)

    def forward(self, distance_seq):
        """Compute physical features from [B, T, C] distance sequences."""
        l_seq = distance_seq[:, :, self.l_idx]
        r_seq = distance_seq[:, :, self.r_idx]

        vel_l = torch.diff(l_seq, dim=1)
        vel_r = torch.diff(r_seq, dim=1)
        mobility = (torch.std(vel_l, dim=1, keepdim=True) +
                    torch.std(vel_r, dim=1, keepdim=True)) / 2

        gap_seq = torch.abs(l_seq) + torch.abs(r_seq)
        mean_gap = torch.mean(gap_seq, dim=1, keepdim=True)
        min_gap, _ = torch.min(gap_seq, dim=1, keepdim=True)

        l_amp = torch.std(l_seq, dim=1, keepdim=True)
        r_amp = torch.std(r_seq, dim=1, keepdim=True)
        asymmetry = torch.abs(l_amp - r_amp) / (l_amp + r_amp + self.eps)

        l_centered = l_seq - l_seq.mean(dim=1, keepdim=True)
        r_centered = r_seq - r_seq.mean(dim=1, keepdim=True)
        correlation = torch.sum(l_centered * r_centered, dim=1, keepdim=True) / \
                      (torch.sqrt(torch.sum(l_centered ** 2, dim=1, keepdim=True)) * torch.sqrt(
                          torch.sum(r_centered ** 2, dim=1, keepdim=True)) + self.eps)
        # Frequency-domain energy and periodicity.
        l_seq_cpu = l_seq.detach().cpu()
        r_seq_cpu = r_seq.detach().cpu()

        fft_l = torch.abs(torch.fft.rfft(l_seq_cpu, dim=1)).to(l_seq.device)
        fft_r = torch.abs(torch.fft.rfft(r_seq_cpu, dim=1)).to(r_seq.device)

        energy_l = torch.sum(fft_l[:, 1:], dim=1, keepdim=True)
        energy_r = torch.sum(fft_r[:, 1:], dim=1, keepdim=True)
        energy = (energy_l + energy_r) / 2

        peak_energy_l, _ = torch.max(fft_l[:, 1:], dim=1, keepdim=True)
        peak_energy_r, _ = torch.max(fft_r[:, 1:], dim=1, keepdim=True)

        periodicity_l = peak_energy_l / (energy_l + self.eps)
        periodicity_r = peak_energy_r / (energy_r + self.eps)
        periodicity = (periodicity_l + periodicity_r) / 2

        zcr_l = ((l_centered[:, :-1] * l_centered[:, 1:]) < 0).float().mean(dim=1, keepdim=True)
        zcr_r = ((r_centered[:, :-1] * r_centered[:, 1:]) < 0).float().mean(dim=1, keepdim=True)
        zcr = (zcr_l + zcr_r) / 2

        raw_feats = torch.cat([
            mobility,
            mean_gap,
            min_gap,
            asymmetry,
            correlation,
            energy,
            periodicity,
            zcr,
            gap_seq.std(dim=1, keepdim=True),
        ], dim=1)
        return self.norm(raw_feats)


class PhysicsAttentionFusion(nn.Module):
    """Gate visual and temporal features with explicit physical features."""

    def __init__(self, physics_dim, visual_dim, lstm_dim):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(physics_dim, 64),
            nn.ReLU(),
            nn.Linear(64, visual_dim + lstm_dim),
            nn.Sigmoid()
        )

    def forward(self, visual_feats, lstm_feats, physics_feats):
        gates = self.gate_fc(physics_feats)
        gate_v, gate_l = torch.split(gates, [visual_feats.shape[1], lstm_feats.shape[1]], dim=1)

        # Residual gating retains each branch's original representation.
        visual_refined = visual_feats * (1 + gate_v)
        lstm_refined = lstm_feats * (1 + gate_l)
        return torch.cat([visual_refined, lstm_refined, physics_feats], dim=1)


class LaryngealRecognitionModel(nn.Module):
    def __init__(self, num_classes=3, distance_feat_dim=8, lstm_hidden=64):
        super().__init__()

        # RGB+mask visual branch.
        self.backbone = CleanSwinTransformer3D(
            in_chans=4,  # RGB + Mask
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            patch_size=(2, 4, 4),
            window_size=(8, 7, 7),
            drop_path_rate=0.1
        )
        self.visual_dim = 768
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # Vocal-fold distance branch.
        self.distance_lstm = nn.LSTM(
            input_size=distance_feat_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.dist_out_dim = lstm_hidden * 2

        # Align the 32-step LSTM output to the 16-step visual sequence.
        self.temporal_align = nn.Conv1d(
            in_channels=self.dist_out_dim,
            out_channels=self.dist_out_dim,
            kernel_size=3, stride=2, padding=1
        )

        self.fusion_dim = 256
        self.wca = WaveletCrossAttention(
            vis_dim=self.visual_dim,
            dist_dim=self.dist_out_dim,
            embed_dim=self.fusion_dim,
            num_heads=4
        )
        # Indices 0 and 3 are the left/right mean distances.
        self.physics_extractor = PhysicsFeatureExtractor(l_idx=0, r_idx=3)
        self.physics_dim = 9

        self.fusion = PhysicsAttentionFusion(self.physics_dim, self.fusion_dim, self.fusion_dim)

        total_dim = self.fusion_dim + self.fusion_dim + self.physics_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, masks, distance_seq):
        """
        images: [B, 3, T, H, W]
        masks: [B, T, H, W]
        distance_seq: [B, T, 8]
        """
        mask_input = masks.unsqueeze(1).float() / 255.0
        x_in = torch.cat([images, mask_input], dim=1)
        x = self.backbone(x_in)

        lstm_out, _ = self.distance_lstm(distance_seq)
        lstm_seq_aligned = self.temporal_align(lstm_out.transpose(1, 2)).transpose(1, 2)
        vis_enhanced_seq, dist_enhanced_seq = self.wca(x, lstm_seq_aligned)

        # Aggregate aligned temporal features before physics-aware gating.
        vis_vec = vis_enhanced_seq.mean(dim=1)
        dist_vec = dist_enhanced_seq.mean(dim=1)
        physics_feats = self.physics_extractor(distance_seq)

        combined_feats = self.fusion(vis_vec, dist_vec, physics_feats)
        return self.classifier(combined_feats)
