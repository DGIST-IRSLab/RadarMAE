# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
# MAE-pretrained encoder readout for K-Radar fine-tuning.
# Adapted from facebookresearch/mae (https://github.com/facebookresearch/mae):
# ViT encoder blocks (models_mae.py) and sincos position embedding
# (util/pos_embed.py). The stripe tokenizer mirrors RadarMAE's models_radarmae.py.
# --------------------------------------------------------
import copy
import math
import os
from functools import partial
from typing import Tuple, Optional, Callable, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from scipy.io import loadmat

from .stripe_modules import DualStreamStripeBlock


class ChannelsLastBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that handles (B, N, C) tensors from ViT blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.transpose(1, 2)
        x = super().forward(x)
        return x.transpose(1, 2)


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors (ViTDet-style)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class PhysicalPositionalEncoding(nn.Module):
    """Physical coordinate-based positional encoding with Fourier features."""
    def __init__(self, embed_dim: int, L: int = 10):
        super().__init__()
        self.L = L
        self.mlp = nn.Sequential(
            nn.Linear(2 * L, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        device = coords.device
        freqs = torch.pow(2.0, torch.arange(self.L, device=device).float()) * np.pi
        x = coords.unsqueeze(-1) * freqs.unsqueeze(0)
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.mlp(enc).unsqueeze(0)


class StandardPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (MAE-style)."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N,) 1D position indices
        Returns:
            pos_embed: (1, N, embed_dim)
        """
        device = coords.device
        dtype = coords.dtype

        # Get 1D sincos position embedding
        omega = torch.arange(self.embed_dim // 2, device=device, dtype=torch.float32)
        omega /= self.embed_dim / 2.0
        omega = 1.0 / (10000.0 ** omega)  # (D/2,)

        coords_1d = coords.reshape(-1).float()  # (N,)
        out = torch.einsum('m,d->md', coords_1d, omega)  # (N, D/2)

        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)

        emb = torch.cat([emb_sin, emb_cos], dim=1)  # (N, D)
        return emb.to(dtype).unsqueeze(0)  # (1, N, D)


def _build_norm_layer(
    norm: Optional[Union[str, Callable[[int], nn.Module]]],
    channels: int,
) -> Optional[nn.Module]:
    if norm is None:
        return None
    if isinstance(norm, nn.Module):
        return copy.deepcopy(norm)
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm_key = norm.upper()
        if norm_key == "BN":
            return nn.BatchNorm2d(channels)
        if norm_key == "LN":
            return LayerNorm2d(channels)
        raise ValueError(f"Unsupported norm '{norm}' for SimpleFeaturePyramid.")
    return norm(channels)


class SimpleFeaturePyramid(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factors: Tuple[float, ...],
        norm: Optional[Union[str, Callable[[int], nn.Module]]] = "LN",
    ):
        super().__init__()
        self.scale_factors = scale_factors
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, scale, norm) for scale in scale_factors]
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        scale: float,
        norm: Optional[Union[str, Callable[[int], nn.Module]]],
    ) -> nn.Module:
        norm_spec = norm

        def _maybe_norm(channels: int) -> Optional[nn.Module]:
            return _build_norm_layer(norm_spec, channels)

        use_bias = norm_spec is None or (isinstance(norm_spec, str) and len(norm_spec) == 0)
        layers = []
        if abs(scale - 4.0) < 1e-6:
            mid_ch = in_channels // 2
            out_ch = in_channels // 4
            layers.append(nn.ConvTranspose2d(in_channels, mid_ch, kernel_size=2, stride=2))
            norm_layer = _maybe_norm(mid_ch)
            if norm_layer is not None:
                layers.append(norm_layer)
            layers.append(nn.GELU())
            layers.append(nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=2, stride=2))
            in_channels = out_ch
        elif abs(scale - 2.0) < 1e-6:
            out_ch = in_channels // 2
            layers.append(nn.ConvTranspose2d(in_channels, out_ch, kernel_size=2, stride=2))
            in_channels = out_ch
        elif abs(scale - 1.0) < 1e-6:
            pass
        elif abs(scale - 0.5) < 1e-6:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            raise NotImplementedError(f"scale factor {scale} is not supported in ViTDet-style SFP")

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias))
        norm_layer = _maybe_norm(out_channels)
        if norm_layer is not None:
            layers.append(norm_layer)
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias))
        norm_layer = _maybe_norm(out_channels)
        if norm_layer is not None:
            layers.append(norm_layer)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(stage(x) for stage in self.stages)


class ResBlock2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ASPPLiteBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPPLiteBlock(nn.Module):
    def __init__(self, channels: int, dilations=(1, 2, 3)) -> None:
        super().__init__()
        dilations = tuple(int(d) for d in dilations)
        if len(dilations) == 0:
            raise ValueError('ASPPLiteBlock requires at least one dilation value.')

        num_branches = 1 + len(dilations)
        branch_channels = max(channels // num_branches, 32)
        self.point_branch = nn.Sequential(
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.dilated_branches = nn.ModuleList(
            [ASPPLiteBranch(channels, branch_channels, dilation=d) for d in dilations]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * num_branches, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        branches = [self.point_branch(x)]
        branches.extend(branch(x) for branch in self.dilated_branches)
        out = torch.cat(branches, dim=1)
        out = self.fuse(out)
        out = out + identity
        out = self.relu(out)
        return out


class BEVRefineNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_blocks: int,
        refine_type: str = 'aspp_lite',
        dilations=(1, 2, 3),
    ) -> None:
        super().__init__()
        self.use_skip = num_blocks > 0
        refine_type = str(refine_type).lower().strip()

        if num_blocks < 1:
            self.blocks = nn.Identity()
        elif refine_type in ('resblock', 'residual', 'res'):
            self.blocks = nn.Sequential(*[ResBlock2D(hidden_channels) for _ in range(num_blocks)])
        elif refine_type in ('aspp_lite', 'aspp-lite', 'aspp'):
            self.blocks = nn.Sequential(
                *[ASPPLiteBlock(hidden_channels, dilations=dilations) for _ in range(num_blocks)]
            )
        else:
            raise ValueError(f"Unsupported REFINE.TYPE='{refine_type}'")

        if hidden_channels == in_channels:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        else:
            self.in_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
            self.out_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.in_proj(x)
        out = self.blocks(out)
        out = self.out_proj(out)
        if self.use_skip:
            out = out + x
        return out


class DirectStripeToBEV(nn.Module):
    """
    Direct 1D-to-BEV deformable fusion from stripe tokens (range + azimuth).
    - Query: learnable BEV embeddings (H_bev x W_bev).
    - Value: concatenated encoded stripe tokens (range + azimuth) in a single token bank.
    - Reference points: geometric prior that maps each BEV pixel to (theta_norm, r_norm) in [-1, 1].

    This avoids reconstructing a dense 2D polar feature map (B,C,H,W) from 1D stripes while still
    letting BEV queries fuse both axes through deformable sampling.
    """
    def __init__(
        self,
        embed_dim: int,
        bev_h: int,
        bev_w: int,
        num_heads: int = 4,
        num_points: int = 4,
        fuse: str = "mean",
        align_corners: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.bev_h = int(bev_h)
        self.bev_w = int(bev_w)
        self.num_heads = num_heads
        self.num_points = num_points
        self.fuse = str(fuse).lower().strip()
        if self.fuse not in ("sum", "mean"):
            raise ValueError(f"DirectStripeToBEV.fuse must be 'sum' or 'mean', got '{self.fuse}'")
        self.align_corners = bool(align_corners)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Learnable BEV Query (target geometry)
        self.bev_query = nn.Parameter(torch.zeros(1, self.bev_h, self.bev_w, embed_dim))
        nn.init.normal_(self.bev_query, std=0.02)

        # Offsets: (delta_theta, delta_r) per head/point
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2)
        # Attention weights: separate weights for range-stream and azimuth-stream samples
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points * 2)
        
        # Linear projections for Value and Output
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize offsets to 0 (start from reference points)
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        
        # Initialize attention weights (uniform distribution initially) -> will be softmaxed per stream
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        all_tok: torch.Tensor,
        num_range_tokens: int,
        num_azim_tokens: int,
        polar_grid: torch.Tensor,
        roi_xy: Optional[Tuple[float, float, float, float]] = None,
        polar_extents: Optional[Tuple[float, float, float, float]] = None,
    ) -> torch.Tensor:
        """
        all_tok: (B, L_r+L_a, C) or (B, C, L_r+L_a), where
                 the first L_r tokens are range tokens and the next L_a are azimuth tokens.
        polar_grid: (1, H_bev, W_bev, 2) normalized (theta, r) per BEV pixel in [-1, 1]
        """

        def _to_blc(x: torch.Tensor, name: str) -> torch.Tensor:
            if x.ndim != 3:
                raise ValueError(f"{name} must be 3D, got shape={tuple(x.shape)}")
            if x.shape[-1] == self.embed_dim:
                return x
            if x.shape[1] == self.embed_dim:
                return x.transpose(1, 2).contiguous()
            raise ValueError(
                f"{name} must be (B,L,C) or (B,C,L) with C={self.embed_dim}, got shape={tuple(x.shape)}"
            )

        all_tok = _to_blc(all_tok, "all_tok")
        B, L_all, C = all_tok.shape
        if C != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got C={C}")
        L_r = int(num_range_tokens)
        L_a = int(num_azim_tokens)
        if L_r <= 0 or L_a <= 0:
            raise ValueError(f"num_range_tokens and num_azim_tokens must be positive, got ({L_r}, {L_a})")
        if L_r + L_a != L_all:
            raise ValueError(
                f"Concatenated token length mismatch: L_all={L_all}, expected L_r+L_a={L_r + L_a}"
            )

        if polar_grid.ndim != 4 or polar_grid.shape[0] != 1 or polar_grid.shape[-1] != 2:
            raise ValueError(f"polar_grid must be (1,H,W,2), got shape={tuple(polar_grid.shape)}")
        if polar_grid.shape[1] != self.bev_h or polar_grid.shape[2] != self.bev_w:
            raise ValueError(
                f"polar_grid spatial mismatch: expected (1,{self.bev_h},{self.bev_w},2), got {tuple(polar_grid.shape)}"
            )

        device = self.bev_query.device
        dtype = self.bev_query.dtype
        polar_grid = polar_grid.to(device=device, dtype=dtype)

        # 1) Query & predictions
        query = self.bev_query.expand(B, -1, -1, -1)  # (B, H_bev, W_bev, C)

        offsets = self.sampling_offsets(query).view(
            B, self.bev_h, self.bev_w, self.num_heads, self.num_points, 2
        )  # (B,H,W,Heads,Points,2) in (d_theta, d_r)

        weights = self.attention_weights(query).view(
            B, self.bev_h, self.bev_w, self.num_heads, self.num_points, 2
        )  # last dim: (range_stream, azim_stream)
        w_range = F.softmax(weights[..., 0], dim=-1)
        w_azim = F.softmax(weights[..., 1], dim=-1)

        # 2) Reference points + offsets
        ref = polar_grid.unsqueeze(3).unsqueeze(3)  # (1,H,W,1,1,2)
        sampling_locations = ref + offsets  # (B,H,W,Heads,Points,2)
        theta_loc = sampling_locations[..., 0]
        r_loc = sampling_locations[..., 1]

        theta_ref = polar_grid[..., 0].expand(B, -1, -1)  # (B,H,W)
        r_ref = polar_grid[..., 1].expand(B, -1, -1)      # (B,H,W)
        theta_ref = theta_ref.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, self.num_heads, self.num_points)
        r_ref = r_ref.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, self.num_heads, self.num_points)

        def _valid_mask(theta_norm: torch.Tensor, r_norm: torch.Tensor) -> torch.Tensor:
            within = (theta_norm >= -1.0) & (theta_norm <= 1.0) & (r_norm >= -1.0) & (r_norm <= 1.0)
            if roi_xy is None or polar_extents is None:
                return within
            x_min, x_max, y_min, y_max = roi_xy
            r_min, r_max, a_min, a_max = polar_extents

            # Compute in float32 for numerical stability (e.g., AMP/FP16).
            t_norm_f = theta_norm.float()
            r_norm_f = r_norm.float()
            t = (t_norm_f + 1.0) * 0.5 * (a_max - a_min) + a_min
            r = (r_norm_f + 1.0) * 0.5 * (r_max - r_min) + r_min
            x = r * torch.cos(t)
            y = -r * torch.sin(t)
            in_roi = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
            return within & in_roi

        # Range stream samples along r; theta is fixed to reference.
        valid_range = _valid_mask(theta_ref, r_loc)
        # Azimuth stream samples along theta; r is fixed to reference.
        valid_azim = _valid_mask(theta_loc, r_ref)

        w_range = w_range * valid_range.to(w_range.dtype)
        w_azim = w_azim * valid_azim.to(w_azim.dtype)
        w_range = w_range / w_range.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        w_azim = w_azim / w_azim.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # 3) Prepare a single concatenated value bank from all encoded tokens.
        # (B, L_all, C) -> (B*Heads, C_head, L_all, 1)
        val_all = self.value_proj(all_tok).view(B, L_all, self.num_heads, self.head_dim)
        val_all = val_all.permute(0, 2, 3, 1).reshape(B * self.num_heads, self.head_dim, L_all, 1)

        def _norm_to_index(coord_norm: torch.Tensor, length: int) -> torch.Tensor:
            if length <= 1:
                return torch.zeros_like(coord_norm)
            return (coord_norm + 1.0) * 0.5 * float(length - 1)

        def _index_to_grid(index: torch.Tensor, total_length: int) -> torch.Tensor:
            if total_length <= 1:
                return torch.zeros_like(index)
            return index / float(total_length - 1) * 2.0 - 1.0

        # Map sampling coordinates to concatenated token-bank indices.
        # - range stream uses [0, L_r)
        # - azimuth stream uses [L_r, L_r+L_a)
        idx_r = _norm_to_index(r_loc, L_r)
        idx_a = _norm_to_index(theta_loc, L_a) + float(L_r)

        grid_r = _index_to_grid(idx_r, L_all)
        grid_a = _index_to_grid(idx_a, L_all)

        def _sample_from_bank(grid_idx_norm: torch.Tensor) -> torch.Tensor:
            # (B,H,W,Heads,Points) -> (B*Heads, H, W*Points)
            loc = grid_idx_norm.permute(0, 3, 1, 2, 4).flatten(0, 1)
            loc = loc.flatten(2, 3)
            grid = torch.stack([torch.zeros_like(loc), loc], dim=-1)

            if not val_all.is_contiguous():
                val = val_all.contiguous()
            else:
                val = val_all
            if not grid.is_contiguous():
                grid = grid.contiguous()

            try:
                sampled = F.grid_sample(
                    val,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=self.align_corners,
                )
            except RuntimeError as e:
                if "CUDNN_STATUS_NOT_SUPPORTED" not in str(e):
                    raise
                with torch.backends.cudnn.flags(enabled=False):
                    sampled = F.grid_sample(
                        val,
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=self.align_corners,
                    )

            # (B*Heads, C_head, H, W*P) -> (B,H,W,Heads,P,C_head)
            sampled = sampled.view(B, self.num_heads, self.head_dim, self.bev_h, self.bev_w, self.num_points)
            sampled = sampled.permute(0, 3, 4, 1, 5, 2)
            return sampled

        # 4) Sample from a single token bank with both axis references.
        samp_r = _sample_from_bank(grid_r)
        samp_a = _sample_from_bank(grid_a)

        out_r = (samp_r * w_range.unsqueeze(-1)).sum(dim=4)  # (B,H,W,Heads,C_head)
        out_a = (samp_a * w_azim.unsqueeze(-1)).sum(dim=4)
        out_r = out_r.flatten(3)
        out_a = out_a.flatten(3)

        if self.fuse == "sum":
            out = out_r + out_a
        else:
            out = 0.5 * (out_r + out_a)
        return self.output_proj(out)


class DirectPatchToBEV(nn.Module):
    """
    Direct 2D patch-token-to-BEV deformable fusion.

    Patch pretraining produces a regular 2D token grid.  This module keeps the
    same BEV-query / offset / attention formulation used by DirectStripeToBEV,
    but samples from the encoded 2D patch-token feature map instead of two 1D
    token banks.
    """
    def __init__(
        self,
        embed_dim: int,
        bev_h: int,
        bev_w: int,
        num_heads: int = 4,
        num_points: int = 4,
        align_corners: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.bev_h = int(bev_h)
        self.bev_w = int(bev_w)
        self.num_heads = int(num_heads)
        self.num_points = int(num_points)
        self.align_corners = bool(align_corners)
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.bev_query = nn.Parameter(torch.zeros(1, self.bev_h, self.bev_w, self.embed_dim))
        nn.init.normal_(self.bev_query, std=0.02)

        # Offsets are represented in normalized polar-coordinate order:
        # (delta_theta, delta_range), matching DirectStripeToBEV.
        self.sampling_offsets = nn.Linear(self.embed_dim, self.num_heads * self.num_points * 2)
        self.attention_weights = nn.Linear(self.embed_dim, self.num_heads * self.num_points)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        patch_tok: torch.Tensor,
        grid_hw: Tuple[int, int],
        polar_grid: torch.Tensor,
        pretrain_axis: str,
        roi_xy: Optional[Tuple[float, float, float, float]] = None,
        polar_extents: Optional[Tuple[float, float, float, float]] = None,
    ) -> torch.Tensor:
        if patch_tok.ndim != 3:
            raise ValueError(f"patch_tok must be (B,L,C), got shape={tuple(patch_tok.shape)}")
        B, L, C = patch_tok.shape
        if C != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got C={C}")
        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
        if grid_h <= 0 or grid_w <= 0 or grid_h * grid_w != L:
            raise ValueError(f"Invalid patch grid {grid_hw} for token length L={L}")

        if polar_grid.ndim != 4 or polar_grid.shape[0] != 1 or polar_grid.shape[-1] != 2:
            raise ValueError(f"polar_grid must be (1,H,W,2), got shape={tuple(polar_grid.shape)}")
        if polar_grid.shape[1] != self.bev_h or polar_grid.shape[2] != self.bev_w:
            raise ValueError(
                f"polar_grid spatial mismatch: expected (1,{self.bev_h},{self.bev_w},2), got {tuple(polar_grid.shape)}"
            )

        axis = str(pretrain_axis).lower().strip()
        if axis not in ("azimuth_range", "range_azimuth"):
            raise ValueError(f"Unsupported pretrain_axis='{pretrain_axis}'")

        device = self.bev_query.device
        dtype = self.bev_query.dtype
        polar_grid = polar_grid.to(device=device, dtype=dtype)

        query = self.bev_query.expand(B, -1, -1, -1)
        offsets = self.sampling_offsets(query).view(
            B, self.bev_h, self.bev_w, self.num_heads, self.num_points, 2
        )
        weights = self.attention_weights(query).view(
            B, self.bev_h, self.bev_w, self.num_heads, self.num_points
        )
        weights = F.softmax(weights, dim=-1)

        ref = polar_grid.unsqueeze(3).unsqueeze(3)
        sampling_locations = ref + offsets
        theta_loc = sampling_locations[..., 0]
        r_loc = sampling_locations[..., 1]

        def _valid_mask(theta_norm: torch.Tensor, r_norm: torch.Tensor) -> torch.Tensor:
            within = (theta_norm >= -1.0) & (theta_norm <= 1.0) & (r_norm >= -1.0) & (r_norm <= 1.0)
            if roi_xy is None or polar_extents is None:
                return within
            x_min, x_max, y_min, y_max = roi_xy
            r_min, r_max, a_min, a_max = polar_extents

            t_norm_f = theta_norm.float()
            r_norm_f = r_norm.float()
            t = (t_norm_f + 1.0) * 0.5 * (a_max - a_min) + a_min
            r = (r_norm_f + 1.0) * 0.5 * (r_max - r_min) + r_min
            x = r * torch.cos(t)
            y = -r * torch.sin(t)
            in_roi = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
            return within & in_roi

        valid = _valid_mask(theta_loc, r_loc)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # Convert polar coordinates to the pretraining tensor grid.
        # grid_sample expects (x=width, y=height).  For azimuth_range, H is
        # azimuth and W is range; for range_azimuth, H is range and W is azimuth.
        if axis == "azimuth_range":
            grid_x = r_loc
            grid_y = theta_loc
        else:
            grid_x = theta_loc
            grid_y = r_loc

        val = self.value_proj(patch_tok).view(B, grid_h, grid_w, self.num_heads, self.head_dim)
        val = val.permute(0, 3, 4, 1, 2).reshape(B * self.num_heads, self.head_dim, grid_h, grid_w)

        grid_x = grid_x.permute(0, 3, 1, 2, 4).flatten(0, 1).flatten(2, 3)
        grid_y = grid_y.permute(0, 3, 1, 2, 4).flatten(0, 1).flatten(2, 3)
        grid = torch.stack([grid_x, grid_y], dim=-1).contiguous()

        try:
            sampled = F.grid_sample(
                val.contiguous(),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=self.align_corners,
            )
        except RuntimeError as e:
            if "CUDNN_STATUS_NOT_SUPPORTED" not in str(e):
                raise
            with torch.backends.cudnn.flags(enabled=False):
                sampled = F.grid_sample(
                    val.contiguous(),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=self.align_corners,
                )

        sampled = sampled.view(B, self.num_heads, self.head_dim, self.bev_h, self.bev_w, self.num_points)
        sampled = sampled.permute(0, 3, 4, 1, 5, 2)
        out = (sampled * weights.unsqueeze(-1)).sum(dim=4)
        out = out.flatten(3)
        return self.output_proj(out)


class MAEPolarBEVBackbone(nn.Module):
    """
    MAE encoder with Deformable Cross-Attention for View Transformation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_dataset_ver2 = cfg.get("cfg_dataset_ver2", False)

        cfg_model = cfg.MODEL
        cfg_bb = cfg_model.BACKBONE
        mae_cfg = cfg_bb.get("MAE", {})
        bev_cfg = cfg_bb.get("POLAR_BEV", {})

        self.input_key = cfg_bb.get("INPUT_KEY", "ra_map")
        self.align_corners = bev_cfg.get("ALIGN_CORNERS", True)
        # Always use sensor physical bins from resources/info_arr.mat
        self.arr_range, self.arr_azimuth = self._load_physical_bins()

        # MAE encoder configuration
        self.embed_dim = mae_cfg.get("EMBED_DIM", 768)
        depth = mae_cfg.get("DEPTH", 12)
        num_heads = mae_cfg.get("NUM_HEADS", 12)
        mlp_ratio = mae_cfg.get("MLP_RATIO", 4.0)

        tokenizer = str(mae_cfg.get("TOKENIZER", "patch")).lower()
        if tokenizer in ("stripe", "1dstripe", "stripe1d", "1d", "1dpatch", "1d_patch"):
            self.tokenizer = "stripe"
        else:
            self.tokenizer = "patch"

        # PATCH_SIZE:
        # - patch tokenizer: (H, W) patch size for PatchEmbed
        # - stripe tokenizer: pooling stride to form a 2D grid (keeps 1D stripe tokenization)
        patch_size = mae_cfg.get("PATCH_SIZE", 16)
        if isinstance(patch_size, (list, tuple)):
            if len(patch_size) != 2:
                raise ValueError("PATCH_SIZE must be an int or a 2-tuple/list.")
            patch_size = (int(patch_size[0]), int(patch_size[1]))
        else:
            patch_size = (int(patch_size), int(patch_size))
        self.patch_size = patch_size

        input_size = mae_cfg.get("IMG_SIZE", None)
        if input_size is None:
            h = mae_cfg.get("IMG_HEIGHT", 256)
            w = mae_cfg.get("IMG_WIDTH", 256)
            input_size = (h, w)
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = tuple(input_size)

        in_chans = int(mae_cfg.get("IN_CHANS", cfg_model.get("PRE_PROCESSOR", {}).get("INPUT_DIM", 1)))
        self.in_chans = in_chans

        # [Changed] Early-convolution ConvStem (3x3, stride=1, padding=1) before stripe tokenization.
        self.stem_out_chans = int(mae_cfg.get("STEM_OUT_CHANS", 64))
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_chans, self.stem_out_chans, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.stem_out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stem_out_chans, self.stem_out_chans, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.stem_out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stem_out_chans, self.stem_out_chans, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.stem_out_chans),
            nn.ReLU(inplace=True),
        )

        use_batch_norm = mae_cfg.get("USE_BATCH_NORM", False)
        norm_layer = ChannelsLastBatchNorm1d if use_batch_norm else partial(nn.LayerNorm, eps=1e-6)

        pretrain_axis = str(mae_cfg.get("PRETRAIN_AXIS", "range_azimuth")).lower()
        if pretrain_axis not in ("range_azimuth", "azimuth_range"):
            raise ValueError(
                f"Unsupported PRETRAIN_AXIS='{pretrain_axis}'. Use 'range_azimuth' or 'azimuth_range'."
            )
        self.pretrain_axis = pretrain_axis
        self._transpose_to_pretrain = self.pretrain_axis == "azimuth_range"

        self.use_input_norm = bool(mae_cfg.get("USE_INPUT_NORM", True))
        self.input_norm = nn.BatchNorm2d(in_chans, affine=False) if self.use_input_norm else None

        # ConvStem application mode: "global" (whole image) or "per_token" (per stripe).
        conv_stem_mode = str(mae_cfg.get("CONV_STEM_MODE", "global")).lower().strip()
        if conv_stem_mode not in ("global", "per_token"):
            raise ValueError(f"CONV_STEM_MODE must be 'global' or 'per_token', got '{conv_stem_mode}'")
        self.conv_stem_mode = conv_stem_mode

        # Pretrained checkpoint path (optionally used for auto stripe-encoder inference).
        pretrain_path = mae_cfg.get(
            "PRETRAIN_PATH",
            str(Path(__file__).resolve().parents[3] / "RadarMAE.pth"),
        )
        if pretrain_path is None:
            self._pretrain_path = ""
        else:
            pretrain_path = str(pretrain_path).strip()
            if pretrain_path.lower() in ("", "none", "null"):
                self._pretrain_path = ""
            else:
                self._pretrain_path = os.path.expanduser(pretrain_path)

        if self.tokenizer == "patch":
            patch_size_for_sfp = max(patch_size)

            self.decoder_img_size = (
                self._ceil_to_multiple(int(self.input_size[0]), patch_size[0]),
                self._ceil_to_multiple(int(self.input_size[1]), patch_size[1]),
            )
            grid_h = self.decoder_img_size[0] // patch_size[0]
            grid_w = self.decoder_img_size[1] // patch_size[1]
            self.grid_hw = (grid_h, grid_w)
            self.num_patches = grid_h * grid_w
            self.patch_proj = nn.Conv2d(
                self.stem_out_chans,
                self.embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
            self.encoder_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, self.embed_dim), requires_grad=False
            )
            self._init_patch_pos_embed()

            self.blocks = nn.ModuleList(
                [
                    Block(self.embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for _ in range(depth)
                ]
            )
            self.norm = norm_layer(self.embed_dim)
            self.norm_h = None
            self.norm_v = None
            self.stripe_pool = None
            self.stripe_patch_width = (0, 0)
            self.stripe_patch_stride = (0, 0)
            self.num_stripe_tokens_hw = (0, 0)
            self.axis_embed = None
            self.stripe_encoder = None
            self.stripe_fuse_conv = None
            self.patch_embed = None
        else:
            stripe_pool = mae_cfg.get("STRIPE_POOL", patch_size)
            if isinstance(stripe_pool, (list, tuple)):
                if len(stripe_pool) != 2:
                    raise ValueError("STRIPE_POOL must be an int or a 2-tuple/list.")
                stripe_pool = (int(stripe_pool[0]), int(stripe_pool[1]))
            else:
                stripe_pool = (int(stripe_pool), int(stripe_pool))

            stripe_patch_width = mae_cfg.get("STRIPE_PATCH_WIDTH", 3)
            if isinstance(stripe_patch_width, (list, tuple)):
                if len(stripe_patch_width) != 2:
                    raise ValueError("STRIPE_PATCH_WIDTH must be an int or a 2-tuple/list.")
                stripe_patch_width = (int(stripe_patch_width[0]), int(stripe_patch_width[1]))
            else:
                stripe_patch_width = (int(stripe_patch_width), int(stripe_patch_width))

            stripe_patch_stride = mae_cfg.get("STRIPE_PATCH_STRIDE", stripe_patch_width)
            if isinstance(stripe_patch_stride, (list, tuple)):
                if len(stripe_patch_stride) != 2:
                    raise ValueError("STRIPE_PATCH_STRIDE must be an int or a 2-tuple/list.")
                stripe_patch_stride = (int(stripe_patch_stride[0]), int(stripe_patch_stride[1]))
            else:
                stripe_patch_stride = (int(stripe_patch_stride), int(stripe_patch_stride))

            if min(stripe_patch_width) <= 0:
                raise ValueError(f"STRIPE_PATCH_WIDTH must be positive, got {stripe_patch_width}.")
            if min(stripe_patch_stride) <= 0:
                raise ValueError(f"STRIPE_PATCH_STRIDE must be positive, got {stripe_patch_stride}.")

            in_h, in_w = int(self.input_size[0]), int(self.input_size[1])
            patch_h, patch_w = stripe_patch_width
            stride_h, stride_w = stripe_patch_stride
            if patch_h > in_h or patch_w > in_w:
                raise ValueError(
                    f"STRIPE_PATCH_WIDTH {stripe_patch_width} must not exceed IMG_SIZE {self.input_size}."
                )

            num_h_tokens = (in_h - patch_h) // stride_h + 1
            num_w_tokens = (in_w - patch_w) // stride_w + 1
            if num_h_tokens <= 0 or num_w_tokens <= 0:
                raise ValueError(
                    f"Invalid stripe patch setup: IMG_SIZE={self.input_size}, "
                    f"STRIPE_PATCH_WIDTH={stripe_patch_width}, STRIPE_PATCH_STRIDE={stripe_patch_stride}."
                )

            self.stripe_patch_width = stripe_patch_width
            self.stripe_patch_stride = stripe_patch_stride
            self.num_stripe_tokens_hw = (num_h_tokens, num_w_tokens)
            self.stripe_pool = stripe_pool
            patch_size_for_sfp = max(
                stripe_pool[0] * self.stripe_patch_stride[0],
                stripe_pool[1] * self.stripe_patch_stride[1],
            )

            if (num_h_tokens % stripe_pool[0] != 0) or (num_w_tokens % stripe_pool[1] != 0):
                raise ValueError(
                    f"Token counts {(num_h_tokens, num_w_tokens)} must be divisible by STRIPE_POOL {stripe_pool}."
                )

            self.patch_embed = None
            self.grid_hw = (num_h_tokens // stripe_pool[0], num_w_tokens // stripe_pool[1])

            stripe_encoder = str(mae_cfg.get("STRIPE_ENCODER", "unified")).lower().strip()
            if stripe_encoder in ("", "none"):
                stripe_encoder = "unified"
            if stripe_encoder != "unified":
                raise ValueError(
                    f"STRIPE_ENCODER must be 'unified' for TOKENIZER='stripe' in this project, got '{stripe_encoder}'."
                )
            self.stripe_encoder = "unified"

            use_lepe = bool(mae_cfg.get("USE_LEPE", True))
            lepe_kernel_size = int(mae_cfg.get("LEPE_KERNEL_SIZE", 3))

            # [Changed] Stripe embedding now consumes ConvStem output channels.
            self.stripe_embed_h = nn.Linear(self.stem_out_chans * self.input_size[1] * patch_h, self.embed_dim, bias=True)
            self.stripe_embed_v = nn.Linear(self.stem_out_chans * self.input_size[0] * patch_w, self.embed_dim, bias=True)

            # Positional embedding mode: 'PPE' or 'PE'
            self.pos_embed_mode = str(mae_cfg.get("POS_EMBED_MODE", "PPE")).upper()
            if self.pos_embed_mode not in ("PPE", "PE"):
                raise ValueError(
                    f"POS_EMBED_MODE must be 'PPE' (Physical) or 'PE' (Standard), got '{self.pos_embed_mode}'."
                )

            if self.pos_embed_mode == "PPE":
                # Physical Positional Encoding (default)
                self.geome_pe = PhysicalPositionalEncoding(self.embed_dim)
                if self._transpose_to_pretrain:
                    # (H, W) = (azimuth, range) for PRETRAIN_AXIS='azimuth_range'
                    phys_h = self._resize_physical_coords(self.arr_azimuth, num_h_tokens)
                    phys_v = self._resize_physical_coords(self.arr_range, num_w_tokens)
                else:
                    # (H, W) = (range, azimuth) for PRETRAIN_AXIS='range_azimuth'
                    phys_h = self._resize_physical_coords(self.arr_range, num_h_tokens)
                    phys_v = self._resize_physical_coords(self.arr_azimuth, num_w_tokens)
                self.register_buffer("phys_h", torch.from_numpy(phys_h))
                self.register_buffer("phys_v", torch.from_numpy(phys_v))
            else:
                # Standard Positional Encoding (MAE-style)
                self.geome_pe = StandardPositionalEncoding(self.embed_dim)
                # Register simple index-based coordinates
                idx_h = np.arange(num_h_tokens, dtype=np.float32)
                idx_v = np.arange(num_w_tokens, dtype=np.float32)
                self.register_buffer("phys_h", torch.from_numpy(idx_h))
                self.register_buffer("phys_v", torch.from_numpy(idx_v))

            self.axis_embed = nn.Parameter(torch.zeros(1, 2, self.embed_dim))
            nn.init.normal_(self.axis_embed, std=0.02)

            self.blocks = nn.ModuleList(
                [Block(self.embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)]
            )
            self.norm = norm_layer(self.embed_dim)
            self.norm_h = None
            self.norm_v = None
            # DirectStripeToBEV consumes 1D stripe tokens directly; no intermediate 2D stripe fusion is used.
            self.stripe_fuse_conv = None

        sfp_cfg = cfg_bb.get("SIMPLE_FPN", {})
        self.use_sfp = bool(sfp_cfg.get("ENABLED", True))
        sfp_scale_factors = sfp_cfg.get("SCALE_FACTORS", None)
        target_strides = sfp_cfg.get("TARGET_STRIDES", None)
        if sfp_scale_factors is not None:
            scale_factors = tuple(float(s) for s in sfp_scale_factors)
            self.sfp_target_strides = tuple(int(round(float(patch_size_for_sfp) / s)) for s in scale_factors)
        elif target_strides is not None:
            self.sfp_target_strides = tuple(int(s) for s in target_strides)
            scale_factors = tuple(float(patch_size_for_sfp) / float(s) for s in self.sfp_target_strides)
        else:
            scale_factors = (4.0, 2.0, 1.0, 0.5)
            self.sfp_target_strides = tuple(int(round(float(patch_size_for_sfp) / s)) for s in scale_factors)
        self.sfp_scales = scale_factors
        self.sfp_fuse = str(sfp_cfg.get("FUSE", "sum")).lower()
        self.sfp_out_channels = int(sfp_cfg.get("OUT_CHANNELS", self.embed_dim))
        sfp_norm = sfp_cfg.get("NORM", "LN")
        if self.use_sfp:
            self.sfp = SimpleFeaturePyramid(
                in_channels=self.embed_dim,
                out_channels=self.sfp_out_channels,
                scale_factors=scale_factors,
                norm=sfp_norm,
            )
            if self.sfp_out_channels == self.embed_dim:
                self.sfp_to_embed = nn.Identity()
            else:
                self.sfp_to_embed = nn.Conv2d(self.sfp_out_channels, self.embed_dim, kernel_size=1)
        else:
            self.sfp = None
            self.sfp_to_embed = nn.Identity()

        out_channels = cfg_bb.get("OUT_CHANNELS", self.embed_dim)
        if out_channels == self.embed_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)
        self.out_channels = out_channels
        self.num_bev_features = out_channels
        
        refine_cfg = bev_cfg.get("REFINE", {})
        self.use_refine = bool(refine_cfg.get("ENABLED", False))
        if self.use_refine:
            num_blocks = int(refine_cfg.get("NUM_BLOCKS", 2))
            hidden_channels = int(refine_cfg.get("HIDDEN_CHANNELS", self.out_channels))
            refine_type = str(refine_cfg.get("TYPE", "aspp_lite")).lower().strip()
            dilations = refine_cfg.get("DILATIONS", [1, 2, 3])
            if isinstance(dilations, int):
                dilations = [dilations]
            dilations = tuple(int(d) for d in dilations if int(d) > 0)
            if len(dilations) == 0:
                dilations = (1, 2, 3)
            self.bev_refine = BEVRefineNet(
                self.out_channels,
                hidden_channels,
                num_blocks,
                refine_type=refine_type,
                dilations=dilations,
            )
        else:
            self.bev_refine = nn.Identity()

        # BEV size from ROI
        if self.cfg_dataset_ver2:
            roi = cfg.DATASET.roi
            x_min, y_min, z_min, x_max, y_max, z_max = roi.xyz
            x_res, y_res, _ = roi.voxel_size
        else:
            roi = cfg.DATASET.RDR_SP_CUBE.ROI
            grid_size = cfg.DATASET.RDR_SP_CUBE.GRID_SIZE
            x_min, x_max = roi["x"]
            y_min, y_max = roi["y"]
            z_min, z_max = roi["z"]
            x_res = y_res = grid_size

        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.x_res, self.y_res = x_res, y_res
        self.bev_w = int(round((x_max - x_min) / x_res))
        self.bev_h = int(round((y_max - y_min) / y_res))

        # Polar coordinate extents (defaults derived from ROI)
        default_r_max = max(
            math.hypot(x_min, y_min),
            math.hypot(x_min, y_max),
            math.hypot(x_max, y_min),
            math.hypot(x_max, y_max),
        )
        if self.arr_range is not None:
            arr_range = self._flatten_nested_array(self.arr_range)
            self.arr_range = arr_range
            self.r_min, self.r_max = float(np.min(arr_range)), float(np.max(arr_range))
        else:
            polar_range = bev_cfg.get("RANGE", (0.0, default_r_max))
            self.r_min, self.r_max = float(polar_range[0]), float(polar_range[1])
        if self.arr_azimuth is not None:
            arr_azimuth = self._flatten_nested_array(self.arr_azimuth)
            self.arr_azimuth = arr_azimuth
            self.a_min, self.a_max = float(np.min(arr_azimuth)), float(np.max(arr_azimuth))
        else:
            az_range = bev_cfg.get("AZIMUTH", (-math.pi, math.pi))
            self.a_min, self.a_max = float(az_range[0]), float(az_range[1])
        self.eps = 1e-6

        # BEV grid (converted from Cartesian ROI)
        base_grid = self._create_base_grid(
            self.bev_h, self.bev_w, self.x_min, self.x_max, self.y_min, self.y_max,
            self.x_res, self.y_res, self.r_min, self.r_max, self.a_min, self.a_max
        )

        self.register_buffer("reference_points", base_grid)

        direct_cfg = bev_cfg.get("DIRECT_STRIPE", {})
        direct_heads = int(direct_cfg.get("NUM_HEADS", num_heads // 2 if num_heads > 4 else 4))
        direct_points = int(direct_cfg.get("NUM_POINTS", 4))
        direct_fuse = str(direct_cfg.get("FUSE", bev_cfg.get("AXIAL_FUSE", "mean"))).lower().strip()
        if direct_fuse not in ("sum", "mean"):
            raise ValueError(f"DIRECT_STRIPE.FUSE must be 'sum' or 'mean', got '{direct_fuse}'")
        # Backward-compat: kept for any external callers relying on _fuse_axial_bev.
        self.axial_fuse = direct_fuse
        if self.embed_dim % max(direct_heads, 1) != 0:
            raise ValueError(f"DIRECT_STRIPE.NUM_HEADS={direct_heads} must divide EMBED_DIM={self.embed_dim}")
        if self.tokenizer == "stripe":
            self.stripe_to_bev = DirectStripeToBEV(
                embed_dim=self.embed_dim,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                num_heads=direct_heads,
                num_points=direct_points,
                fuse=direct_fuse,
                align_corners=self.align_corners,
            )
            self.patch_to_bev = None
        else:
            self.stripe_to_bev = None
            self.patch_to_bev = DirectPatchToBEV(
                embed_dim=self.embed_dim,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                num_heads=direct_heads,
                num_points=direct_points,
                align_corners=self.align_corners,
            )

        if self.use_sfp and self.sfp_fuse not in ("sum", "mean", "concat"):
            raise ValueError(f"Unsupported SIMPLE_FPN.FUSE='{self.sfp_fuse}'")
        if self.use_sfp and self.sfp_fuse == "concat":
            self.sfp_fuse_conv = nn.Conv2d(
                self.out_channels * len(self.sfp_scales),
                self.out_channels,
                kernel_size=1,
            )
        else:
            self.sfp_fuse_conv = None

        self._load_pretrained_encoder()

    def _create_base_grid(self, h, w, x_min, x_max, y_min, y_max, x_res, y_res, r_min, r_max, a_min, a_max):
        # Create Reference Points (Geometric Prior)
        x_centers = torch.linspace(x_min + 0.5 * x_res, x_max - 0.5 * x_res, w)
        y_centers = torch.linspace(y_min + 0.5 * y_res, y_max - 0.5 * y_res, h)
        try:
            yv, xv = torch.meshgrid(y_centers, x_centers, indexing="ij")
        except TypeError:
            yv, xv = torch.meshgrid(y_centers, x_centers)

        r = torch.sqrt(xv ** 2 + yv ** 2)
        t = torch.atan2(-yv, xv)

        r_norm = (r - r_min) / max(r_max - r_min, self.eps)
        t_norm = (t - a_min) / max(a_max - a_min, self.eps)

        r_norm = r_norm * 2.0 - 1.0
        t_norm = t_norm * 2.0 - 1.0

        # Stack (Azimuth=x, Range=y order)
        grid = torch.stack([t_norm, r_norm], dim=-1).unsqueeze(0)
        # NOTE: Do not clamp here. Out-of-range coordinates should remain out-of-range so that
        # grid_sample(padding_mode='zeros') returns 0 instead of replicating border values.
        return grid

    def _fuse_axial_bev(self, bev_levels):
        if len(bev_levels) == 1:
            return bev_levels[0]
        if self.axial_fuse == "sum":
            return torch.stack(bev_levels, dim=0).sum(0)
        return torch.stack(bev_levels, dim=0).mean(0)

    @staticmethod
    def _flatten_nested_array(arr) -> np.ndarray:
        if isinstance(arr, np.ndarray):
            return arr.astype(float).reshape(-1)
        if isinstance(arr, (list, tuple)):
            pieces = []
            for a in arr:
                pieces.append(np.array(a, dtype=float).reshape(-1))
            return np.concatenate(pieces)
        return np.array(arr, dtype=float).reshape(-1)

    @staticmethod
    def _resize_physical_coords(coords, target_len: int) -> np.ndarray:
        coords_arr = np.array(coords, dtype=np.float32).reshape(-1)
        if coords_arr.size == 0:
            raise ValueError("physical coordinates are empty")
        if target_len <= 0:
            raise ValueError(f"target_len must be positive, got {target_len}")
        if coords_arr.size == target_len:
            return coords_arr
        if coords_arr.size == 1:
            return np.full((target_len,), float(coords_arr[0]), dtype=np.float32)

        src_idx = np.linspace(0.0, 1.0, coords_arr.size, dtype=np.float32)
        dst_idx = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
        return np.interp(dst_idx, src_idx, coords_arr).astype(np.float32)

    def _load_physical_bins(self):
        path_info = Path(__file__).resolve().parents[2] / "resources" / "info_arr.mat"
        if not path_info.is_file():
            raise FileNotFoundError(f"* info_arr.mat not found at {path_info}")
        vals = loadmat(str(path_info))
        arr_range = vals["arrRange"].reshape(-1)
        arr_azimuth = vals["arrAzimuth"].reshape(-1) * np.pi / 180.0
        return arr_range, arr_azimuth

    @staticmethod
    def _ceil_to_multiple(value: int, multiple: int) -> int:
        if multiple <= 0:
            raise ValueError(f"multiple must be positive, got {multiple}")
        return int(math.ceil(float(value) / float(multiple)) * multiple)

    def _pad_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x to be 4D (B,C,H,W), got {tuple(x.shape)}")
        target_h, target_w = self.decoder_img_size
        pad_h = int(target_h) - int(x.shape[-2])
        pad_w = int(target_w) - int(x.shape[-1])
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"Cannot pad input shape {tuple(x.shape[-2:])} to smaller target {(target_h, target_w)}"
            )
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even for sincos PE, got {embed_dim}")
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.0
        omega = 1.0 / (10000.0 ** omega)
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    @classmethod
    def _get_2d_sincos_pos_embed(cls, embed_dim: int, grid_size: Tuple[int, int]) -> np.ndarray:
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even for 2D sincos PE, got {embed_dim}")
        grid_h_size, grid_w_size = int(grid_size[0]), int(grid_size[1])
        grid_h = np.arange(grid_h_size, dtype=np.float32)
        grid_w = np.arange(grid_w_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0).reshape([2, 1, grid_h_size, grid_w_size])
        emb_h = cls._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = cls._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        return np.concatenate([emb_h, emb_w], axis=1)

    def _init_patch_pos_embed(self) -> None:
        pos_embed = self._get_2d_sincos_pos_embed(self.embed_dim, self.grid_hw)
        self.encoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _patch_tokenize_per_token(self, x: torch.Tensor) -> torch.Tensor:
        if self.tokenizer != "patch":
            raise RuntimeError("patch tokenization is only available when TOKENIZER=patch")
        if x.ndim != 4:
            raise ValueError(f"Expected x to be 4D (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_chans:
            raise ValueError(f"Expected in_chans={self.in_chans}, got C={x.shape[1]}")

        x = self._pad_nchw(x)
        bsz, ch, _, _ = x.shape
        patch_h, patch_w = self.patch_size
        patches = F.unfold(x, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        num_patches = patches.shape[-1]
        if num_patches != self.num_patches:
            raise RuntimeError(f"Patch count mismatch: got {num_patches}, expected {self.num_patches}")
        patches = patches.view(bsz, ch, patch_h, patch_w, num_patches)
        patches = patches.permute(0, 4, 1, 2, 3).contiguous()
        patches = patches.view(bsz * num_patches, ch, patch_h, patch_w)
        patches = self.conv_stem(patches)
        tokens = self.patch_proj(patches)
        return tokens.view(bsz, num_patches, -1)

    def _stripe_tokenize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer != "stripe":
            raise RuntimeError("stripe tokenization is only available when TOKENIZER=stripe")
        if x.ndim != 4:
            raise ValueError(f"Expected x to be 4D (B,C,H,W), got {tuple(x.shape)}")
        b, c, h, w = x.shape
        # [Changed] _stripe_tokenize input channel check follows ConvStem output channels.
        if c != self.stem_out_chans:
            raise ValueError(f"Expected C={self.stem_out_chans}, got C={c}")
        if (h, w) != tuple(self.input_size):
            raise ValueError(f"Expected input size {self.input_size}, got {(h, w)}")

        patch_h, patch_w = self.stripe_patch_width
        stride_h, stride_w = self.stripe_patch_stride

        # Horizontal-axis stream:
        # - patch shape: (patch_h x W), e.g., 3x256 when patch_h=3 and W=256
        # - stride: move only along H by stride_h
        x_h = F.unfold(x, kernel_size=(patch_h, w), stride=(stride_h, 1))  # (B, C*patch_h*W, L_h)
        x_h = x_h.view(b, c, patch_h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
        x_h = x_h.view(b, -1, c * patch_h * w)  # (B, L_h, C*(patch_h*W))

        # Vertical-axis stream:
        # - patch shape: (patch_w x H), e.g., 3x107 when patch_w=3 and H=107
        # - stride: move only along W by stride_w
        x_v = F.unfold(x, kernel_size=(h, patch_w), stride=(1, stride_w))  # (B, C*H*patch_w, L_w)
        x_v = x_v.view(b, c, h, patch_w, -1).permute(0, 4, 1, 3, 2).contiguous()
        x_v = x_v.view(b, -1, c * patch_w * h)  # (B, L_w, C*(patch_w*H))

        expected_h, expected_w = self.num_stripe_tokens_hw
        if x_h.shape[1] != expected_h or x_v.shape[1] != expected_w:
            raise ValueError(
                f"Unexpected stripe token lengths: got ({x_h.shape[1]}, {x_v.shape[1]}), "
                f"expected {self.num_stripe_tokens_hw}."
            )

        h_tok = self.stripe_embed_h(x_h)
        v_tok = self.stripe_embed_v(x_v)
        return h_tok, v_tok

    def _stripe_tokenize_per_token(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply conv_stem independently to each stripe token, eliminating inter-token information leakage.
        x: (B, in_chans, H, W) — raw input prior to conv_stem.
        """
        if self.tokenizer != "stripe":
            raise RuntimeError("stripe tokenization is only available when TOKENIZER=stripe")
        if x.ndim != 4:
            raise ValueError(f"Expected x to be 4D (B,C,H,W), got {tuple(x.shape)}")
        bsz, ch, h, w = x.shape
        if ch != self.in_chans:
            raise ValueError(f"Expected C={self.in_chans}, got C={ch}")
        if (h, w) != tuple(self.input_size):
            raise ValueError(f"Expected input size {self.input_size}, got {(h, w)}")

        patch_h, patch_w = self.stripe_patch_width
        stride_h, stride_w = self.stripe_patch_stride

        # ── H-stripe: (B, num_h, in_chans, patch_h, W) ──
        x_h = F.unfold(x, kernel_size=(patch_h, w), stride=(stride_h, 1))
        num_h = x_h.shape[-1]
        x_h = x_h.view(bsz, ch, patch_h, w, num_h).permute(0, 4, 1, 2, 3).contiguous()
        x_h = x_h.view(bsz * num_h, ch, patch_h, w)         # (B*num_h, C, patch_h, W)
        x_h = self.conv_stem(x_h)                             # (B*num_h, stem_C, patch_h, W)
        x_h = x_h.view(bsz, num_h, -1)                       # (B, num_h, stem_C*patch_h*W)
        h_tok = self.stripe_embed_h(x_h)                      # (B, num_h_tokens, embed_dim)

        # ── V-stripe: (B, num_v, in_chans, H, patch_w) ──
        x_v = F.unfold(x, kernel_size=(h, patch_w), stride=(1, stride_w))
        num_v = x_v.shape[-1]
        # NOTE: V-stripes enter conv_stem as (patch_w, H), i.e. transposed w.r.t.
        # RadarMAE pretraining (models_radarmae.py uses (H, patch_w)). The paper
        # results use this layout; switching to the pretraining layout lowered
        # K-Radar Sedan BEV AP@0.3 from 64.3 to 54.7 (seed 2025).
        x_v = x_v.view(bsz, ch, h, patch_w, num_v).permute(0, 4, 1, 3, 2).contiguous()
        x_v = x_v.view(bsz * num_v, ch, patch_w, h)          # (B*num_v, C, patch_w, H)
        x_v = self.conv_stem(x_v)                             # (B*num_v, stem_C, patch_w, H)
        x_v = x_v.view(bsz, num_v, -1)                       # (B, num_v, stem_C*patch_w*H)
        v_tok = self.stripe_embed_v(x_v)                      # (B, num_v_tokens, embed_dim)

        expected_h, expected_w = self.num_stripe_tokens_hw
        if h_tok.shape[1] != expected_h or v_tok.shape[1] != expected_w:
            raise ValueError(
                f"Token count mismatch: expected ({expected_h}, {expected_w}), "
                f"got ({h_tok.shape[1]}, {v_tok.shape[1]})"
            )
        return h_tok, v_tok

    @staticmethod
    def _pool_tokens_1d(x: torch.Tensor, stride: int) -> torch.Tensor:
        if stride <= 1:
            return x
        if x.ndim != 3:
            raise ValueError(f"Expected x to be 3D (B,L,C), got {tuple(x.shape)}")
        b, l, c = x.shape
        if l % stride != 0:
            raise ValueError(f"Token length L={l} must be divisible by stride={stride}")
        return x.view(b, l // stride, stride, c).mean(dim=2)

    def _resize_stripe_embed_weight(self, key: str, weight: torch.Tensor) -> torch.Tensor:
        if self.tokenizer != "stripe":
            raise ValueError("stripe embed resize is only valid for TOKENIZER=stripe")
        if key.endswith("stripe_embed_h.weight"):
            target = self.stripe_embed_h.weight
        elif key.endswith("stripe_embed_v.weight"):
            target = self.stripe_embed_v.weight
        else:
            raise ValueError(f"Unsupported stripe embed key: {key}")

        if weight.shape == target.shape:
            return weight
        if weight.ndim != 2 or target.ndim != 2:
            raise ValueError("unexpected stripe embed weight shape")
        if weight.shape[0] != target.shape[0]:
            raise ValueError("embed_dim mismatch")

        # Stripe embeddings operate on the ConvStem output, not raw input channels.
        # Keep feature channels separated and resize only along the spatial axis.
        channel_groups = None
        for cand in (getattr(self, "stem_out_chans", None), self.in_chans):
            if cand is None or cand <= 0:
                continue
            if (weight.shape[1] % cand == 0) and (target.shape[1] % cand == 0):
                channel_groups = int(cand)
                break
        if channel_groups is None:
            raise ValueError("unable to infer stripe embedding channel groups for resize")

        axis_len_src = weight.shape[1] // channel_groups
        axis_len_tgt = target.shape[1] // channel_groups

        w = weight.view(weight.shape[0], channel_groups, axis_len_src)  # (D, C, L)
        w = w.reshape(weight.shape[0] * channel_groups, 1, axis_len_src)
        w = F.interpolate(w, size=axis_len_tgt, mode="linear", align_corners=True)
        w = w.reshape(weight.shape[0], channel_groups, axis_len_tgt)
        return w.reshape(target.shape[0], target.shape[1])

    def _resize_patch_embed_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if self.tokenizer != "patch":
            raise ValueError("patch projection resize is only valid for TOKENIZER=patch")
        target = self.patch_proj.weight
        if weight.shape == target.shape:
            return weight
        if weight.shape[1] != target.shape[1]:
            raise ValueError(f"channel mismatch")
        if weight.dim() != 4:
            raise ValueError(f"unexpected shape")

        tgt_hw = target.shape[-2:]
        src_hw = weight.shape[-2:]
        resized = weight.reshape(weight.shape[0] * weight.shape[1], 1, src_hw[0], src_hw[1])
        resized = F.interpolate(resized, size=tgt_hw, mode="bilinear", align_corners=True)
        resized = resized.reshape(target.shape[0], target.shape[1], tgt_hw[0], tgt_hw[1])
        return resized

    def _load_pretrained_encoder(self, *, state: Optional[dict] = None) -> None:
        path = self._pretrain_path
        if state is None:
            if not path or not os.path.isfile(path):
                return
            # This project stores argparse.Namespace training metadata together
            # with the weights. PyTorch 2.6+ defaults to weights_only=True,
            # which rejects that trusted checkpoint metadata.
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            if not isinstance(state, dict):
                return

        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module.") :]: v for k, v in state.items()}

        if self.tokenizer == "stripe":
            allowed_prefix = ("conv_stem", "stripe_embed", "axis_embed", "geome_pe", "blocks", "norm")
        else:
            allowed_prefix = ("conv_stem", "patch_proj", "patch_embed", "encoder_pos_embed", "blocks", "norm")
        if self.input_norm is not None:
            allowed_prefix = ("input_norm",) + allowed_prefix

        model_state = self.state_dict()
        filtered = {}
        for k, v in state.items():
            if not k.startswith(allowed_prefix):
                continue
            if self.tokenizer == "patch" and k.startswith("patch_embed.proj."):
                k = "patch_proj." + k[len("patch_embed.proj.") :]
            if k not in model_state or not hasattr(v, "shape"):
                continue
            tgt = model_state[k]
            if v.shape == tgt.shape:
                filtered[k] = v
                continue

            if self.tokenizer == "patch" and k == "patch_proj.weight":
                try:
                    v2 = self._resize_patch_embed_weight(v)
                except ValueError:
                    continue
                if v2.shape == tgt.shape:
                    filtered[k] = v2
                continue

            if self.tokenizer == "stripe":
                if k in ("stripe_embed_h.weight", "stripe_embed_v.weight"):
                    try:
                        v2 = self._resize_stripe_embed_weight(k, v)
                    except ValueError:
                        continue
                    if v2.shape == tgt.shape:
                        filtered[k] = v2
                    continue

        msg = self.load_state_dict(filtered, strict=False)
        print(
            f"* MAEPolarBEVBackbone: loaded pretrained encoder from {path} "
            f"(tokenizer={self.tokenizer}, stripe_encoder={getattr(self, 'stripe_encoder', None)}, "
            f"loaded={len(filtered)}, missing={len(msg.missing_keys)})"
        )

    def forward(self, batch: dict) -> dict:
        polar_map = batch[self.input_key]
        bev_projector = self.stripe_to_bev if self.tokenizer == "stripe" else self.patch_to_bev
        device = bev_projector.bev_query.device
        polar_map = polar_map.to(device)
        if polar_map.dim() == 3:
            polar_map = polar_map.unsqueeze(1)

        if self._transpose_to_pretrain:
            polar_map = polar_map.transpose(-2, -1)

        if polar_map.shape[-2:] != self.input_size:
            polar_map = F.interpolate(
                polar_map, size=self.input_size, mode="bilinear", align_corners=self.align_corners
            )

        if self.input_norm is not None:
            polar_map = self.input_norm(polar_map)

        if self.tokenizer == "stripe":
            # Branch by ConvStem application mode.
            if self.conv_stem_mode == "per_token":
                # Apply conv_stem independently to each stripe token (input_norm already applied).
                h_tok, v_tok = self._stripe_tokenize_per_token(polar_map)
            else:
                # Global mode (default): apply conv_stem to the whole image.
                polar_map = self.conv_stem(polar_map)
                h_tok, v_tok = self._stripe_tokenize(polar_map)
            if self.axis_embed is None or self.norm is None:
                raise RuntimeError("Unified stripe encoder is required but axis_embed/norm is not initialized.")
            pe_h = self.geome_pe(self.phys_h).to(device=h_tok.device, dtype=h_tok.dtype)
            pe_v = self.geome_pe(self.phys_v).to(device=v_tok.device, dtype=v_tok.dtype)
            axis_h = self.axis_embed[:, 0:1, :].to(device=h_tok.device, dtype=h_tok.dtype)
            axis_v = self.axis_embed[:, 1:2, :].to(device=v_tok.device, dtype=v_tok.dtype)

            h_tok = h_tok + pe_h + axis_h
            v_tok = v_tok + pe_v + axis_v

            h_len = int(h_tok.shape[1])
            w_len = int(v_tok.shape[1])
            x = torch.cat([h_tok, v_tok], dim=1)  # (B, H+W, C)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

            h_tok = x[:, :h_len, :]
            v_tok = x[:, h_len : h_len + w_len, :]

            pool_h, pool_w = self.stripe_pool
            h_tok_ds = self._pool_tokens_1d(h_tok, pool_h)
            v_tok_ds = self._pool_tokens_1d(v_tok, pool_w)

            if self._transpose_to_pretrain:
                # (H,W) = (azimuth, range) after transpose -> range tokens are v_tok_ds, azimuth tokens are h_tok_ds.
                azim_tok_ds = h_tok_ds
                range_tok_ds = v_tok_ds
            else:
                # (H,W) = (range, azimuth) -> range tokens are h_tok_ds, azimuth tokens are v_tok_ds.
                range_tok_ds = h_tok_ds
                azim_tok_ds = v_tok_ds

            all_tok_ds = torch.cat([range_tok_ds, azim_tok_ds], dim=1)
            bev_level = self.stripe_to_bev(
                all_tok_ds,
                num_range_tokens=range_tok_ds.shape[1],
                num_azim_tokens=azim_tok_ds.shape[1],
                polar_grid=self.reference_points,
                roi_xy=(self.x_min, self.x_max, self.y_min, self.y_max),
                polar_extents=(self.r_min, self.r_max, self.a_min, self.a_max),
            )  # (B, H_bev, W_bev, C)
        else:
            if self.conv_stem_mode == "per_token":
                patch_tok = self._patch_tokenize_per_token(polar_map)
            else:
                patch_map = self.conv_stem(polar_map)
                patch_map = self._pad_nchw(patch_map)
                patch_tok = self.patch_proj(patch_map).flatten(2).transpose(1, 2)
            if patch_tok.shape[1] != self.num_patches:
                raise RuntimeError(f"Patch token count mismatch: got {patch_tok.shape[1]}, expected {self.num_patches}")

            pos_embed = self.encoder_pos_embed.to(device=patch_tok.device, dtype=patch_tok.dtype)
            patch_tok = patch_tok + pos_embed
            for blk in self.blocks:
                patch_tok = blk(patch_tok)
            patch_tok = self.norm(patch_tok)

            bev_level = self.patch_to_bev(
                patch_tok,
                grid_hw=self.grid_hw,
                polar_grid=self.reference_points,
                pretrain_axis=self.pretrain_axis,
                roi_xy=(self.x_min, self.x_max, self.y_min, self.y_max),
                polar_extents=(self.r_min, self.r_max, self.a_min, self.a_max),
            )  # (B, H_bev, W_bev, C)

        # 1) Build BEV via deformable cross-attention with BEV queries.
        bev_base = bev_level.permute(0, 3, 1, 2).contiguous()  # (B, C, H_bev, W_bev)

        # 2) Apply FPN in BEV space, then fuse to a single feature map for the detection head.
        if self.use_sfp:
            base_hw = bev_base.shape[-2:]
            pyramid = list(self.sfp(bev_base))
            pyramid = [self.sfp_to_embed(pf) for pf in pyramid]

            bev_feats = []
            for pf in pyramid:
                if pf.shape[-2:] != base_hw:
                    pf = F.interpolate(
                        pf, size=base_hw, mode="bilinear", align_corners=self.align_corners
                    )
                pf = self.proj(pf)
                bev_feats.append(pf)

            if len(bev_feats) == 1:
                bev_feat = bev_feats[0]
            else:
                if self.sfp_fuse == "sum":
                    bev_feat = torch.stack(bev_feats, dim=0).sum(0)
                elif self.sfp_fuse == "mean":
                    bev_feat = torch.stack(bev_feats, dim=0).mean(0)
                else:
                    bev_feat = torch.cat(bev_feats, dim=1)
                    bev_feat = self.sfp_fuse_conv(bev_feat)
        else:
            bev_feats = None
            bev_feat = self.proj(bev_base)

        bev_feat = self.bev_refine(bev_feat)

        if self.use_sfp:
            batch["bev_feat_pyramid"] = tuple(bev_feats)

        batch["bev_feat"] = bev_feat    # (B, C, H_bev, W_bev)

        return batch
