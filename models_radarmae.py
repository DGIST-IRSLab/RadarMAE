# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
# RadarMAE encoder/decoder.
# Adapted from facebookresearch/mae (https://github.com/facebookresearch/mae).
# Adds axial / stripe tokenization, physical positional encoding, weighted-block
# masking, and axis-specific distribution-matching reconstruction losses on top
# of the original MaskedAutoencoderViT.
# --------------------------------------------------------
from functools import partial
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class ChannelsLastBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.transpose(1, 2)
        x = super().forward(x)
        return x.transpose(1, 2)


class PatchGridMetadata(nn.Module):
    def __init__(self, patch_size: Tuple[int, int], grid_size: Tuple[int, int]) -> None:
        super().__init__()
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.num_patches = int(self.grid_size[0] * self.grid_size[1])


class PhysicalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, L: int = 10) -> None:
        super().__init__()
        self.L = int(L)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.L, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        device = coords.device
        freqs = torch.pow(2.0, torch.arange(self.L, device=device).float()) * np.pi
        x = coords.unsqueeze(-1) * freqs.unsqueeze(0)
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.mlp(enc).unsqueeze(0)


class AxialDeformCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_h: int,
        out_w: int,
        num_heads: int = 4,
        num_points: int = 4,
        fuse: str = "mean",
        align_corners: bool = True,
        reference_points: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.num_heads = int(num_heads)
        self.num_points = int(num_points)
        self.align_corners = bool(align_corners)
        self.fuse = str(fuse).lower().strip()
        if self.fuse not in ("sum", "mean"):
            raise ValueError(f"fuse must be 'sum' or 'mean', got '{self.fuse}'")

        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.polar_query = nn.Parameter(torch.zeros(1, self.out_h, self.out_w, self.embed_dim))
        self.sampling_offsets = nn.Linear(self.embed_dim, self.num_heads * self.num_points * 2)
        self.attention_weights = nn.Linear(self.embed_dim, self.num_heads * self.num_points * 2)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)

        if reference_points is None:
            reference_points = self._make_uniform_reference_grid()
        if reference_points.shape != (1, self.out_h, self.out_w, 2):
            raise ValueError(
                f"reference_points must be (1,{self.out_h},{self.out_w},2), got {tuple(reference_points.shape)}"
            )
        self.register_buffer("reference_points", reference_points.to(dtype=torch.float32), persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.polar_query, std=0.02)
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.value_proj.bias is not None:
            nn.init.constant_(self.value_proj.bias, 0.0)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0.0)

    def _make_uniform_reference_grid(self) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, self.out_h, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, self.out_w, dtype=torch.float32)
        try:
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        except TypeError:
            yy, xx = torch.meshgrid(ys, xs)
        return torch.stack([xx, yy], dim=-1).unsqueeze(0)

    @staticmethod
    def _norm_to_index(coord_norm: torch.Tensor, length: int) -> torch.Tensor:
        if length <= 1:
            return torch.zeros_like(coord_norm)
        return (coord_norm + 1.0) * 0.5 * float(length - 1)

    def _sample_from_bank(self, bank: torch.Tensor, idx_norm: torch.Tensor) -> torch.Tensor:
        b_heads, c_head, _, _ = bank.shape
        loc = idx_norm.permute(0, 3, 1, 2, 4).flatten(0, 1)
        loc = loc.reshape(b_heads, self.out_h, self.out_w * self.num_points, 1)
        zeros = torch.zeros_like(loc)
        grid = torch.cat([zeros, loc], dim=-1)
        bank_dtype = bank.dtype
        bank = bank.contiguous()
        grid = grid.contiguous()
        with torch.cuda.amp.autocast(enabled=False):
            with torch.backends.cudnn.flags(enabled=False):
                sampled = F.grid_sample(
                    bank.float(),
                    grid.float(),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=self.align_corners,
                )
        sampled = sampled.to(dtype=bank_dtype)
        sampled = sampled.view(-1, self.num_heads, c_head, self.out_h, self.out_w, self.num_points)
        sampled = sampled.permute(0, 3, 4, 1, 5, 2).contiguous()
        return sampled

    def forward(self, all_tok: torch.Tensor, num_h_tokens: int, num_v_tokens: int) -> torch.Tensor:
        if all_tok.ndim != 3:
            raise ValueError(f"Expected all_tok to be (B,L,C), got {tuple(all_tok.shape)}")
        bsz, len_all, dim = all_tok.shape
        if dim != self.embed_dim:
            raise ValueError(f"Expected embed dim {self.embed_dim}, got {dim}")

        l_h = int(num_h_tokens)
        l_v = int(num_v_tokens)
        if l_h <= 0 or l_v <= 0 or (l_h + l_v != len_all):
            raise ValueError(
                f"Invalid token lengths: len_all={len_all}, num_h_tokens={l_h}, num_v_tokens={l_v}"
            )

        query = self.polar_query.expand(bsz, -1, -1, -1)
        offsets = self.sampling_offsets(query).view(
            bsz, self.out_h, self.out_w, self.num_heads, self.num_points, 2
        )
        weights = self.attention_weights(query).view(
            bsz, self.out_h, self.out_w, self.num_heads, self.num_points, 2
        )
        w_h = F.softmax(weights[..., 0], dim=-1)
        w_v = F.softmax(weights[..., 1], dim=-1)

        ref = self.reference_points.to(device=query.device, dtype=query.dtype).unsqueeze(3).unsqueeze(3)
        sampling = ref + offsets
        x_loc = sampling[..., 0]
        y_loc = sampling[..., 1]

        valid_h = (y_loc >= -1.0) & (y_loc <= 1.0)
        valid_v = (x_loc >= -1.0) & (x_loc <= 1.0)
        w_h = w_h * valid_h.to(w_h.dtype)
        w_v = w_v * valid_v.to(w_v.dtype)
        w_h = w_h / w_h.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        w_v = w_v / w_v.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

        val_all = self.value_proj(all_tok).view(bsz, len_all, self.num_heads, self.head_dim)
        val_all = val_all.permute(0, 2, 3, 1).reshape(bsz * self.num_heads, self.head_dim, len_all, 1)

        idx_h = self._norm_to_index(y_loc, l_h)
        idx_v = self._norm_to_index(x_loc, l_v) + float(l_h)
        denom = float(max(len_all - 1, 1))
        idx_h_norm = idx_h / denom * 2.0 - 1.0
        idx_v_norm = idx_v / denom * 2.0 - 1.0

        sampled_h = self._sample_from_bank(val_all, idx_h_norm)
        sampled_v = self._sample_from_bank(val_all, idx_v_norm)

        out_h = (sampled_h * w_h.unsqueeze(-1)).sum(dim=4)
        out_v = (sampled_v * w_v.unsqueeze(-1)).sum(dim=4)
        out_h = out_h.reshape(bsz, self.out_h, self.out_w, self.embed_dim)
        out_v = out_v.reshape(bsz, self.out_h, self.out_w, self.embed_dim)

        if self.fuse == "sum":
            out = out_h + out_v
        else:
            out = 0.5 * (out_h + out_v)
        return self.output_proj(out)


class MaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        recon_loss_type: str = "mse",
        loss_weighting: str = "none",
        masking_mode: str = "random",
        stem_out_chans=64,
        stripe_patch_width=(2, 3),
        stripe_patch_stride=(1, 1),
        deform_num_heads=4,
        deform_num_points=4,
        arr_range: Optional[np.ndarray] = None,
        arr_azimuth: Optional[np.ndarray] = None,
        pretrain_axis: str = "azimuth_range",
        tokenization_mode: str = "axial",
        use_input_norm: bool = True,
        use_batch_norm: bool = False,
        enable_dist_loss: bool = False,
        lambda_multipath: float = 0.1,
        lambda_sidelobs: float = 0.1,
        dist_loss_axis_routing: str = "physical",
        pos_embed_mode: str = "ppe",
        conv_stem_mode: str = "global",
        patch_block_min_size: int = 3,
        patch_block_max_size: int = 5,
        patch_block_max_steps: int = 40,
    ):
        super().__init__()

        self.embed_dim = int(embed_dim)
        self.in_chans = int(in_chans)
        self.img_size = img_size if isinstance(img_size, tuple) else (int(img_size), int(img_size))
        self.pretrain_axis = str(pretrain_axis).lower().strip()
        if self.pretrain_axis not in ("azimuth_range", "range_azimuth"):
            raise ValueError(
                f"Unsupported pretrain_axis='{pretrain_axis}'. Use 'azimuth_range' or 'range_azimuth'."
            )

        self.tokenization_mode = str(tokenization_mode).lower().strip()
        if self.tokenization_mode not in ("axial", "patch"):
            raise ValueError(
                f"Unsupported tokenization_mode='{tokenization_mode}'. Use 'axial' or 'patch'."
            )

        if isinstance(patch_size, (list, tuple)):
            self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        else:
            self.patch_size = (int(patch_size), int(patch_size))
        self.decoder_img_size = (
            self._ceil_to_multiple(self.img_size[0], self.patch_size[0]),
            self._ceil_to_multiple(self.img_size[1], self.patch_size[1]),
        )
        self.patch_embed = PatchGridMetadata(
            self.patch_size,
            (self.decoder_img_size[0] // self.patch_size[0], self.decoder_img_size[1] // self.patch_size[1]),
        )
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        self.use_input_norm = bool(use_input_norm)
        self.input_norm = nn.BatchNorm2d(self.in_chans, affine=False) if self.use_input_norm else None
        self.conv_stem = nn.Sequential(
            nn.Conv2d(self.in_chans, int(stem_out_chans), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(stem_out_chans)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(stem_out_chans), int(stem_out_chans), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(stem_out_chans)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(stem_out_chans), int(stem_out_chans), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(stem_out_chans)),
            nn.ReLU(inplace=True),
        )
        self.stem_out_chans = int(stem_out_chans)

        # Conv stem mode (validated for both patch and axial modes).
        # - 'global'    : apply conv_stem to the full image before tokenization.
        #                 Cheap, but the cumulative 7x7 receptive field crosses
        #                 patch/stripe boundaries and leaks neighbour content
        #                 into the masked positions, letting MAE solve the task
        #                 by trivial copy.
        # - 'per_token' : unfold first, apply conv_stem to each patch/stripe
        #                 in isolation (zero-padded boundaries).  No leakage.
        self.conv_stem_mode = str(conv_stem_mode).lower().strip()
        if self.conv_stem_mode not in ("global", "per_token"):
            raise ValueError(
                f"conv_stem_mode must be 'global' or 'per_token', got '{self.conv_stem_mode}'."
            )

        if self.tokenization_mode == "patch":
            # ── Standard 2D patch tokenization (like original MAE) ──
            self.patch_proj = nn.Conv2d(
                self.stem_out_chans, embed_dim,
                kernel_size=self.patch_size, stride=self.patch_size,
            )
            self.encoder_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
            )
            # Axial-specific attributes (set to zero/None for compatibility)
            self.stripe_patch_width = (0, 0)
            self.stripe_patch_stride = (0, 0)
            self.num_h_tokens = 0
            self.num_v_tokens = 0
            self.num_axial_tokens = 0
            self.num_stripe_tokens_hw = (0, 0)
            self.num_range_tokens = 0
            self.num_azimuth_tokens = 0
            # pos_embed_mode is irrelevant under patch tokenization (the encoder
            # uses a fixed 2D sincos PE), but store it for arg round-tripping.
            self.pos_embed_mode = str(pos_embed_mode).lower().strip()
        else:
            # ── Axial stripe tokenization ──
            if isinstance(stripe_patch_width, (list, tuple)):
                self.stripe_patch_width = (int(stripe_patch_width[0]), int(stripe_patch_width[1]))
            else:
                self.stripe_patch_width = (int(stripe_patch_width), int(stripe_patch_width))
            if isinstance(stripe_patch_stride, (list, tuple)):
                self.stripe_patch_stride = (int(stripe_patch_stride[0]), int(stripe_patch_stride[1]))
            else:
                self.stripe_patch_stride = (int(stripe_patch_stride), int(stripe_patch_stride))

            patch_h, patch_w = self.stripe_patch_width
            stride_h, stride_w = self.stripe_patch_stride
            in_h, in_w = self.img_size
            self.num_h_tokens = (in_h - patch_h) // stride_h + 1
            self.num_v_tokens = (in_w - patch_w) // stride_w + 1
            if self.num_h_tokens <= 0 or self.num_v_tokens <= 0:
                raise ValueError(
                    f"Invalid stripe setup for img_size={self.img_size}, "
                    f"stripe_patch_width={self.stripe_patch_width}, stripe_patch_stride={self.stripe_patch_stride}"
                )
            self.num_stripe_tokens_hw = (self.num_h_tokens, self.num_v_tokens)
            self.num_axial_tokens = self.num_h_tokens + self.num_v_tokens
            if self.pretrain_axis == "azimuth_range":
                self.num_azimuth_tokens = self.num_h_tokens
                self.num_range_tokens = self.num_v_tokens
            else:
                self.num_range_tokens = self.num_h_tokens
                self.num_azimuth_tokens = self.num_v_tokens

            self.stripe_embed_h = nn.Linear(self.stem_out_chans * in_w * patch_h, embed_dim, bias=True)
            self.stripe_embed_v = nn.Linear(self.stem_out_chans * in_h * patch_w, embed_dim, bias=True)

            # Positional embedding mode: 'ppe' (learnable MLP) or 'pe' (fixed sincos)
            self.pos_embed_mode = str(pos_embed_mode).lower().strip()
            if self.pos_embed_mode not in ("ppe", "pe"):
                raise ValueError(f"pos_embed_mode must be 'ppe' or 'pe', got '{self.pos_embed_mode}'.")

            if self.pos_embed_mode == "ppe":
                self.geome_pe = PhysicalPositionalEncoding(embed_dim)
            else:
                # PE mode: fixed sincos embedding (filled in by initialize_weights).
                self.register_buffer("encoder_pos_embed_h", torch.zeros(1, self.num_h_tokens, embed_dim))
                self.register_buffer("encoder_pos_embed_v", torch.zeros(1, self.num_v_tokens, embed_dim))

            self.axis_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))

        if use_batch_norm:
            encoder_norm_layer = ChannelsLastBatchNorm1d
        else:
            encoder_norm_layer = norm_layer
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=encoder_norm_layer) for _ in range(depth)]
        )
        self.norm = encoder_norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        if self.tokenization_mode == "patch":
            # Standard single mask token for decoder
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        else:
            # Axial: physical coords + deform cross attention
            pixel_h_coords, pixel_w_coords, token_h_coords, token_v_coords = self._build_coordinate_sets(
                arr_range, arr_azimuth
            )
            self.register_buffer("phys_h", torch.from_numpy(token_h_coords).to(dtype=torch.float32), persistent=False)
            self.register_buffer("phys_v", torch.from_numpy(token_v_coords).to(dtype=torch.float32), persistent=False)
            query_h_coords, query_w_coords = self._build_patch_query_coords(pixel_h_coords, pixel_w_coords)
            reference_points = self._build_reference_points(query_h_coords, query_w_coords)

            self.mask_token_h = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.mask_token_v = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            # ── [v8] Decoder-side axial pos/axis embeds ──
            # The decoder also runs self-attn in axial (1D) space, so it needs per-axis pos_embed and axis_embed.
            self.decoder_axis_embed = nn.Parameter(torch.zeros(1, 2, decoder_embed_dim))
            if self.pos_embed_mode == "ppe":
                self.decoder_geome_pe = PhysicalPositionalEncoding(decoder_embed_dim)
            else:
                self.register_buffer(
                    "decoder_pos_embed_h",
                    torch.zeros(1, self.num_h_tokens, decoder_embed_dim),
                )
                self.register_buffer(
                    "decoder_pos_embed_v",
                    torch.zeros(1, self.num_v_tokens, decoder_embed_dim),
                )

            self.axial_deform_fuse = AxialDeformCrossAttention(
                embed_dim=decoder_embed_dim,
                out_h=self.grid_size[0],
                out_w=self.grid_size[1],
                num_heads=deform_num_heads,
                num_points=deform_num_points,
                fuse="mean",
                align_corners=True,
                reference_points=reference_points,
            )
            self.fused_patch_proj = nn.Identity()
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size[0] * self.patch_size[1], bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.recon_loss_type = str(recon_loss_type).lower().strip()
        self.loss_weighting = str(loss_weighting).lower().strip()
        if self.recon_loss_type in ("weighted_mse", "weighted_l1"):
            self.loss_weighting = "patch_intensity"
            self.recon_loss_type = self.recon_loss_type.replace("weighted_", "", 1)
        if self.recon_loss_type not in ("mse", "l1"):
            raise ValueError(
                f"Unsupported recon_loss_type='{recon_loss_type}'. "
                "Use 'mse', 'l1', 'weighted_mse', or 'weighted_l1'."
            )
        if self.loss_weighting not in ("none", "patch_intensity"):
            raise ValueError(
                f"Unsupported loss_weighting='{loss_weighting}'. Use 'none' or 'patch_intensity'."
            )
        self.masking_mode = str(masking_mode).lower().strip()
        if self.masking_mode not in (
            "random", "weighted_masking", "weighted_block_masking", "random_block_masking",
        ):
            raise ValueError(
                "Unsupported masking_mode="
                f"'{masking_mode}'. Use 'random', 'weighted_masking', "
                "'weighted_block_masking', or 'random_block_masking'."
            )

        self.enable_dist_loss = bool(enable_dist_loss)
        self.lambda_multipath = float(lambda_multipath)
        self.lambda_sidelobs = float(lambda_sidelobs)
        self.dist_loss_axis_routing = str(dist_loss_axis_routing).lower().strip()
        if self.dist_loss_axis_routing not in ("physical", "swapped"):
            raise ValueError(
                "Unsupported dist_loss_axis_routing="
                f"'{dist_loss_axis_routing}'. Use 'physical' or 'swapped'."
            )
        self.weighted_mask_top_ratio = 0.30
        self.weighted_mask_hint_ratio = 0.05
        self.weighted_block_min_size = 5
        self.weighted_block_max_size = 8
        # 2D square-block masking for patch tokenization.  Only odd sizes are
        # used so the block spreads symmetrically in both directions from the
        # intensity-sampled centre patch.
        patch_sizes = [
            s
            for s in range(int(patch_block_min_size), int(patch_block_max_size) + 1)
            if s % 2 == 1
        ]
        if not patch_sizes:
            patch_sizes = [max(1, int(patch_block_min_size) | 1)]
        self.patch_block_sizes = tuple(patch_sizes)
        self.patch_block_max_steps = int(patch_block_max_steps)
        self._last_patch_block_hint = None
        if self.tokenization_mode == "axial":
            self._register_block_mask_banks()
        self.last_mask_axial = None
        self.last_mask_h = None
        self.last_mask_v = None
        self.last_mask_range = None
        self.last_mask_azimuth = None
        self.last_hint_range = None
        self.last_hint_azimuth = None
        self.last_pixel_mask_h = None
        self.last_pixel_mask_v = None
        self.last_pixel_mask_range = None
        self.last_pixel_mask_azimuth = None
        self.last_pixel_hint_range = None
        self.last_pixel_hint_azimuth = None
        self.last_pixel_mask = None
        self.last_patch_mask = None
        self.last_patch_hint = None
        self.last_fused_map = None
        self.last_profile = {}
        self.last_loss_breakdown = {}
        self.last_manifold_metrics = {}
        self.profile_runtime = False
        self._profile_timings = None

        valid_mask = torch.zeros(1, 1, self.decoder_img_size[0], self.decoder_img_size[1], dtype=torch.float32)
        valid_mask[:, :, : self.img_size[0], : self.img_size[1]] = 1.0
        self.register_buffer("decoder_valid_mask", valid_mask, persistent=False)
        self.register_buffer("patch_valid_weights", self.patchify_target(valid_mask), persistent=False)

        self.initialize_weights()

    @staticmethod
    def _ceil_to_multiple(value: int, divisor: int) -> int:
        return ((int(value) + int(divisor) - 1) // int(divisor)) * int(divisor)

    @staticmethod
    def _build_block_mask_bank(length: int, block_size: int) -> torch.Tensor:
        length = int(length)
        block_size = int(block_size)
        num_starts = max(0, length - block_size + 1)
        bank = torch.zeros((num_starts, length), dtype=torch.bool)
        if num_starts == 0:
            return bank
        starts = torch.arange(num_starts, dtype=torch.long).unsqueeze(1)
        offsets = torch.arange(block_size, dtype=torch.long).unsqueeze(0)
        bank.scatter_(1, starts + offsets, True)
        return bank

    def _register_block_mask_banks(self) -> None:
        for block_size in range(self.weighted_block_min_size, self.weighted_block_max_size + 1):
            self.register_buffer(
                f"block_kernel_{block_size}",
                torch.ones((1, 1, int(block_size)), dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                f"range_block_bank_{block_size}",
                self._build_block_mask_bank(self.num_range_tokens, block_size),
                persistent=False,
            )
            self.register_buffer(
                f"azimuth_block_bank_{block_size}",
                self._build_block_mask_bank(self.num_azimuth_tokens, block_size),
                persistent=False,
            )

    def _get_block_kernel(self, block_size: int) -> torch.Tensor:
        return getattr(self, f"block_kernel_{int(block_size)}")

    def _get_block_bank(self, axis_name: str, block_size: int) -> torch.Tensor:
        return getattr(self, f"{axis_name}_block_bank_{int(block_size)}")

    def set_runtime_profile(self, enabled: bool) -> None:
        self.profile_runtime = bool(enabled)
        if not self.profile_runtime:
            self._profile_timings = None

    def _profile_sync(self) -> None:
        if not self.profile_runtime:
            return
        try:
            device = next(self.parameters()).device
        except StopIteration:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    def _profile_start(self):
        if not self.profile_runtime:
            return None
        self._profile_sync()
        return time.perf_counter()

    def _profile_stop(self, key: str, start) -> None:
        if start is None:
            return
        self._profile_sync()
        if self._profile_timings is None:
            self._profile_timings = {}
        self._profile_timings[key] = self._profile_timings.get(key, 0.0) + (time.perf_counter() - start)

    @staticmethod
    def _resize_physical_coords(coords: Optional[np.ndarray], target_len: int) -> np.ndarray:
        if coords is None:
            if target_len <= 1:
                return np.zeros((target_len,), dtype=np.float32)
            return np.linspace(-1.0, 1.0, target_len, dtype=np.float32)
        coords_arr = np.asarray(coords, dtype=np.float32).reshape(-1)
        if coords_arr.size == 0:
            raise ValueError("physical coordinates are empty")
        if coords_arr.size == target_len:
            return coords_arr
        if coords_arr.size == 1:
            return np.full((target_len,), float(coords_arr[0]), dtype=np.float32)
        src = np.linspace(0.0, 1.0, coords_arr.size, dtype=np.float32)
        dst = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
        return np.interp(dst, src, coords_arr).astype(np.float32)

    def _build_coordinate_sets(
        self,
        arr_range: Optional[np.ndarray],
        arr_azimuth: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.pretrain_axis == "azimuth_range":
            pixel_h_source = arr_azimuth
            pixel_w_source = arr_range
        else:
            pixel_h_source = arr_range
            pixel_w_source = arr_azimuth
        pixel_h = self._resize_physical_coords(pixel_h_source, self.img_size[0])
        pixel_w = self._resize_physical_coords(pixel_w_source, self.img_size[1])
        token_h = self._resize_physical_coords(pixel_h_source, self.num_h_tokens)
        token_v = self._resize_physical_coords(pixel_w_source, self.num_v_tokens)
        return pixel_h, pixel_w, token_h, token_v

    @staticmethod
    def _normalize_coords(coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32).reshape(-1)
        lo = float(coords.min())
        hi = float(coords.max())
        if hi <= lo:
            return np.zeros_like(coords)
        return ((coords - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)

    def _build_reference_points(self, pixel_h: np.ndarray, pixel_w: np.ndarray) -> torch.Tensor:
        h_norm = self._normalize_coords(pixel_h)
        w_norm = self._normalize_coords(pixel_w)
        try:
            yy, xx = np.meshgrid(h_norm, w_norm, indexing="ij")
        except TypeError:
            yy, xx = np.meshgrid(h_norm, w_norm)
        grid = np.stack([xx, yy], axis=-1).astype(np.float32)
        return torch.from_numpy(grid).unsqueeze(0)

    @staticmethod
    def _pad_physical_coords(coords: np.ndarray, target_len: int) -> np.ndarray:
        coords_arr = np.asarray(coords, dtype=np.float32).reshape(-1)
        if coords_arr.size == target_len:
            return coords_arr
        if coords_arr.size > target_len:
            return coords_arr[:target_len]
        if coords_arr.size == 0:
            return np.zeros((target_len,), dtype=np.float32)
        pad_width = target_len - coords_arr.size
        return np.pad(coords_arr, (0, pad_width), mode="edge").astype(np.float32)

    def _build_patch_query_coords(self, pixel_h: np.ndarray, pixel_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        padded_h = self._pad_physical_coords(pixel_h, self.decoder_img_size[0])
        padded_w = self._pad_physical_coords(pixel_w, self.decoder_img_size[1])
        query_h = padded_h.reshape(self.grid_size[0], self.patch_size[0]).mean(axis=1).astype(np.float32)
        query_w = padded_w.reshape(self.grid_size[1], self.patch_size[1]).mean(axis=1).astype(np.float32)
        return query_h, query_w

    def initialize_weights(self) -> None:
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.tokenization_mode == "patch":
            encoder_pos_embed = get_2d_sincos_pos_embed(
                self.encoder_pos_embed.shape[-1], self.patch_embed.grid_size
            )
            self.encoder_pos_embed.data.copy_(torch.from_numpy(encoder_pos_embed).float().unsqueeze(0))
            nn.init.normal_(self.mask_token, std=0.02)
        else:
            nn.init.normal_(self.mask_token_h, std=0.02)
            nn.init.normal_(self.mask_token_v, std=0.02)
            nn.init.normal_(self.axis_embed, std=0.02)
            nn.init.normal_(self.decoder_axis_embed, std=0.02)
            # PE mode: initialize fixed sincos embeddings.
            if self.pos_embed_mode == "pe":
                pe_h = get_1d_sincos_pos_embed_from_grid(
                    self.embed_dim, np.arange(self.num_h_tokens, dtype=np.float32)
                )
                pe_v = get_1d_sincos_pos_embed_from_grid(
                    self.embed_dim, np.arange(self.num_v_tokens, dtype=np.float32)
                )
                self.encoder_pos_embed_h.copy_(torch.from_numpy(pe_h).float().unsqueeze(0))
                self.encoder_pos_embed_v.copy_(torch.from_numpy(pe_v).float().unsqueeze(0))

                dec_dim = self.decoder_pos_embed_h.shape[-1]
                dec_pe_h = get_1d_sincos_pos_embed_from_grid(
                    dec_dim, np.arange(self.num_h_tokens, dtype=np.float32)
                )
                dec_pe_v = get_1d_sincos_pos_embed_from_grid(
                    dec_dim, np.arange(self.num_v_tokens, dtype=np.float32)
                )
                self.decoder_pos_embed_h.copy_(torch.from_numpy(dec_pe_h).float().unsqueeze(0))
                self.decoder_pos_embed_v.copy_(torch.from_numpy(dec_pe_v).float().unsqueeze(0))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _pad_nchw(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = self.decoder_img_size[0] - x.shape[-2]
        pad_w = self.decoder_img_size[1] - x.shape[-1]
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"Expected spatial size <= {self.decoder_img_size}, got {(x.shape[-2], x.shape[-1])}"
            )
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h))

    def patchify_target(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs = self._pad_nchw(imgs)
        p_h, p_w = self.patch_size
        g_h, g_w = self.patch_embed.grid_size
        assert imgs.shape[1] == 1
        x = imgs.reshape(shape=(imgs.shape[0], 1, g_h, p_h, g_w, p_w))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], g_h * g_w, p_h * p_w))
        return x

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs = self._pad_nchw(imgs)
        p_h, p_w = self.patch_size
        g_h, g_w = self.patch_embed.grid_size
        assert imgs.shape[1] == self.in_chans
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, g_h, p_h, g_w, p_w))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], g_h * g_w, p_h * p_w * self.in_chans))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p_h, p_w = self.patch_size
        g_h, g_w = self.patch_embed.grid_size
        assert g_h * g_w == x.shape[1]
        assert x.shape[2] == p_h * p_w
        x = x.reshape(shape=(x.shape[0], g_h, g_w, p_h, p_w, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 1, g_h * p_h, g_w * p_w))
        return imgs[:, :, : self.img_size[0], : self.img_size[1]]

    def _stripe_tokenize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected x to be 4D (B,C,H,W), got {tuple(x.shape)}")
        bsz, ch, h, w = x.shape
        if ch != self.stem_out_chans:
            raise ValueError(f"Expected C={self.stem_out_chans}, got C={ch}")
        if (h, w) != self.img_size:
            raise ValueError(f"Expected input size {self.img_size}, got {(h, w)}")

        patch_h, patch_w = self.stripe_patch_width
        stride_h, stride_w = self.stripe_patch_stride
        x_h = F.unfold(x, kernel_size=(patch_h, w), stride=(stride_h, 1))
        x_h = x_h.view(bsz, ch, patch_h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
        x_h = x_h.view(bsz, -1, ch * patch_h * w)

        x_v = F.unfold(x, kernel_size=(h, patch_w), stride=(1, stride_w))
        x_v = x_v.view(bsz, ch, h, patch_w, -1).permute(0, 4, 1, 3, 2).contiguous()
        x_v = x_v.view(bsz, -1, ch * patch_w * h)

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
        x: (B, in_chans, H, W) — input before conv_stem
        Returns: (h_tok, v_tok) with shape (B, num_h_tokens, embed_dim) and (B, num_v_tokens, embed_dim)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x to be 4D (B,C,H,W), got {tuple(x.shape)}")
        bsz, ch, h, w = x.shape
        if ch != self.in_chans:
            raise ValueError(f"Expected C={self.in_chans}, got C={ch}")
        if (h, w) != self.img_size:
            raise ValueError(f"Expected input size {self.img_size}, got {(h, w)}")

        patch_h, patch_w = self.stripe_patch_width
        stride_h, stride_w = self.stripe_patch_stride

        # ── H-stripe: (B, num_h, in_chans, patch_h, W) ──
        x_h = F.unfold(x, kernel_size=(patch_h, w), stride=(stride_h, 1))
        num_h = x_h.shape[-1]
        x_h = x_h.view(bsz, ch, patch_h, w, num_h).permute(0, 4, 1, 2, 3).contiguous()
        x_h = x_h.view(bsz * num_h, ch, patch_h, w)                    # (B*num_h, in_chans, patch_h, W)
        x_h = self.conv_stem(x_h)                                       # (B*num_h, stem_out_chans, patch_h, W)
        x_h = x_h.view(bsz, num_h, self.stem_out_chans, patch_h, w)
        x_h = x_h.view(bsz, num_h, self.stem_out_chans * patch_h * w)  # (B, num_h, stem_out_chans*patch_h*W)
        h_tok = self.stripe_embed_h(x_h)                                # (B, num_h_tokens, embed_dim)

        # ── V-stripe: (B, num_v, in_chans, H, patch_w) ──
        x_v = F.unfold(x, kernel_size=(h, patch_w), stride=(1, stride_w))
        num_v = x_v.shape[-1]
        x_v = x_v.view(bsz, ch, h, patch_w, num_v).permute(0, 4, 1, 2, 3).contiguous()
        x_v = x_v.view(bsz * num_v, ch, h, patch_w)                    # (B*num_v, in_chans, H, patch_w)
        x_v = self.conv_stem(x_v)                                       # (B*num_v, stem_out_chans, H, patch_w)
        x_v = x_v.view(bsz, num_v, self.stem_out_chans, h, patch_w)
        x_v = x_v.view(bsz, num_v, self.stem_out_chans * h * patch_w)  # (B, num_v, stem_out_chans*H*patch_w)
        v_tok = self.stripe_embed_v(x_v)                                # (B, num_v_tokens, embed_dim)

        expected_h, expected_w = self.num_stripe_tokens_hw
        if h_tok.shape[1] != expected_h or v_tok.shape[1] != expected_w:
            raise ValueError(
                f"Token count mismatch: expected ({expected_h}, {expected_w}), "
                f"got ({h_tok.shape[1]}, {v_tok.shape[1]})"
            )
        return h_tok, v_tok

    def _patch_tokenize_per_token(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-patch conv-stem tokenization for patch mode.  Mirrors
        ``_stripe_tokenize_per_token``: each non-overlapping patch is
        unfolded, conv_stem is applied to each patch in isolation
        (zero-padded across the patch boundary), then patch_proj projects
        to embed_dim.  No information crosses patch boundaries, so the MAE
        encoder cannot satisfy the reconstruction loss by trivially copying
        a leaked neighbour.

        x: (B, in_chans, H, W) — input AFTER input_norm only (no global stem).
        Returns: (B, num_patches, embed_dim).
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x to be 4D (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_chans:
            raise ValueError(f"Expected in_chans={self.in_chans}, got C={x.shape[1]}")

        x = self._pad_nchw(x)                                          # (B, C, G_h*p_h, G_w*p_w)
        bsz, ch, _, _ = x.shape
        p_h, p_w = self.patch_size

        patches = F.unfold(x, kernel_size=(p_h, p_w), stride=(p_h, p_w))   # (B, C*p_h*p_w, N)
        num_patches = patches.shape[-1]
        if num_patches != self.num_patches:
            raise RuntimeError(
                f"Patch count mismatch: got {num_patches}, expected {self.num_patches}"
            )
        patches = patches.view(bsz, ch, p_h, p_w, num_patches)
        patches = patches.permute(0, 4, 1, 2, 3).contiguous()
        patches = patches.view(bsz * num_patches, ch, p_h, p_w)             # (B*N, C, p_h, p_w)

        patches = self.conv_stem(patches)                                   # (B*N, stem_out, p_h, p_w)
        tokens = self.patch_proj(patches)                                   # (B*N, embed, 1, 1)
        tokens = tokens.view(bsz, num_patches, -1)                          # (B, N, embed_dim)
        return tokens

    def _pixel_mask_to_patch_mask(self, pixel_mask: torch.Tensor) -> torch.Tensor:
        pixel_mask = self._pad_nchw(pixel_mask)
        valid = self.decoder_valid_mask.expand(pixel_mask.shape[0], -1, -1, -1)
        patch_sum = F.avg_pool2d(pixel_mask, kernel_size=self.patch_size, stride=self.patch_size)
        valid_sum = F.avg_pool2d(valid, kernel_size=self.patch_size, stride=self.patch_size)
        patch_mask = patch_sum / valid_sum.clamp_min(1.0e-6)
        return patch_mask.flatten(1).clamp_(0.0, 1.0)

    def _axial_mask_to_patch_mask(
        self,
        mask_axial: torch.Tensor,
        precomputed: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if precomputed is None:
            mask_h_px, mask_v_px, _ = self._axial_mask_to_pixel_masks(mask_axial)
        else:
            mask_h_px, mask_v_px, _ = precomputed
        pixel_mask = 0.5 * (mask_h_px + mask_v_px)
        return self._pixel_mask_to_patch_mask(pixel_mask)

    def _axial_mask_to_pixel_masks(self, mask_axial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, total_len = mask_axial.shape
        if total_len != self.num_axial_tokens:
            raise ValueError(
                f"mask_axial length mismatch: got {total_len}, expected {self.num_axial_tokens}"
            )

        mask_h = mask_axial[:, : self.num_h_tokens]
        mask_v = mask_axial[:, self.num_h_tokens :]

        in_h, in_w = self.img_size
        patch_h, patch_w = self.stripe_patch_width
        stride_h, stride_w = self.stripe_patch_stride

        weight_h = torch.ones((1, 1, patch_h), dtype=mask_axial.dtype, device=mask_axial.device)
        weight_v = torch.ones((1, 1, patch_w), dtype=mask_axial.dtype, device=mask_axial.device)

        h_sum = F.conv_transpose1d(mask_h.unsqueeze(1), weight_h, stride=stride_h)
        h_cnt = F.conv_transpose1d(torch.ones_like(mask_h).unsqueeze(1), weight_h, stride=stride_h)
        v_sum = F.conv_transpose1d(mask_v.unsqueeze(1), weight_v, stride=stride_w)
        v_cnt = F.conv_transpose1d(torch.ones_like(mask_v).unsqueeze(1), weight_v, stride=stride_w)

        def crop_or_pad_1d(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
            current_len = tensor.shape[-1]
            if current_len < target_len:
                return F.pad(tensor, (0, target_len - current_len))
            return tensor[:, :, :target_len]

        h_sum = crop_or_pad_1d(h_sum, in_h)
        h_cnt = crop_or_pad_1d(h_cnt, in_h)
        v_sum = crop_or_pad_1d(v_sum, in_w)
        v_cnt = crop_or_pad_1d(v_cnt, in_w)

        h_sum_2d = h_sum.unsqueeze(-1).expand(bsz, 1, in_h, in_w)
        h_cnt_2d = h_cnt.unsqueeze(-1).expand(bsz, 1, in_h, in_w)
        v_sum_2d = v_sum.unsqueeze(-2).expand(bsz, 1, in_h, in_w)
        v_cnt_2d = v_cnt.unsqueeze(-2).expand(bsz, 1, in_h, in_w)

        mask_h_px = h_sum_2d / h_cnt_2d.clamp_min(1.0e-6)
        mask_v_px = v_sum_2d / v_cnt_2d.clamp_min(1.0e-6)
        # Match the reference axial MAE: any masked axial token should contribute.
        mask_px = 0.5 * (mask_h_px + mask_v_px)
        return mask_h_px, mask_v_px, mask_px

    def _compute_axial_mask_scores(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected masking source to be 4D (B,C,H,W), got {tuple(x.shape)}")
        if tuple(x.shape[-2:]) != tuple(self.img_size):
            raise ValueError(f"Expected masking source size {self.img_size}, got {tuple(x.shape[-2:])}")

        intensity = x.mean(dim=1, keepdim=True)
        in_h, in_w = self.img_size
        patch_h, patch_w = self.stripe_patch_width
        stride_h, stride_w = self.stripe_patch_stride

        h_scores = F.unfold(intensity, kernel_size=(patch_h, in_w), stride=(stride_h, 1)).mean(dim=1)
        v_scores = F.unfold(intensity, kernel_size=(in_h, patch_w), stride=(1, stride_w)).mean(dim=1)
        return torch.cat([h_scores, v_scores], dim=1)

    def _axial_to_range_azimuth(
        self,
        h_values: torch.Tensor,
        v_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pretrain_axis == "azimuth_range":
            return v_values, h_values
        return h_values, v_values

    def _range_azimuth_to_axial(
        self,
        range_values: torch.Tensor,
        azimuth_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pretrain_axis == "azimuth_range":
            return azimuth_values, range_values
        return range_values, azimuth_values

    def _compute_range_azimuth_mask_scores(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        axial_scores = self._compute_axial_mask_scores(x)
        h_scores = axial_scores[:, : self.num_h_tokens]
        v_scores = axial_scores[:, self.num_h_tokens :]
        range_scores, azimuth_scores = self._axial_to_range_azimuth(h_scores, v_scores)
        return range_scores, azimuth_scores

    def _sample_block_starts(
        self,
        token_scores: torch.Tensor,
        current_mask: torch.Tensor,
        protected_hint: torch.Tensor,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_scores.shape != current_mask.shape or token_scores.shape != protected_hint.shape:
            raise ValueError(
                "token_scores, current_mask, and protected_hint must match, got "
                f"{tuple(token_scores.shape)}, {tuple(current_mask.shape)}, {tuple(protected_hint.shape)}"
            )

        block_size = int(block_size)
        num_starts = token_scores.shape[1] - block_size + 1
        starts = torch.zeros(token_scores.shape[0], device=token_scores.device, dtype=torch.long)
        valid_rows = torch.zeros(token_scores.shape[0], device=token_scores.device, dtype=torch.bool)
        if num_starts <= 0:
            return starts, valid_rows

        kernel = self._get_block_kernel(block_size).to(device=token_scores.device)
        available = ((~current_mask) & (~protected_hint)).to(dtype=torch.float32).unsqueeze(1)
        valid_starts = F.conv1d(available, kernel, stride=1).squeeze(1) > 0

        start_weights = token_scores[:, :num_starts].detach().clone().to(dtype=torch.float32).clamp_min_(0.0)
        start_weights.masked_fill_(~valid_starts, 0.0)
        fallback = valid_starts.to(dtype=start_weights.dtype)
        zero_rows = start_weights.sum(dim=1) <= 0
        if zero_rows.any():
            start_weights[zero_rows] = fallback[zero_rows]
        valid_rows = start_weights.sum(dim=1) > 0
        if valid_rows.any():
            starts[valid_rows] = torch.multinomial(start_weights[valid_rows], num_samples=1, replacement=False).squeeze(1)
        return starts, valid_rows

    @staticmethod
    def _sample_token_indices(
        weights_1d: torch.Tensor,
        candidate_mask: torch.Tensor,
        num_samples: int,
        prefer_low_scores: bool = False,
    ) -> torch.Tensor:
        if num_samples <= 0:
            return torch.empty((0,), dtype=torch.long, device=weights_1d.device)
        available = int(candidate_mask.sum().item())
        if available < num_samples:
            raise ValueError(f"Requested {num_samples} samples but only {available} candidates are available.")

        sample_weights = weights_1d.detach().clone().to(dtype=torch.float32).clamp_min_(0.0)
        if prefer_low_scores and available > 0:
            score_cap = sample_weights[candidate_mask].amax()
            sample_weights = score_cap - sample_weights + 1.0e-6
        sample_weights.masked_fill_(~candidate_mask, 0.0)
        if sample_weights.sum() <= 0:
            sample_weights = candidate_mask.to(dtype=torch.float32)
        return torch.multinomial(sample_weights, num_samples=num_samples, replacement=False)

    def _align_axis_mask_budget(
        self,
        axis_mask: torch.Tensor,
        axis_scores: torch.Tensor,
        target_mask_count: int,
        protected_hint: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current_mask_count = int(axis_mask.sum().item())
        if current_mask_count == target_mask_count:
            return axis_mask

        protected_hint = torch.zeros_like(axis_mask) if protected_hint is None else protected_hint.to(dtype=torch.bool)
        axis_mask = axis_mask.clone()
        axis_scores = axis_scores.detach().clone()

        if current_mask_count < target_mask_count:
            extra = target_mask_count - current_mask_count
            add_candidates = (~axis_mask) & (~protected_hint)
            add_count = min(extra, int(add_candidates.sum().item()))
            if add_count > 0:
                add_ids = self._sample_token_indices(axis_scores, add_candidates, add_count)
                axis_mask[add_ids] = True
        else:
            excess = current_mask_count - target_mask_count
            drop_candidates = axis_mask
            drop_count = min(excess, int(drop_candidates.sum().item()))
            if drop_count > 0:
                drop_ids = self._sample_token_indices(axis_scores, drop_candidates, drop_count, prefer_low_scores=True)
                axis_mask[drop_ids] = False

        axis_mask[protected_hint] = False
        return axis_mask

    def _align_total_mask_budget(
        self,
        mask_r: torch.Tensor,
        mask_a: torch.Tensor,
        score_r: torch.Tensor,
        score_a: torch.Tensor,
        target_mask_count: int,
        protected_hint_r: Optional[torch.Tensor] = None,
        protected_hint_a: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        protected_hint_r = torch.zeros_like(mask_r) if protected_hint_r is None else protected_hint_r.to(dtype=torch.bool)
        protected_hint_a = torch.zeros_like(mask_a) if protected_hint_a is None else protected_hint_a.to(dtype=torch.bool)

        num_range = mask_r.numel()
        flat_mask = torch.cat([mask_r.clone(), mask_a.clone()], dim=0)
        flat_scores = torch.cat([score_r.detach().clone(), score_a.detach().clone()], dim=0)
        flat_protected = torch.cat([protected_hint_r, protected_hint_a], dim=0)
        flat_mask[flat_protected] = False

        current_mask_count = int(flat_mask.sum().item())
        if current_mask_count < target_mask_count:
            extra = target_mask_count - current_mask_count
            add_candidates = (~flat_mask) & (~flat_protected)
            available = int(add_candidates.sum().item())
            if available < extra:
                raise ValueError(
                    "weighted_block_masking cannot satisfy the total mask budget while preserving hints: "
                    f"target={target_mask_count}, current={current_mask_count}, available_add={available}, "
                    f"protected={int(flat_protected.sum().item())}"
                )
            if extra > 0:
                add_ids = self._sample_token_indices(flat_scores, add_candidates, extra)
                flat_mask[add_ids] = True
        elif current_mask_count > target_mask_count:
            excess = current_mask_count - target_mask_count
            drop_candidates = flat_mask
            available = int(drop_candidates.sum().item())
            if available < excess:
                raise ValueError(
                    "weighted_block_masking produced fewer removable masked tokens than required: "
                    f"target={target_mask_count}, current={current_mask_count}, removable={available}"
                )
            if excess > 0:
                drop_ids = self._sample_token_indices(flat_scores, drop_candidates, excess, prefer_low_scores=True)
                flat_mask[drop_ids] = False

        flat_mask[flat_protected] = False
        return flat_mask[:num_range], flat_mask[num_range:]

    # ── Batched budget alignment (vectorized, no per-sample Python loops) ──

    @staticmethod
    def _gumbel_topk_sample(
        weights: torch.Tensor,
        candidates: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched weighted sampling of *variable* counts per row via the
        Gumbel-top-k trick.  Returns a bool mask of selected positions.

        Args:
            weights:    (n, L) non-negative sampling weights
            candidates: (n, L) bool — eligible positions
            counts:     (n,)   int  — how many to select per row (may differ)

        Returns:
            selected:   (n, L) bool mask with exactly counts[i] True per row i
                        (clamped to available candidates)
        """
        max_k = int(counts.max().item())
        if max_k <= 0:
            return torch.zeros_like(candidates)

        safe_w = weights.clone()
        safe_w[~candidates] = 0.0
        # Rows with zero total weight → uniform over candidates
        zero_rows = safe_w.sum(dim=1) <= 0
        safe_w[zero_rows] = candidates[zero_rows].float()

        # Gumbel-max trick: argmax of (log w + Gumbel) ≡ categorical sample
        # Taking top-k of these gives k samples without replacement.
        gumbel = -torch.log(-torch.log(torch.rand_like(safe_w).clamp_min_(1e-20)).clamp_min_(1e-20))
        perturbed = torch.log(safe_w.clamp_min_(1e-20)) + gumbel
        perturbed[~candidates] = -float("inf")

        _, topk_idx = perturbed.topk(min(max_k, perturbed.shape[1]), dim=1)  # (n, max_k)
        step = torch.arange(topk_idx.shape[1], device=counts.device).unsqueeze(0)
        use = step < counts.unsqueeze(1)  # (n, max_k)

        selected = torch.zeros_like(candidates, dtype=torch.long)
        safe_idx = topk_idx.clone()
        safe_idx[~use] = 0  # dummy index for unused slots
        selected.scatter_add_(1, safe_idx, use.long())
        return selected > 0

    def _align_axis_mask_budget_batched(
        self,
        axis_mask: torch.Tensor,
        axis_scores: torch.Tensor,
        target_mask_count: int,
        protected_hint: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Vectorized budget alignment for one axis across the full batch."""
        if protected_hint is None:
            protected_hint = torch.zeros_like(axis_mask)
        protected_hint = protected_hint.bool()
        axis_mask = axis_mask.clone()
        axis_mask[protected_hint] = False

        current = axis_mask.sum(dim=1)  # (n,)

        # Under-budget: add tokens
        deficit = (target_mask_count - current).clamp_min_(0).to(dtype=torch.long)
        if (deficit > 0).any():
            add_cand = (~axis_mask) & (~protected_hint)
            add_w = axis_scores.detach().float().clamp_min_(0.0)
            axis_mask |= self._gumbel_topk_sample(add_w, add_cand, deficit)

        # Over-budget: drop tokens (prefer low-score)
        current = axis_mask.sum(dim=1)
        excess = (current - target_mask_count).clamp_min_(0).to(dtype=torch.long)
        if (excess > 0).any():
            drop_cand = axis_mask.clone()
            drop_w = axis_scores.detach().float().clamp_min_(0.0)
            cap = drop_w.amax(dim=1, keepdim=True)
            drop_w = cap - drop_w + 1e-6  # invert: prefer low-score
            axis_mask &= ~self._gumbel_topk_sample(drop_w, drop_cand, excess)

        axis_mask[protected_hint] = False
        return axis_mask

    def _align_total_mask_budget_batched(
        self,
        mask_r: torch.Tensor,
        mask_a: torch.Tensor,
        score_r: torch.Tensor,
        score_a: torch.Tensor,
        target_mask_count: int,
        protected_hint_r: Optional[torch.Tensor] = None,
        protected_hint_a: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized total budget alignment across range + azimuth."""
        R = mask_r.shape[1]
        hint_r = torch.zeros_like(mask_r) if protected_hint_r is None else protected_hint_r.bool()
        hint_a = torch.zeros_like(mask_a) if protected_hint_a is None else protected_hint_a.bool()

        flat_mask = torch.cat([mask_r.clone(), mask_a.clone()], dim=1)
        flat_scores = torch.cat([score_r.detach(), score_a.detach()], dim=1)
        flat_protected = torch.cat([hint_r, hint_a], dim=1)
        flat_mask[flat_protected] = False

        current = flat_mask.sum(dim=1)

        # Add
        deficit = (target_mask_count - current).clamp_min_(0).to(dtype=torch.long)
        if (deficit > 0).any():
            add_cand = (~flat_mask) & (~flat_protected)
            flat_mask |= self._gumbel_topk_sample(flat_scores.float().clamp_min_(0.0), add_cand, deficit)

        # Drop (prefer low-score)
        current = flat_mask.sum(dim=1)
        excess = (current - target_mask_count).clamp_min_(0).to(dtype=torch.long)
        if (excess > 0).any():
            drop_cand = flat_mask.clone()
            drop_w = flat_scores.float().clamp_min_(0.0)
            cap = drop_w.amax(dim=1, keepdim=True)
            drop_w = cap - drop_w + 1e-6
            flat_mask &= ~self._gumbel_topk_sample(drop_w, drop_cand, excess)

        flat_mask[flat_protected] = False
        return flat_mask[:, :R], flat_mask[:, R:]

    def _apply_boolean_mask(
        self,
        x: torch.Tensor,
        mask_bool: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, length, dim = x.shape
        if mask_bool.shape != (n, length):
            raise ValueError(f"mask_bool must be shaped {(n, length)}, got {tuple(mask_bool.shape)}")

        mask_bool = mask_bool.to(device=x.device, dtype=torch.bool)
        masked_per_sample = mask_bool.sum(dim=1)
        if masked_per_sample.numel() > 0 and not torch.equal(masked_per_sample, masked_per_sample[:1].expand_as(masked_per_sample)):
            unique_counts = torch.unique(masked_per_sample).tolist()
            raise ValueError(
                "All samples must share the same masked-token count for encoder gathering. "
                f"Observed counts={unique_counts}"
            )

        len_mask = int(masked_per_sample[0].item()) if masked_per_sample.numel() > 0 else 0
        len_keep = length - len_mask
        keep_noise = torch.rand(n, length, device=x.device)
        mask_noise = torch.rand(n, length, device=x.device) + 1.0
        ids_shuffle = torch.argsort(torch.where(mask_bool, mask_noise, keep_noise), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))
        return x_masked, mask_bool.to(dtype=x.dtype), ids_restore

    def weighted_block_masking(
        self,
        mask_source: torch.Tensor,
        mask_ratio: float,
        weighted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized weighted block masking.

        Replaces the original O(total_len * 4) Python loop with K=20 fixed
        steps, each fully batched over all n samples.  GPU↔CPU syncs are
        eliminated inside the block-placement phase; the budget-alignment
        phase uses at most 6 syncs total (via _gumbel_topk_sample).
        """
        n = mask_source.shape[0]
        device = mask_source.device
        R = self.num_range_tokens
        A = self.num_azimuth_tokens
        mask_r = torch.zeros((n, R), device=device, dtype=torch.bool)
        mask_a = torch.zeros((n, A), device=device, dtype=torch.bool)
        hint_r = torch.zeros_like(mask_r)
        hint_a = torch.zeros_like(mask_a)
        self._last_block_hint_range = None
        self._last_block_hint_azimuth = None

        # Apply mask_ratio independently along each axis.
        target_mask_r = int(round(R * mask_ratio))
        target_mask_a = int(round(A * mask_ratio))
        target_mask_r = max(0, min(target_mask_r, R))
        target_mask_a = max(0, min(target_mask_a, A))

        if target_mask_r <= 0 and target_mask_a <= 0:
            self._last_block_hint_range = hint_r.detach()
            self._last_block_hint_azimuth = hint_a.detach()
            return mask_r, mask_a

        if weighted:
            score_r, score_a = self._compute_range_azimuth_mask_scores(mask_source)
            score_r = (score_r - score_r.amin(dim=1, keepdim=True)).clamp_min_(0.0) + 1.0e-6
            score_a = (score_a - score_a.amin(dim=1, keepdim=True)).clamp_min_(0.0) + 1.0e-6
        else:
            # Uniform: block placement stays, intensity bias goes away.
            score_r = torch.ones((n, R), device=device, dtype=torch.float32)
            score_a = torch.ones((n, A), device=device, dtype=torch.float32)

        # ── Phase 1: Vectorized block placement (K fixed steps, 0 CPU syncs) ──
        # Each step places one block per axis for all active samples.
        # Block size is uniform per step (randomly sampled) to avoid an inner
        # loop over block sizes, reducing kernel launches from K*4 to K.
        K = 20  # generous upper bound: ceil(max_target / min_block) ≈ 65/5 = 13
        blocked = torch.zeros(n, device=device, dtype=torch.bool)

        step_block_sizes = torch.randint(
            low=self.weighted_block_min_size,
            high=self.weighted_block_max_size + 1,
            size=(K,),
            device=device,
        )

        for k in range(K):
            bs = int(step_block_sizes[k].item())
            kernel = self._get_block_kernel(bs).to(device=device)
            num_starts_r = R - bs + 1
            num_starts_a = A - bs + 1
            if num_starts_r <= 0 or num_starts_a <= 0:
                continue

            active = (~blocked) & (mask_r.sum(1) < target_mask_r) & (mask_a.sum(1) < target_mask_a)

            # Range: valid starts & weighted sampling (full batch, no branching)
            avail_r = ((~mask_r) & (~hint_r)).float().unsqueeze(1)
            vs_r = F.conv1d(avail_r, kernel).squeeze(1) > 0  # (n, num_starts_r)
            sw_r = score_r[:, :num_starts_r].clone().clamp_min_(0.0)
            sw_r.masked_fill_(~vs_r, 0.0)
            sw_r[~active] = 0.0
            fb_r = active & (sw_r.sum(1) <= 0)
            sw_r[fb_r] = vs_r[fb_r].float()
            valid_r = active & (sw_r.sum(1) > 0)
            sw_r[~valid_r] = 1.0  # dummy for multinomial safety
            starts_r = torch.multinomial(sw_r, 1).squeeze(1)

            # Azimuth: same
            avail_a = ((~mask_a) & (~hint_a)).float().unsqueeze(1)
            vs_a = F.conv1d(avail_a, kernel).squeeze(1) > 0
            sw_a = score_a[:, :num_starts_a].clone().clamp_min_(0.0)
            sw_a.masked_fill_(~vs_a, 0.0)
            sw_a[~active] = 0.0
            fb_a = active & (sw_a.sum(1) <= 0)
            sw_a[fb_a] = vs_a[fb_a].float()
            valid_a = active & (sw_a.sum(1) > 0)
            sw_a[~valid_a] = 1.0
            starts_a = torch.multinomial(sw_a, 1).squeeze(1)

            can_continue = valid_r & valid_a
            blocked |= active & ~can_continue

            # Fetch block patterns from pre-built banks
            range_blocks = self._get_block_bank("range", bs)[starts_r.clamp(max=num_starts_r - 1)]
            azimuth_blocks = self._get_block_bank("azimuth", bs)[starts_a.clamp(max=num_starts_a - 1)]

            new_r = range_blocks & (~mask_r) & (~hint_r)
            new_a = azimuth_blocks & (~mask_a) & (~hint_a)
            worthwhile = can_continue & ((new_r.sum(1) + new_a.sum(1)) > 1)

            updated_r = (mask_r | range_blocks) & (~hint_r)
            updated_a = (mask_a | azimuth_blocks) & (~hint_a)

            # Hint selection — one visible token per block placement
            has_new_r = new_r.any(dim=1)
            has_new_a = new_a.any(dim=1)
            prefer_range = torch.rand(n, device=device) < 0.5
            choose_range = worthwhile & has_new_r & (prefer_range | ~has_new_a)
            choose_azimuth = worthwhile & has_new_a & (~prefer_range | ~has_new_r)

            # Sample hint indices (multinomial-safe: dummy weights for inactive rows)
            hw_r = new_r.float()
            hw_r[~choose_range] = 1.0
            hidx_r = torch.multinomial(hw_r, 1).squeeze(1)

            hw_a = new_a.float()
            hw_a[~choose_azimuth] = 1.0
            hidx_a = torch.multinomial(hw_a, 1).squeeze(1)

            # Apply hints via advanced indexing
            updated_r[choose_range, hidx_r[choose_range]] = False
            updated_a[choose_azimuth, hidx_a[choose_azimuth]] = False
            new_hint_r = hint_r.clone()
            new_hint_a = hint_a.clone()
            new_hint_r[choose_range, hidx_r[choose_range]] = True
            new_hint_a[choose_azimuth, hidx_a[choose_azimuth]] = True

            # Commit only for worthwhile samples (branchless via torch.where)
            w_r = worthwhile.unsqueeze(1).expand_as(mask_r)
            mask_r = torch.where(w_r, updated_r, mask_r)
            hint_r = torch.where(w_r, new_hint_r, hint_r)
            w_a = worthwhile.unsqueeze(1).expand_as(mask_a)
            mask_a = torch.where(w_a, updated_a, mask_a)
            hint_a = torch.where(w_a, new_hint_a, hint_a)

        # ── Phase 2: Batched budget alignment (Gumbel top-k, ≤ 6 CPU syncs) ──
        mask_r = self._align_axis_mask_budget_batched(
            mask_r, score_r, target_mask_r, protected_hint=hint_r,
        )
        mask_a = self._align_axis_mask_budget_batched(
            mask_a, score_a, target_mask_a, protected_hint=hint_a,
        )

        # ── Phase 3: Ensure each axis matches its target independently ──
        # Encoder gathering requires every sample in the batch to have the same number of masks.
        # Phase 2 almost always fills the budget, so Phase 3 is a safety net, but the previous
        # per-sample Python loop + .item() sync ran B*2 times per iteration.
        # Vectorize it so syncs happen at most once per axis.
        need_fix_r = mask_r.sum(dim=1) != target_mask_r
        if need_fix_r.any():
            k_r = min(target_mask_r, mask_r.shape[1])
            replacement_r = torch.zeros_like(mask_r)
            if k_r > 0:
                _, top_r = score_r.topk(k_r, dim=1)
                replacement_r.scatter_(1, top_r, True)
            mask_r = torch.where(need_fix_r[:, None], replacement_r, mask_r)

        need_fix_a = mask_a.sum(dim=1) != target_mask_a
        if need_fix_a.any():
            k_a = min(target_mask_a, mask_a.shape[1])
            replacement_a = torch.zeros_like(mask_a)
            if k_a > 0:
                _, top_a = score_a.topk(k_a, dim=1)
                replacement_a.scatter_(1, top_a, True)
            mask_a = torch.where(need_fix_a[:, None], replacement_a, mask_a)

        self._last_block_hint_range = hint_r.detach()
        self._last_block_hint_azimuth = hint_a.detach()
        return mask_r, mask_a

    def _compute_patch_mask_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Mean input intensity per patch — the patch-mode analogue of
        ``_compute_axial_mask_scores``.  Returns (B, G_h, G_w)."""
        if x.ndim != 4:
            raise ValueError(f"Expected masking source to be 4D (B,C,H,W), got {tuple(x.shape)}")
        if tuple(x.shape[-2:]) != tuple(self.img_size):
            raise ValueError(f"Expected masking source size {self.img_size}, got {tuple(x.shape[-2:])}")

        intensity = self._pad_nchw(x.mean(dim=1, keepdim=True))
        scores = F.avg_pool2d(intensity, kernel_size=self.patch_size, stride=self.patch_size)
        return scores.squeeze(1)

    def patch_block_masking(
        self,
        mask_source: torch.Tensor,
        mask_ratio: float,
        weighted: bool = True,
    ) -> torch.Tensor:
        """
        2D square-block masking for patch tokenization — the patch-mode
        counterpart of :meth:`weighted_block_masking`.

        Each step samples a centre patch, then masks the square block of
        ``b x b`` patches that spreads symmetrically in both directions from
        that centre (clamped at the grid border).  Exactly one randomly chosen
        patch inside the newly masked block is left visible as a hint and is
        protected from being re-masked later.

        ``weighted=True`` draws the centre from an intensity-weighted
        distribution over the still-available patches; ``weighted=False`` draws
        it uniformly.  Everything else -- block sizes, hint rule, budget -- is
        identical, so the pair isolates the intensity weighting from the block
        structure itself.

        Returns a (B, num_patches) bool mask with exactly
        ``round(num_patches * mask_ratio)`` masked patches per sample — the
        encoder gather requires a uniform count across the batch.
        """
        n = mask_source.shape[0]
        device = mask_source.device
        g_h, g_w = self.grid_size
        length = self.num_patches

        target_mask = int(round(length * float(mask_ratio)))
        target_mask = max(0, min(target_mask, length))

        mask = torch.zeros((n, length), device=device, dtype=torch.bool)
        hint = torch.zeros_like(mask)
        if target_mask <= 0:
            self._last_patch_block_hint = hint
            return mask

        if weighted:
            score = self._compute_patch_mask_scores(mask_source)
            score = score.reshape(n, length).to(dtype=torch.float32)
            score = (score - score.amin(dim=1, keepdim=True)).clamp_min_(0.0) + 1.0e-6
        else:
            # Uniform scores make every downstream use of `score` -- centre
            # sampling, budget alignment, and the top-k safety net -- unbiased,
            # so the only structure left in the mask is the blocks themselves.
            score = torch.ones((n, length), device=device, dtype=torch.float32)

        rows = torch.arange(g_h, device=device).unsqueeze(0)
        cols = torch.arange(g_w, device=device).unsqueeze(0)
        # Sampled on CPU: the block size is consumed by Python control flow, so
        # drawing it on-device would force a GPU->CPU sync every step.
        step_sizes = [
            self.patch_block_sizes[i]
            for i in torch.randint(0, len(self.patch_block_sizes), (self.patch_block_max_steps,)).tolist()
        ]

        blocked = torch.zeros(n, device=device, dtype=torch.bool)
        for b in step_sizes:
            half = b // 2
            span_h = min(b, g_h)
            span_w = min(b, g_w)
            active = (~blocked) & (mask.sum(dim=1) < target_mask)

            # Centre sampling: intensity-weighted over not-yet-masked, non-hint patches.
            avail = (~mask) & (~hint)
            weights = score * avail.to(score.dtype)
            weights[~active] = 0.0
            fallback = active & (weights.sum(dim=1) <= 0)
            weights[fallback] = avail[fallback].to(weights.dtype)
            valid = active & (weights.sum(dim=1) > 0)
            weights[~valid] = 1.0  # dummy row for multinomial safety
            centre = torch.multinomial(weights, 1).squeeze(1)
            blocked |= active & ~valid

            # Square block spreading both ways from the centre, clamped to the grid.
            r0 = (centre // g_w - half).clamp(0, max(g_h - span_h, 0))
            c0 = (centre % g_w - half).clamp(0, max(g_w - span_w, 0))
            row_in = (rows >= r0.unsqueeze(1)) & (rows < (r0 + span_h).unsqueeze(1))
            col_in = (cols >= c0.unsqueeze(1)) & (cols < (c0 + span_w).unsqueeze(1))
            block = (row_in.unsqueeze(2) & col_in.unsqueeze(1)).reshape(n, length)

            new = block & (~mask) & (~hint)
            worthwhile = valid & (new.sum(dim=1) > 1)
            updated = (mask | block) & (~hint)

            # Hint: exactly one visible patch per placed block.
            hint_w = new.to(torch.float32)
            hint_w[~worthwhile] = 1.0
            hint_idx = torch.multinomial(hint_w, 1).squeeze(1)
            updated[worthwhile, hint_idx[worthwhile]] = False
            new_hint = hint.clone()
            new_hint[worthwhile, hint_idx[worthwhile]] = True

            keep = worthwhile.unsqueeze(1)
            mask = torch.where(keep, updated, mask)
            hint = torch.where(keep, new_hint, hint)

        # Budget alignment (intensity-weighted Gumbel top-k, hints protected).
        mask = self._align_axis_mask_budget_batched(
            mask, score, target_mask, protected_hint=hint,
        )

        # Safety net: the encoder gather needs an identical count for every sample.
        need_fix = mask.sum(dim=1) != target_mask
        if need_fix.any():
            replacement = torch.zeros_like(mask)
            _, top_idx = score.topk(min(target_mask, length), dim=1)
            replacement.scatter_(1, top_idx, True)
            mask = torch.where(need_fix.unsqueeze(1), replacement, mask)

        self._last_patch_block_hint = hint.detach()
        return mask

    @staticmethod
    def _adjust_single_axis_mask(
        mask: torch.Tensor,
        score: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        """
        Adjust a binary mask for a single axis to have exactly target_count True values.

        Args:
            mask: (1, axis_len) boolean tensor
            score: (1, axis_len) scores for tokens on this axis
            target_count: desired number of True values

        Returns:
            adjusted mask with exactly target_count True values
        """
        # Reset mask and select target_count tokens with highest scores
        new_mask = torch.zeros_like(mask, dtype=torch.bool)
        target_count = min(target_count, mask.shape[1])
        if target_count > 0:
            _, top_indices = torch.topk(score.squeeze(0), target_count)
            new_mask[0, top_indices] = True

        return new_mask

    @staticmethod
    def random_masking(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, length, dim = x.shape
        len_keep = int(length * (1 - mask_ratio))
        noise = torch.rand(n, length, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        mask = torch.ones([n, length], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def weighted_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        token_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, length, dim = x.shape
        if token_scores.shape != (n, length):
            raise ValueError(
                f"token_scores must be shaped {(n, length)}, got {tuple(token_scores.shape)}"
            )

        len_keep = int(length * (1 - mask_ratio))
        len_mask = length - len_keep
        mask = torch.zeros((n, length), device=x.device, dtype=x.dtype)

        if len_mask > 0:
            top_count = max(1, int(np.ceil(length * self.weighted_mask_top_ratio)))
            protected_count = 0
            if len_keep > 0:
                protected_count = max(1, int(np.ceil(length * self.weighted_mask_hint_ratio)))
                protected_count = min(protected_count, top_count, len_keep)

            sample_weights = token_scores - token_scores.amin(dim=1, keepdim=True)
            sample_weights = sample_weights + 1.0e-6

            if protected_count > 0:
                top_ids = torch.topk(token_scores, k=top_count, dim=1, largest=True, sorted=True).indices
                protected_order = torch.argsort(torch.rand(n, top_count, device=x.device), dim=1)
                protected_pos = protected_order[:, :protected_count]
                protected_ids = torch.gather(top_ids, dim=1, index=protected_pos)
                protected_mask = torch.zeros((n, length), device=x.device, dtype=torch.bool)
                protected_mask.scatter_(1, protected_ids, True)
                sample_weights = sample_weights.masked_fill(protected_mask, 0.0)

            masked_ids = torch.multinomial(sample_weights, num_samples=len_mask, replacement=False)
            mask.scatter_(1, masked_ids, 1.0)

        keep_noise = torch.rand(n, length, device=x.device)
        mask_noise = torch.rand(n, length, device=x.device) + 1.0
        ids_shuffle = torch.argsort(torch.where(mask > 0, mask_noise, keep_noise), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))
        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        mask_source = x

        stage_start = self._profile_start()
        if self.input_norm is not None:
            x = self.input_norm(x)
        self._profile_stop("input_norm", stage_start)

        stage_start = self._profile_start()
        if self.conv_stem_mode == "global":
            x = self.conv_stem(x)
        self._profile_stop("conv_stem", stage_start)

        if self.tokenization_mode == "patch":
            # ── Standard 2D patch tokenization ──
            stage_start = self._profile_start()
            if self.conv_stem_mode == "per_token":
                # x is still raw input (only input_norm applied); per-patch
                # unfold + conv_stem prevents inter-patch information leakage.
                x = self._patch_tokenize_per_token(x)
            else:
                # global stem already applied above.
                x = self._pad_nchw(x)
                x = self.patch_proj(x)                     # (B, embed_dim, G_h, G_w)
                x = x.flatten(2).transpose(1, 2)           # (B, num_patches, embed_dim)
            x = x + self.encoder_pos_embed
            self._profile_stop("patch_tokenize", stage_start)

            stage_start = self._profile_start()
            self._last_patch_block_hint = None
            if self.masking_mode in ("weighted_block_masking", "random_block_masking"):
                mask_bool = self.patch_block_masking(
                    mask_source, mask_ratio,
                    weighted=self.masking_mode == "weighted_block_masking",
                )
                x, mask, ids_restore = self._apply_boolean_mask(x, mask_bool)
            elif self.masking_mode == "weighted_masking":
                token_scores = (
                    self._compute_patch_mask_scores(mask_source)
                    .reshape(x.shape[0], self.num_patches)
                    .to(device=x.device, dtype=x.dtype)
                )
                x, mask, ids_restore = self.weighted_masking(x, mask_ratio, token_scores)
            else:
                x, mask, ids_restore = self.random_masking(x, mask_ratio)
            self._profile_stop("token_masking", stage_start)
        else:
            # ── Axial stripe tokenization ──
            stage_start = self._profile_start()
            if self.conv_stem_mode == "per_token":
                # x is still raw input (input_norm only), apply conv_stem per stripe
                h_tok, v_tok = self._stripe_tokenize_per_token(x)
            else:
                # x already has conv_stem applied (global mode)
                h_tok, v_tok = self._stripe_tokenize(x)
            self._profile_stop("stripe_tokenize", stage_start)

            stage_start = self._profile_start()
            if self.pos_embed_mode == "ppe":
                pe_h = self.geome_pe(self.phys_h).to(device=h_tok.device, dtype=h_tok.dtype)
                pe_v = self.geome_pe(self.phys_v).to(device=v_tok.device, dtype=v_tok.dtype)
            else:  # "pe" mode
                pe_h = self.encoder_pos_embed_h.to(device=h_tok.device, dtype=h_tok.dtype)
                pe_v = self.encoder_pos_embed_v.to(device=v_tok.device, dtype=v_tok.dtype)
            axis_h = self.axis_embed[:, 0:1, :].to(device=h_tok.device, dtype=h_tok.dtype)
            axis_v = self.axis_embed[:, 1:2, :].to(device=v_tok.device, dtype=v_tok.dtype)
            h_tok = h_tok + pe_h + axis_h
            v_tok = v_tok + pe_v + axis_v
            self._profile_stop("encoder_pos_axis", stage_start)

            stage_start = self._profile_start()
            x = torch.cat([h_tok, v_tok], dim=1)
            self._last_block_hint_range = None
            self._last_block_hint_azimuth = None
            if self.masking_mode == "weighted_masking":
                token_scores = self._compute_axial_mask_scores(mask_source).to(device=x.device, dtype=x.dtype)
                x, mask, ids_restore = self.weighted_masking(x, mask_ratio, token_scores)
            elif self.masking_mode in ("weighted_block_masking", "random_block_masking"):
                mask_r, mask_a = self.weighted_block_masking(
                    mask_source, mask_ratio,
                    weighted=self.masking_mode == "weighted_block_masking",
                )
                mask_h, mask_v = self._range_azimuth_to_axial(mask_r, mask_a)
                mask_axial_bool = torch.cat([mask_h, mask_v], dim=1)
                x, mask, ids_restore = self._apply_boolean_mask(x, mask_axial_bool)
            else:
                x, mask, ids_restore = self.random_masking(x, mask_ratio)

            self._profile_stop("token_masking", stage_start)

        stage_start = self._profile_start()
        for block_idx, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)
        self._profile_stop("encoder_blocks", stage_start)
        return x, mask, ids_restore, {}

    def _fused_to_patch_tokens(self, fused_map: torch.Tensor) -> torch.Tensor:
        if fused_map.ndim != 4:
            raise ValueError(f"Expected fused_map to be (B,Gh,Gw,C), got {tuple(fused_map.shape)}")
        if tuple(fused_map.shape[1:3]) != tuple(self.grid_size):
            raise ValueError(
                f"Expected patch-grid fused map with spatial size {self.grid_size}, got {tuple(fused_map.shape[1:3])}"
            )
        x = fused_map.reshape(fused_map.shape[0], self.num_patches, fused_map.shape[-1]).contiguous()
        return self.fused_patch_proj(x)

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        stage_start = self._profile_start()
        x = self.decoder_embed(x)
        self._profile_stop("decoder_embed", stage_start)

        bsz, len_keep, dim = x.shape
        len_full = ids_restore.shape[1]

        if self.tokenization_mode == "patch":
            # ── Standard MAE decoder: unshuffle visible + mask tokens ──
            stage_start = self._profile_start()
            num_mask = len_full - len_keep
            mask_tokens = self.mask_token.repeat(bsz, num_mask, 1).to(dtype=x.dtype)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, dim))
            self._profile_stop("restore_mask_tokens", stage_start)

            stage_start = self._profile_start()
            x = x + self.decoder_pos_embed
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)
            x = self.decoder_pred(x)
            self._profile_stop("decoder_blocks", stage_start)
            return x
        else:
            # ── [v8] Axial decoder: fill mask tokens → axial pos/axis → decoder blocks
            #         → deform fuse (readout) → decoder_pred ──
            stage_start = self._profile_start()
            mask_h = self.mask_token_h.expand(bsz, self.num_h_tokens, dim)
            mask_v = self.mask_token_v.expand(bsz, self.num_v_tokens, dim)
            x_full = torch.cat([mask_h, mask_v], dim=1).clone().to(dtype=x.dtype)
            if x_full.shape[1] != len_full:
                raise ValueError(
                    f"Axial token length mismatch: base={x_full.shape[1]}, ids_restore={len_full}"
                )

            ids_shuffle = torch.argsort(ids_restore, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            x_full.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, dim), x)
            self._profile_stop("restore_mask_tokens", stage_start)

            # ── [v8] Add decoder-side axial pos_embed + axis_embed ──
            stage_start = self._profile_start()
            if self.pos_embed_mode == "ppe":
                dec_pe_h = self.decoder_geome_pe(self.phys_h).to(device=x_full.device, dtype=x_full.dtype)
                dec_pe_v = self.decoder_geome_pe(self.phys_v).to(device=x_full.device, dtype=x_full.dtype)
            else:
                dec_pe_h = self.decoder_pos_embed_h.to(device=x_full.device, dtype=x_full.dtype)
                dec_pe_v = self.decoder_pos_embed_v.to(device=x_full.device, dtype=x_full.dtype)
            dec_axis_h = self.decoder_axis_embed[:, 0:1, :].to(device=x_full.device, dtype=x_full.dtype)
            dec_axis_v = self.decoder_axis_embed[:, 1:2, :].to(device=x_full.device, dtype=x_full.dtype)
            x_full[:, : self.num_h_tokens] = x_full[:, : self.num_h_tokens] + dec_pe_h + dec_axis_h
            x_full[:, self.num_h_tokens :] = x_full[:, self.num_h_tokens :] + dec_pe_v + dec_axis_v
            self._profile_stop("decoder_pos_axis", stage_start)

            # ── [v8] Decoder self-attn in axial space ──
            stage_start = self._profile_start()
            for blk in self.decoder_blocks:
                x_full = blk(x_full)
            x_full = self.decoder_norm(x_full)
            self._profile_stop("decoder_blocks", stage_start)

            # ── [v8] Readout: axial tokens → 2D patch grid via deform cross-attn ──
            stage_start = self._profile_start()
            fused_map = self.axial_deform_fuse(
                x_full,
                num_h_tokens=self.num_h_tokens,
                num_v_tokens=self.num_v_tokens,
            )
            self.last_fused_map = fused_map.detach()
            self._profile_stop("axial_deform_fuse", stage_start)

            stage_start = self._profile_start()
            patch_tokens = self._fused_to_patch_tokens(fused_map)
            self._profile_stop("fused_to_patch_tokens", stage_start)

            x = self.decoder_pred(patch_tokens)
            return x

    def forward_loss(self, target_imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify_target(target_imgs)
        target_for_weight = target
        weights = self.patch_valid_weights.to(device=target.device, dtype=target.dtype).expand(target.shape[0], -1, -1)
        patch_denom = weights.sum(dim=-1).clamp_min(1.0e-6)

        if self.norm_pix_loss:
            denom = patch_denom.unsqueeze(-1)
            mean = (target * weights).sum(dim=-1, keepdim=True) / denom
            var = ((target - mean) ** 2 * weights).sum(dim=-1, keepdim=True) / denom
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        if self.recon_loss_type == "l1":
            loss = torch.abs(pred - target)
        else:
            loss = (pred - target) ** 2
        loss = (loss * weights).sum(dim=-1) / patch_denom

        if self.loss_weighting == "patch_intensity":
            patch_intensity = (target_for_weight * weights).sum(dim=-1) / patch_denom
            masked = mask > 0
            patch_min = torch.where(masked, patch_intensity, torch.inf).amin(dim=1, keepdim=True)
            patch_max = torch.where(masked, patch_intensity, -torch.inf).amax(dim=1, keepdim=True)
            patch_min = torch.where(torch.isfinite(patch_min), patch_min, torch.zeros_like(patch_min))
            patch_max = torch.where(torch.isfinite(patch_max), patch_max, patch_min + 1.0)
            patch_weight = 1.0 + (patch_intensity - patch_min) / (patch_max - patch_min + 1.0e-6)
            loss = (loss * mask * patch_weight).sum() / (mask.sum() + 1.0e-6)
        else:
            loss = (loss * mask).sum() / (mask.sum() + 1.0e-6)
        return loss

    # ── Distribution Matching Loss Helpers ──────────────────────────────────

    @staticmethod
    def _compute_moments(x: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute variance, skewness, kurtosis of a 1D tensor.
        All operations are element-wise or reduction-based, so fully differentiable.
        """
        mean = x.mean()
        diff = x - mean
        var = (diff ** 2).mean()
        std = (var + eps).sqrt()
        skew = (diff ** 3).mean() / (std ** 3)
        kurt = (diff ** 4).mean() / (std ** 4)
        return var, skew, kurt

    def _multipath_distribution_loss(
        self,
        pred_img: torch.Tensor,
        target_img: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-path distribution loss (paper §4.3.1) — the 1D Wasserstein
        distance between predicted and target pixel-value distributions in
        range-axis masked regions.

        *Why Wasserstein for Gamma-like tails*:
        Multi-path reflections under vehicles produce heavy right tails along
        the range axis — a Gamma distribution hallmark.  MSE penalises each
        pixel independently, so the model can satisfy MSE by predicting the
        conditional mean while completely ignoring the tail shape.  The 1D
        Wasserstein (Earth Mover's) distance compares the full sorted quantile
        functions, so any mismatch in tail decay or peak concentration incurs
        a gradient that directly corrects the CDF shape.

        Implementation: W₁(P, Q) = mean(|sort(P) − sort(Q)|) is differentiable
        through ``torch.sort`` (the Jacobian is a permutation matrix) and
        requires no explicit parameter fitting.
        """
        mask = pixel_mask[:, 0] > 0.5
        n = mask.flatten(1).sum(dim=1)

        # Sort + sum reductions in fp32: under fp16 AMP, summing ~30k absolute
        # diffs (each up to O(10) early in training) can exceed fp16 max.
        flat_p = pred_img[:, 0].to(torch.float32).masked_fill(~mask, float("inf")).flatten(1)
        flat_t = target_img[:, 0].to(torch.float32).masked_fill(~mask, float("inf")).flatten(1)
        sort_p, _ = flat_p.sort(dim=1)
        sort_t, _ = flat_t.sort(dim=1)

        idx = torch.arange(sort_p.shape[1], device=sort_p.device).unsqueeze(0)
        valid_pos = idx < n.unsqueeze(1)
        # Zero the +inf padding *after* sort: inf-inf=nan would otherwise
        # propagate even through torch.where (both branches are evaluated).
        sort_p = torch.where(valid_pos, sort_p, sort_p.new_zeros(()))
        sort_t = torch.where(valid_pos, sort_t, sort_t.new_zeros(()))
        diff = (sort_p - sort_t).abs()
        per_sample = diff.sum(dim=1) / n.clamp_min(1).to(diff.dtype)

        weights = (n >= 2).to(per_sample.dtype)
        denom = weights.sum().clamp_min(1.0)
        return (per_sample * weights).sum() / denom

    def _sidelobs_distribution_loss(
        self,
        pred_img: torch.Tensor,
        target_img: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sidelobs distribution loss (paper §4.3.2) — higher-order statistical
        moment matching between predicted and target pixel-value distributions
        in azimuth-axis masked regions.

        *Why moment matching for Weibull-like sidelobss*:
        Antenna sidelobss create extreme spikes (high kurtosis) and periodic
        diffusion patterns (specific skewness) along the azimuth axis — features
        well-characterised by the Weibull extreme-value family.  Pixel-wise MSE
        tends to smooth out these peaks (regression-to-mean effect).

        We match three moments to encourage Weibull-like reconstruction:
        • Variance  — ensures correct energy spread of the sidelobs envelope.
        • Skewness  — captures the asymmetry of sidelobs power distribution.
        • Kurtosis  — penalises failure to reproduce sharp sidelobs peaks that
          MSE alone would average away.

        All moments are computed from standard pytorch ops (mean, pow, sqrt)
        and are therefore fully differentiable via autograd.
        """
        eps = 1e-8
        mask = pixel_mask[:, 0] > 0.5
        # Moments use ^3 / ^4 reductions over ~30k pixels — under fp16 AMP the
        # raw sums can overflow (e.g. peak^4 ≈ 1e4 × 30k ≈ 3e8 ≫ 65504).
        # Cast to fp32 inside this loss only; cost is trivial (small tensors).
        mask_f = mask.to(torch.float32)
        n_safe = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)

        def _moments(x: torch.Tensor):
            x = x[:, 0].to(torch.float32)
            mean = (x * mask_f).flatten(1).sum(dim=1) / n_safe
            diff = (x - mean[:, None, None]) * mask_f
            var = (diff ** 2).flatten(1).sum(dim=1) / n_safe
            std = (var + eps).sqrt()
            skew = (diff ** 3).flatten(1).sum(dim=1) / n_safe / (std ** 3)
            kurt = (diff ** 4).flatten(1).sum(dim=1) / n_safe / (std ** 4)
            return var, skew, kurt

        pv, ps, pk = _moments(pred_img)
        tv, ts, tk = _moments(target_img)

        loss_var = (pv - tv).abs() / (tv.abs() + 1e-6)
        loss_skew = (ps - ts).abs()
        # Kurtosis is inherently larger in magnitude; weight at 0.5 to
        # avoid dominating the gradient.
        loss_kurt = 0.5 * (pk - tk).abs()
        per_sample = loss_var + loss_skew + loss_kurt

        # n_safe = max(actual_count, 1); samples with <4 masked pixels yield
        # meaningless moments and are excluded via this weight, matching the
        # original `if num_masked < 4: continue` semantics.
        weights = (n_safe >= 4.0).to(per_sample.dtype)
        denom = weights.sum().clamp_min(1.0)
        return (per_sample * weights).sum() / denom

    def _patch_mask_to_pixel_mask(self, patch_mask: torch.Tensor) -> torch.Tensor:
        """Convert patch-level mask [B, num_patches] to pixel-level mask [B, 1, H, W]."""
        B = patch_mask.shape[0]
        g_h, g_w = self.patch_embed.grid_size
        p_h, p_w = self.patch_size
        # [B, num_patches] → [B, g_h, g_w] → repeat per patch → [B, 1, H, W]
        mask_2d = patch_mask.reshape(B, g_h, g_w)
        mask_2d = mask_2d[:, :, None, :, None].expand(B, g_h, p_h, g_w, p_w)
        mask_2d = mask_2d.reshape(B, g_h * p_h, g_w * p_w)
        pixel_mask = mask_2d[:, :self.img_size[0], :self.img_size[1]].unsqueeze(1).float()
        return pixel_mask

    def _compute_distribution_losses(
        self,
        pred_patches: torch.Tensor,
        target_imgs: torch.Tensor,
        pixel_mask_range: torch.Tensor,
        pixel_mask_azimuth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hard Routing: apply axis-specific distribution matching losses.
        physical:
        - Range-masked pixels   → multi-path distribution loss   (§4.3.1)
        - Azimuth-masked pixels → sidelobs distribution loss     (§4.3.2)

        swapped:
        - Azimuth-masked pixels → multi-path distribution loss
        - Range-masked pixels   → sidelobs distribution loss
        """
        pred_img = self.unpatchify(pred_patches)                       # [B, 1, H, W]
        target_img = target_imgs[:, :1, :self.img_size[0], :self.img_size[1]]

        if self.dist_loss_axis_routing == "swapped":
            multipath_mask = pixel_mask_azimuth
            sidelobs_mask = pixel_mask_range
        else:
            multipath_mask = pixel_mask_range
            sidelobs_mask = pixel_mask_azimuth

        multipath_loss = self._multipath_distribution_loss(
            pred_img, target_img, multipath_mask.detach(),
        )
        sidelobs_loss = self._sidelobs_distribution_loss(
            pred_img, target_img, sidelobs_mask.detach(),
        )
        return multipath_loss, sidelobs_loss

    def forward(
        self,
        imgs: torch.Tensor,
        target_imgs: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.75,
        importance_scale=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del importance_scale
        if target_imgs is None:
            target_imgs = imgs[:, :1, :, :]
        self.last_profile = {}
        self.last_loss_breakdown = {}
        self.last_manifold_metrics = {}
        self._profile_timings = {} if self.profile_runtime else None

        total_start = self._profile_start()
        latent, mask, ids_restore, block_means = self.forward_encoder(imgs, mask_ratio)
        pred_patches = self.forward_decoder(latent, ids_restore)

        if self.tokenization_mode == "patch":
            # ── Patch mode: mask is already patch-level ──
            stage_start = self._profile_start()
            patch_mask = mask
            self._profile_stop("patch_mask", stage_start)

            self.last_mask_axial = None
            self.last_mask_h = None
            self.last_mask_v = None
            self.last_mask_range = None
            self.last_mask_azimuth = None
            self.last_hint_range = None
            self.last_hint_azimuth = None
            self.last_pixel_mask_h = None
            self.last_pixel_mask_v = None
            self.last_pixel_mask_range = None
            self.last_pixel_mask_azimuth = None
            self.last_pixel_hint_range = None
            self.last_pixel_hint_azimuth = None
            self.last_pixel_mask = None
            self.last_patch_mask = patch_mask.detach()
            patch_hint = getattr(self, "_last_patch_block_hint", None)
            self.last_patch_hint = patch_hint.detach() if isinstance(patch_hint, torch.Tensor) else None
            self.last_fused_map = None
        else:
            # ── Axial mode: convert axial mask → pixel mask → patch mask ──
            mask_axial = mask
            stage_start = self._profile_start()
            mask_h_px, mask_v_px, mask_px = self._axial_mask_to_pixel_masks(mask_axial)
            patch_mask = self._axial_mask_to_patch_mask(
                mask_axial,
                precomputed=(mask_h_px, mask_v_px, mask_px),
            )
            self._profile_stop("patch_mask", stage_start)
            mask_h = mask_axial[:, : self.num_h_tokens]
            mask_v = mask_axial[:, self.num_h_tokens :]
            mask_range, mask_azimuth = self._axial_to_range_azimuth(mask_h, mask_v)
            pixel_mask_range, pixel_mask_azimuth = self._axial_to_range_azimuth(mask_h_px, mask_v_px)
            self.last_mask_axial = mask_axial.detach()
            self.last_mask_h = mask_h.detach()
            self.last_mask_v = mask_v.detach()
            self.last_mask_range = mask_range.detach()
            self.last_mask_azimuth = mask_azimuth.detach()
            hint_range = getattr(self, "_last_block_hint_range", None)
            hint_azimuth = getattr(self, "_last_block_hint_azimuth", None)
            if isinstance(hint_range, torch.Tensor) and isinstance(hint_azimuth, torch.Tensor):
                hint_h, hint_v = self._range_azimuth_to_axial(
                    hint_range.to(device=mask_axial.device, dtype=mask_axial.dtype),
                    hint_azimuth.to(device=mask_axial.device, dtype=mask_axial.dtype),
                )
                hint_axial = torch.cat([hint_h, hint_v], dim=1)
                hint_h_px, hint_v_px, _ = self._axial_mask_to_pixel_masks(hint_axial)
                pixel_hint_range, pixel_hint_azimuth = self._axial_to_range_azimuth(hint_h_px, hint_v_px)
                self.last_hint_range = hint_range.detach()
                self.last_hint_azimuth = hint_azimuth.detach()
                self.last_pixel_hint_range = pixel_hint_range.detach()
                self.last_pixel_hint_azimuth = pixel_hint_azimuth.detach()
            else:
                self.last_hint_range = None
                self.last_hint_azimuth = None
                self.last_pixel_hint_range = None
                self.last_pixel_hint_azimuth = None
            self.last_pixel_mask_h = mask_h_px.detach()
            self.last_pixel_mask_v = mask_v_px.detach()
            self.last_pixel_mask_range = pixel_mask_range.detach()
            self.last_pixel_mask_azimuth = pixel_mask_azimuth.detach()
            self.last_pixel_mask = mask_px.detach()
            self.last_patch_mask = patch_mask.detach()
            self.last_patch_hint = None

        stage_start = self._profile_start()
        reconstruction_loss = self.forward_loss(target_imgs, pred_patches, patch_mask)
        self._profile_stop("forward_loss", stage_start)

        # ── Physical distribution-based loss (§4.3) ──
        multipath_loss = pred_patches.new_zeros(())
        sidelobs_loss = pred_patches.new_zeros(())
        if self.enable_dist_loss:
            stage_start = self._profile_start()
            if self.tokenization_mode == "axial":
                multipath_loss, sidelobs_loss = self._compute_distribution_losses(
                    pred_patches, target_imgs,
                    pixel_mask_range, pixel_mask_azimuth,
                )
            else:
                # Patch mode: convert patch_mask → pixel mask and apply the same loss (weight=1).
                pixel_mask_patch = self._patch_mask_to_pixel_mask(patch_mask)
                multipath_loss, sidelobs_loss = self._compute_distribution_losses(
                    pred_patches, target_imgs,
                    pixel_mask_patch, pixel_mask_patch,
                )
            self._profile_stop("dist_matching_loss", stage_start)
        scaled_multipath_loss = self.lambda_multipath * multipath_loss
        scaled_sidelobs_loss = self.lambda_sidelobs * sidelobs_loss

        loss = reconstruction_loss + scaled_multipath_loss + scaled_sidelobs_loss
        self.last_loss_breakdown = {
            "reconstruction_loss": reconstruction_loss.detach(),
            "multipath_loss": multipath_loss.detach(),
            "multipath_loss_scaled": scaled_multipath_loss.detach(),
            "sidelobs_loss": sidelobs_loss.detach(),
            "sidelobs_loss_scaled": scaled_sidelobs_loss.detach(),
        }

        self._profile_stop("forward_total", total_start)
        if self._profile_timings is not None:
            self.last_profile = dict(self._profile_timings)
            self._profile_timings = None
        return loss, pred_patches, patch_mask


def mae_vit_base_patch16_dec512d8b(patch_size=16, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch8_dec512d8b(patch_size=8, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch32_dec512d8b(patch_size=32, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(patch_size=16, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_dec512d8b(patch_size=14, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_base_patch8 = mae_vit_base_patch8_dec512d8b
mae_vit_base_patch32 = mae_vit_base_patch32_dec512d8b
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b
