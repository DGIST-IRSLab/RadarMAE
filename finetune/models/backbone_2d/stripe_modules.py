from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@dataclass(frozen=True)
class StripeGrid:
    h: int
    w: int


class StripeLePEAttention(nn.Module):
    """
    1D stripe self-attention with LePE injected by:
      - reshape V to 2D grid (H,W)
      - depthwise conv (k x k) on V
      - add to attention output (outside score path)

    Also supports attention capture for visualization:
      - last_attn_map: (B, L) key-importance averaged over heads/queries.
      - last_attn_matrix: (B, L, L) attention weights averaged over heads.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_hw: Tuple[int, int],
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_scale: Optional[float] = None,
        use_lepe: bool = True,
        lepe_kernel_size: int = 3,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.grid_hw = StripeGrid(int(grid_hw[0]), int(grid_hw[1]))

        head_dim = dim // num_heads
        self.scale = float(qk_scale) if qk_scale is not None else head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_lepe = bool(use_lepe)
        if self.use_lepe:
            k = int(lepe_kernel_size)
            if k % 2 == 0:
                raise ValueError(f"lepe_kernel_size must be odd, got {k}")
            self.lepe_conv = nn.Conv2d(dim, dim, kernel_size=k, stride=1, padding=k // 2, groups=dim)
        else:
            self.lepe_conv = None

        self._capture_attn = False
        self.last_attn_map: Optional[torch.Tensor] = None  # (B, L)
        self.last_attn_matrix: Optional[torch.Tensor] = None  # (B, L, L)

    def set_capture_attn(self, enabled: bool = True):
        self._capture_attn = bool(enabled)
        if not self._capture_attn:
            self.last_attn_map = None
            self.last_attn_matrix = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B,L,C), got {tuple(x.shape)}")
        b, l, c = x.shape
        if c != self.dim:
            raise ValueError(f"Channel mismatch: x.C={c} vs dim={self.dim}")
        h, w = self.grid_hw.h, self.grid_hw.w
        if l != h * w:
            raise ValueError(f"Stripe grid mismatch: L={l} vs H*W={h*w} (H={h}, W={w})")

        qkv = self.qkv(x).reshape(b, l, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, L, head_dim)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        if self._capture_attn:
            key_imp = attn.mean(dim=1).mean(dim=2)  # (B, L)
            self.last_attn_map = key_imp.detach()
            self.last_attn_matrix = attn.mean(dim=1).detach()  # (B, L, L)

        out = attn @ v  # (B, heads, L, head_dim)

        if self.use_lepe and self.lepe_conv is not None:
            v_full = v.permute(0, 2, 1, 3).contiguous().view(b, l, c)  # (B, L, C)
            v_2d = v_full.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            lepe = self.lepe_conv(v_2d)  # (B, C, H, W)
            lepe = lepe.permute(0, 2, 3, 1).contiguous().view(b, l, c)  # (B, L, C)
            lepe = lepe.view(b, l, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
            out = out + lepe

        out = out.transpose(1, 2).contiguous().view(b, l, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class StripeTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_hw: Tuple[int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        use_lepe: bool = True,
        lepe_kernel_size: int = 3,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = StripeLePEAttention(
            dim=dim,
            num_heads=num_heads,
            grid_hw=grid_hw,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_scale=qk_scale,
            use_lepe=use_lepe,
            lepe_kernel_size=lepe_kernel_size,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop)

    def set_capture_attn(self, enabled: bool = True):
        self.attn.set_capture_attn(enabled)

    @property
    def last_attn_map(self) -> Optional[torch.Tensor]:
        return self.attn.last_attn_map

    @property
    def last_attn_matrix(self) -> Optional[torch.Tensor]:
        return self.attn.last_attn_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DualStreamStripeBlock(nn.Module):
    """
    Parallel horizontal/vertical stripe blocks with lightweight interaction at the end of each layer.

    Interaction (concat + projection):
      h <- h + W_h([h, mean(v)])
      v <- v + W_v([v, mean(h)])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_hw_h: Tuple[int, int],
        grid_hw_v: Tuple[int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        use_lepe: bool = True,
        lepe_kernel_size: int = 3,
    ):
        super().__init__()
        self.h_blk = StripeTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            grid_hw=grid_hw_h,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_lepe=use_lepe,
            lepe_kernel_size=lepe_kernel_size,
        )
        self.v_blk = StripeTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            grid_hw=grid_hw_v,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_lepe=use_lepe,
            lepe_kernel_size=lepe_kernel_size,
        )

        self.h_fuse = nn.Linear(dim * 2, dim, bias=True)
        self.v_fuse = nn.Linear(dim * 2, dim, bias=True)

        self._capture_attn = False
        self.last_attn_maps: Optional[dict] = None  # keys: horizontal/vertical + *_attn

    def set_capture_attn(self, enabled: bool = True):
        self._capture_attn = bool(enabled)
        self.h_blk.set_capture_attn(enabled)
        self.v_blk.set_capture_attn(enabled)
        if not enabled:
            self.last_attn_maps = None

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.h_blk(h)
        v = self.v_blk(v)

        h_ctx = h.mean(dim=1, keepdim=True)
        v_ctx = v.mean(dim=1, keepdim=True)

        h = h + self.h_fuse(torch.cat([h, v_ctx.expand(-1, h.shape[1], -1)], dim=-1))
        v = v + self.v_fuse(torch.cat([v, h_ctx.expand(-1, v.shape[1], -1)], dim=-1))

        if self._capture_attn:
            self.last_attn_maps = {
                "horizontal": self.h_blk.last_attn_map,
                "vertical": self.v_blk.last_attn_map,
                "horizontal_attn": self.h_blk.last_attn_matrix,
                "vertical_attn": self.v_blk.last_attn_matrix,
            }
        return h, v

