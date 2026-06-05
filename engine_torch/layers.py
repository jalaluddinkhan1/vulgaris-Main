"""engine_torch/layers.py — Layers implemented with native PyTorch ops."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine_torch.tensor import Tensor, Parameter, _DEFAULT_DEVICE, _DEFAULT_DTYPE
from engine_torch.module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self._linear = nn.Linear(in_features, out_features, bias=bias)
        self._linear.to(dtype=_DEFAULT_DTYPE, device=_DEFAULT_DEVICE)

    @property
    def weight(self):
        return Parameter(self._linear.weight.data.cpu().numpy())

    @property
    def bias(self):
        if self._linear.bias is None:
            return None
        return Parameter(self._linear.bias.data.cpu().numpy())

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(self._linear(x._t))


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self._norm = nn.RMSNorm(dim, eps=eps)
        self._norm.to(dtype=_DEFAULT_DTYPE, device=_DEFAULT_DEVICE)

    @property
    def weight(self):
        return Parameter(self._norm.weight.data.cpu().numpy())

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(self._norm(x._t))


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self._ln = nn.LayerNorm(dim, eps=eps)
        self._ln.to(dtype=_DEFAULT_DTYPE, device=_DEFAULT_DEVICE)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(self._ln(x._t))


class CausalAttention(Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self._attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, bias=True
        )
        self._attn.to(dtype=_DEFAULT_DTYPE, device=_DEFAULT_DEVICE)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x._t.shape
        mask = torch.triu(torch.ones(T, T, device=x._t.device), diagonal=1).bool()
        out, _ = self._attn(x._t, x._t, x._t, attn_mask=mask)
        return Tensor(out)


class RevIN(Module):
    """Reversible Instance Normalization — thread-safe torch implementation."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps   = eps
        self._affine = affine
        if affine:
            self.weight_p = nn.Parameter(torch.ones(num_features))
            self.bias_p   = nn.Parameter(torch.zeros(num_features))
        # Thread-local stats
        import threading
        object.__setattr__(self, "_tl", threading.local())

    @property
    def weight(self):
        return Parameter(self.weight_p.data.cpu().numpy()) if self._affine else None

    @property
    def bias(self):
        return Parameter(self.bias_p.data.cpu().numpy()) if self._affine else None

    def normalize(self, x: Tensor) -> Tensor:
        t = x._t                                     # (B, C, T)
        mean = t.mean(dim=-1, keepdim=True)           # (B, C, 1)
        std  = t.std(dim=-1, keepdim=True) + self.eps
        self._tl.mean = mean.detach().cpu().numpy()
        self._tl.std  = std.detach().cpu().numpy()
        x_hat = (t - mean) / std
        if self._affine:
            x_hat = x_hat * self.weight_p[None, :, None] + self.bias_p[None, :, None]
        return Tensor(x_hat)

    @property
    def _mean(self):
        return getattr(self._tl, "mean", None)

    @property
    def _std(self):
        return getattr(self._tl, "std", None)

    def denormalize(self, x: Tensor) -> Tensor:
        m = self._tl.mean if hasattr(self._tl, "mean") else None
        s = self._tl.std  if hasattr(self._tl, "std")  else None
        if m is None:
            return x
        mean = torch.from_numpy(m).to(x._t.device)
        std  = torch.from_numpy(s).to(x._t.device)
        return Tensor(x._t * std + mean)

    def forward(self, x: Tensor) -> Tensor:
        return self.normalize(x)
