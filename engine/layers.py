import numpy as np
from typing import Tuple, Union, Optional
from .tensor import Tensor, Parameter, zeros, ones
from .module import Module
from . import ops


class Linear(Module):
    """Fully-connected linear layer: y = x @ W.T + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = np.sqrt(1.0 / in_features)
        w_data = np.random.uniform(-bound, bound, (out_features, in_features))
        self.weight = Parameter(w_data, name="weight")
        if bias:
            b_data = np.random.uniform(-bound, bound, (out_features,))
            self.bias = Parameter(b_data, name="bias")
        else:
            # Store None without triggering Module registration
            object.__setattr__(self, "bias", None)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return (f"Linear(in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, n_params={self.n_params():,})")


class LayerNorm(Module):
    """Layer normalization with learnable affine parameters."""

    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(np.ones(normalized_shape, dtype=np.float32), name="weight")
        self.bias = Parameter(np.zeros(normalized_shape, dtype=np.float32), name="bias")

    def forward(self, x: Tensor) -> Tensor:
        return x.layer_norm(self.normalized_shape, self.weight, self.bias, self.eps)

    def __repr__(self) -> str:
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"


class Dropout(Module):
    """Dropout regularization with inverted scaling."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32)
        scale = 1.0 / (1.0 - self.p)
        mask_t = Tensor(mask * scale, requires_grad=False)
        return x * mask_t

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class Conv1d(Module):
    """1D convolution: x (B, C_in, L) -> y (B, C_out, L_out)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, groups: int = 1, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        bound = np.sqrt(1.0 / (in_channels // groups * kernel_size))
        w_data = np.random.uniform(-bound, bound,
                                   (out_channels, in_channels // groups, kernel_size))
        self.weight = Parameter(w_data, name="weight")
        if bias:
            b_data = np.random.uniform(-bound, bound, (out_channels,))
            self.bias = Parameter(b_data, name="bias")
        else:
            object.__setattr__(self, "bias", None)

    def forward(self, x: Tensor) -> Tensor:
        return ops.conv1d_forward(x, self.weight,
                                  self.bias if self.bias is not None else None,
                                  self.stride, self.padding, self.groups)

    def __repr__(self) -> str:
        return (f"Conv1d({self.in_channels}, {self.out_channels}, "
                f"kernel={self.kernel_size}, stride={self.stride}, padding={self.padding})")


class Embedding(Module):
    """Learnable embedding lookup table."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        w_data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        self.weight = Parameter(w_data, name="weight")

    def forward(self, idx) -> Tensor:
        if isinstance(idx, Tensor):
            idx_data = idx.data.astype(int)
        else:
            idx_data = np.asarray(idx, dtype=int)

        out_data = self.weight.data[idx_data]
        out = Tensor(out_data, requires_grad=self.weight.requires_grad,
                     _children=(self.weight,), _op="embedding")

        weight = self.weight

        def _back():
            if weight.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out_data.shape)
                contrib = np.zeros_like(weight.data)
                np.add.at(contrib, idx_data, g)
                weight.grad = weight.grad + contrib if weight.grad is not None else contrib

        out._backward = _back
        return out

    def __repr__(self) -> str:
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


class RMSNorm(Module):
    """Root Mean Square Layer Normalization (no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(np.ones(dim, dtype=np.float32), name="weight")

    def forward(self, x: Tensor) -> Tensor:
        # rms = sqrt(mean(x^2) + eps)
        sq = x * x
        ms = sq.mean(axis=-1, keepdims=True)
        # We need sqrt(ms + eps); build as Tensor op
        rms_data = np.sqrt(ms.data + self.eps)
        rms_t = Tensor(rms_data, requires_grad=x.requires_grad,
                       _children=(ms,), _op="rms")

        def _rms_back():
            if ms.requires_grad:
                g = rms_t.grad if rms_t.grad is not None else np.ones(rms_data.shape)
                contrib = g * 0.5 / rms_data
                ms.grad = ms.grad + contrib if ms.grad is not None else contrib

        rms_t._backward = _rms_back

        x_norm = x / rms_t
        return x_norm * self.weight

    def __repr__(self) -> str:
        return f"RMSNorm({self.dim}, eps={self.eps})"


class Sequential(Module):
    """Sequential container of modules."""

    def __init__(self, *modules_list):
        super().__init__()
        for i, mod in enumerate(modules_list):
            # Register with string index as name
            mods = object.__getattribute__(self, "_modules")
            mods[str(i)] = mod
            object.__setattr__(self, str(i), mod)

    def forward(self, x: Tensor) -> Tensor:
        for mod in self._modules.values():
            x = mod(x)
        return x

    def __repr__(self) -> str:
        lines = ["Sequential("]
        for i, mod in self._modules.items():
            lines.append(f"  ({i}): {repr(mod)}")
        lines.append(")")
        return "\n".join(lines)


class SwiGLU(Module):
    """Gated MLP: gate=silu(W1(x)), val=W2(x), out=W3(gate*val)"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, bias=False)  # gate projection
        self.w2 = Linear(d_model, d_ff, bias=False)  # value projection
        self.w3 = Linear(d_ff, d_model, bias=False)  # output projection

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w1(x).silu()
        val = self.w2(x)
        return self.w3(gate * val)

    def __repr__(self) -> str:
        return f"SwiGLU(d_model={self.d_model}, d_ff={self.d_ff}, n_params={self.n_params():,})"


# ---------------------------------------------------------------------------
# New classes — appended; existing code above is untouched.
# ---------------------------------------------------------------------------
import math as _math


class CausalAttention(Module):
    """Multi-head causal (masked) self-attention.

    Parameters
    ----------
    d_model  : input/output dimension
    n_heads  : number of attention heads (d_model must be divisible by n_heads)
    dropout  : attention-weight dropout probability (default 0.0, currently
               applied as a simple inverted-scaling mask like Dropout)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / _math.sqrt(self.head_dim)

        self.q_proj = Linear(d_model, d_model, bias=False)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)
        self.out_proj = Linear(d_model, d_model, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.data.shape
        nh = self.n_heads
        hd = self.head_dim
        scale = self.scale

        # 1. Project
        Q = self.q_proj(x)   # (B, T, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. Reshape to (B, nh, T, hd) in numpy
        q_np = Q.data.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)  # (B, nh, T, hd)
        k_np = K.data.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        v_np = V.data.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)

        # 3. Scaled dot-product scores  (B, nh, T, T)
        scores_np = q_np @ k_np.transpose(0, 1, 3, 2) * scale

        # 4. Causal mask
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores_np[..., mask] = -1e9

        # 5. Softmax (row-wise)
        scores_np -= scores_np.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores_np)
        attn_np = exp_s / exp_s.sum(axis=-1, keepdims=True)  # (B, nh, T, T)

        # 6. Optional dropout on attention weights
        if self.training and self.dropout > 0.0:
            drop_mask = (np.random.rand(*attn_np.shape) > self.dropout).astype(np.float32)
            attn_np = attn_np * drop_mask / (1.0 - self.dropout)

        # 7. Weighted sum over values  (B, nh, T, hd)
        out_np = attn_np @ v_np

        # 8. Reshape back to (B, T, d_model)
        out_np = out_np.transpose(0, 2, 1, 3).reshape(B, T, nh * hd)

        # 9. Wrap output as Tensor; custom backward accumulates into Q, K, V
        requires_grad = Q.requires_grad or K.requires_grad or V.requires_grad
        out_tensor = Tensor(
            out_np.astype(np.float32),
            requires_grad=requires_grad,
            _children=(Q, K, V),
            _op="causal_attn",
        )

        # Capture loop-invariant values for the closure
        _attn_np = attn_np
        _q_np = q_np
        _k_np = k_np
        _v_np = v_np
        _mask = mask
        _B, _T, _nh, _hd = B, T, nh, hd
        _scale = scale

        def _back():
            g = out_tensor.grad if out_tensor.grad is not None else np.ones(out_np.shape, dtype=np.float32)
            # g: (B, T, d_model) -> reshape to (B, nh, T, hd)
            g_out = g.reshape(_B, _T, _nh, _hd).transpose(0, 2, 1, 3)  # (B, nh, T, hd)

            # dV = attn^T @ g_out   (B, nh, T, hd)
            dV_np = _attn_np.transpose(0, 1, 3, 2) @ g_out

            # dA = g_out @ V^T      (B, nh, T, T)
            dA_np = g_out @ _v_np.transpose(0, 1, 3, 2)

            # Softmax backward: ds[i] = a[i] * (dA[i] - sum(dA * a, axis=-1, keepdims))
            dS_np = _attn_np * (dA_np - (dA_np * _attn_np).sum(axis=-1, keepdims=True))

            # Zero out future positions in gradient
            dS_np[..., _mask] = 0.0

            # Apply scale
            dS_np = dS_np * _scale  # (B, nh, T, T)

            # dQ = dS @ K   (B, nh, T, hd)
            dQ_np = dS_np @ _k_np
            # dK = dS^T @ Q  (B, nh, T, hd)
            dK_np = dS_np.transpose(0, 1, 3, 2) @ _q_np

            # Reshape back to (B, T, d_model)
            dQ_flat = dQ_np.transpose(0, 2, 1, 3).reshape(_B, _T, _nh * _hd).astype(np.float32)
            dK_flat = dK_np.transpose(0, 2, 1, 3).reshape(_B, _T, _nh * _hd).astype(np.float32)
            dV_flat = dV_np.transpose(0, 2, 1, 3).reshape(_B, _T, _nh * _hd).astype(np.float32)

            if Q.requires_grad:
                Q.grad = (Q.grad + dQ_flat) if Q.grad is not None else dQ_flat
            if K.requires_grad:
                K.grad = (K.grad + dK_flat) if K.grad is not None else dK_flat
            if V.requires_grad:
                V.grad = (V.grad + dV_flat) if V.grad is not None else dV_flat

        out_tensor._backward = _back

        # 10. Final output projection
        return self.out_proj(out_tensor)

    def __repr__(self) -> str:
        return (f"CausalAttention(d_model={self.d_model}, n_heads={self.n_heads}, "
                f"dropout={self.dropout}, n_params={self.n_params():,})")


class PatchEmbed(Module):
    """Split a (B, C, T) time-series into non-overlapping patches and project.

    Parameters
    ----------
    in_channels : number of input channels C
    patch_size  : number of timesteps per patch P
    d_model     : output embedding dimension
    stride      : patch stride (default = patch_size for non-overlapping patches)
    """

    def __init__(self, in_channels: int, patch_size: int, d_model: int,
                 stride: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.stride = stride if stride is not None else patch_size

        self.proj = Linear(in_channels * patch_size, d_model, bias=True)
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, C, T = x.data.shape
        P = self.patch_size
        S = self.stride
        n_patches = (T - P) // S + 1

        # 1. Extract patches in numpy -> (B, n_patches, C*P)
        patches_list = []
        for i in range(n_patches):
            patch = x.data[:, :, i * S: i * S + P]          # (B, C, P)
            patches_list.append(patch.reshape(B, C * P))     # (B, C*P)
        patches_np = np.stack(patches_list, axis=1).astype(np.float32)  # (B, n_patches, C*P)

        # 2. Wrap as Tensor with backward that scatters grads back to x
        patches_t = Tensor(
            patches_np,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="patch_embed",
        )

        _B, _C, _T, _P, _S, _n = B, C, T, P, S, n_patches

        def _back():
            g = patches_t.grad if patches_t.grad is not None else np.ones(patches_np.shape, dtype=np.float32)
            # g: (B, n_patches, C*P)
            if x.requires_grad:
                dx = np.zeros((_B, _C, _T), dtype=np.float32)
                for i in range(_n):
                    # patch_grad[b, i, :] reshaped to (C, P) and scattered
                    patch_g = g[:, i, :].reshape(_B, _C, _P)   # (B, C, P)
                    dx[:, :, i * _S: i * _S + _P] += patch_g
                x.grad = (x.grad + dx) if x.grad is not None else dx

        patches_t._backward = _back

        # 3. Project then normalise
        out = self.proj(patches_t)   # (B, n_patches, d_model)
        out = self.norm(out)
        return out

    def __repr__(self) -> str:
        return (f"PatchEmbed(in_channels={self.in_channels}, patch_size={self.patch_size}, "
                f"d_model={self.d_model}, stride={self.stride}, n_params={self.n_params():,})")


class RevIN(Module):
    """Reversible Instance Normalization for non-stationary time-series.

    Parameters
    ----------
    num_features : number of channels C
    eps          : numerical stability epsilon (default 1e-5)
    affine       : learnable per-channel affine scale/shift (default True)

    Usage
    -----
    rev = RevIN(C)
    x_norm = rev.normalize(x)   # or rev(x) — same thing
    # ... run model ...
    x_pred = rev.denormalize(y_norm)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = Parameter(np.ones(num_features, dtype=np.float32), name="weight")
            self.bias = Parameter(np.zeros(num_features, dtype=np.float32), name="bias")
        else:
            object.__setattr__(self, "weight", None)
            object.__setattr__(self, "bias", None)

        # Thread-local storage for per-request mean/std so concurrent inference
        # requests cannot overwrite each other's statistics.
        import threading
        object.__setattr__(self, "_tl", threading.local())

    @property
    def _mean(self) -> Optional[np.ndarray]:
        return getattr(self._tl, "mean", None)

    @_mean.setter
    def _mean(self, v: Optional[np.ndarray]):
        self._tl.mean = v

    @property
    def _std(self) -> Optional[np.ndarray]:
        return getattr(self._tl, "std", None)

    @_std.setter
    def _std(self, v: Optional[np.ndarray]):
        self._tl.std = v

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize x: (B, C, T) -> (B, C, T).  Stores mean/std for denormalize."""
        # 1. Statistics over time axis
        mean = x.data.mean(axis=-1, keepdims=True)          # (B, C, 1)
        std = x.data.std(axis=-1, keepdims=True) + self.eps  # (B, C, 1)

        # 2. Store thread-locally — safe in multi-threaded serving
        self._mean = mean
        self._std = std

        # 3. Normalize
        x_hat_np = ((x.data - mean) / std).astype(np.float32)

        x_hat = Tensor(
            x_hat_np,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="revin_norm",
        )

        _std = std

        def _back():
            g = x_hat.grad if x_hat.grad is not None else np.ones(x_hat_np.shape, dtype=np.float32)
            if x.requires_grad:
                dx = (g / _std).astype(np.float32)
                x.grad = (x.grad + dx) if x.grad is not None else dx

        x_hat._backward = _back

        # 4. Learnable affine:  weight (C,) and bias (C,) broadcast over (B, C, T)
        if self.affine:
            # weight/bias are (C,) -> reshape to (C, 1) for (B, C, T) broadcast
            w = self.weight                         # Parameter, shape (C,)
            b = self.bias                           # Parameter, shape (C,)

            # Reshape to (1, C, 1) via numpy then wrap as Tensor for grad flow
            w_np = w.data.reshape(1, self.num_features, 1).astype(np.float32)
            b_np = b.data.reshape(1, self.num_features, 1).astype(np.float32)

            w_t = Tensor(w_np, requires_grad=w.requires_grad, _children=(w,), _op="revin_w_reshape")
            b_t = Tensor(b_np, requires_grad=b.requires_grad, _children=(b,), _op="revin_b_reshape")

            def _w_back():
                g = w_t.grad if w_t.grad is not None else np.ones(w_np.shape, dtype=np.float32)
                contrib = g.reshape(self.num_features).astype(np.float32)
                w.grad = (w.grad + contrib) if w.grad is not None else contrib

            def _b_back():
                g = b_t.grad if b_t.grad is not None else np.ones(b_np.shape, dtype=np.float32)
                contrib = g.reshape(self.num_features).astype(np.float32)
                b.grad = (b.grad + contrib) if b.grad is not None else contrib

            w_t._backward = _w_back
            b_t._backward = _b_back

            x_hat = x_hat * w_t + b_t

        return x_hat

    def denormalize(self, x: Tensor) -> Tensor:
        """Reverse the normalization applied by normalize().  x: (B, C, T)."""
        assert self._mean is not None and self._std is not None, (
            "denormalize() called before normalize()"
        )
        mean = self._mean   # (B, C, 1)
        std = self._std     # (B, C, 1)

        x_orig_np = (x.data * std + mean).astype(np.float32)

        x_orig = Tensor(
            x_orig_np,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="revin_denorm",
        )

        _std = std

        def _back():
            g = x_orig.grad if x_orig.grad is not None else np.ones(x_orig_np.shape, dtype=np.float32)
            if x.requires_grad:
                dx = (g * _std).astype(np.float32)
                x.grad = (x.grad + dx) if x.grad is not None else dx

        x_orig._backward = _back
        return x_orig

    def forward(self, x: Tensor) -> Tensor:
        """forward() calls normalize() so RevIN can be used as a plain Module."""
        return self.normalize(x)

    def __repr__(self) -> str:
        return (f"RevIN(num_features={self.num_features}, eps={self.eps}, "
                f"affine={self.affine})")


class CrossAttention(Module):
    """Multi-head cross-attention: query attends to external key/value context.

    No causal mask — every query position can attend to every context position.

    Parameters
    ----------
    d_model  : query and output dimension
    n_heads  : number of attention heads
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / _math.sqrt(self.head_dim)

        self.q_proj   = Linear(d_model, d_model, bias=False)
        self.k_proj   = Linear(d_model, d_model, bias=False)
        self.v_proj   = Linear(d_model, d_model, bias=False)
        self.out_proj = Linear(d_model, d_model, bias=True)
        self.norm_q   = RMSNorm(d_model)
        self.norm_ctx = RMSNorm(d_model)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        """
        query   : (B, T_q, d_model)
        context : (B, T_ctx, d_model)
        Returns : (B, T_q, d_model)
        """
        B, T_q, _ = query.data.shape
        T_ctx = context.data.shape[1]
        nh = self.n_heads
        hd = self.head_dim
        scale = self.scale

        # Pre-norm
        q_n = self.norm_q(query)
        c_n = self.norm_ctx(context)

        Q = self.q_proj(q_n)    # (B, T_q,   d_model)
        K = self.k_proj(c_n)    # (B, T_ctx, d_model)
        V = self.v_proj(c_n)    # (B, T_ctx, d_model)

        # Reshape to (B, nh, T, hd)
        q_np = Q.data.reshape(B, T_q,   nh, hd).transpose(0, 2, 1, 3)
        k_np = K.data.reshape(B, T_ctx, nh, hd).transpose(0, 2, 1, 3)
        v_np = V.data.reshape(B, T_ctx, nh, hd).transpose(0, 2, 1, 3)

        # Scaled dot-product  (B, nh, T_q, T_ctx)
        scores_np = q_np @ k_np.transpose(0, 1, 3, 2) * scale

        # Softmax over context positions
        scores_np -= scores_np.max(axis=-1, keepdims=True)
        exp_s  = np.exp(scores_np)
        attn_np = exp_s / (exp_s.sum(axis=-1, keepdims=True) + 1e-8)

        # Weighted sum  (B, nh, T_q, hd)
        out_np = attn_np @ v_np
        out_np = out_np.transpose(0, 2, 1, 3).reshape(B, T_q, nh * hd).astype(np.float32)

        requires_grad = Q.requires_grad or K.requires_grad or V.requires_grad
        out_t = Tensor(out_np, requires_grad=requires_grad,
                       _children=(Q, K, V), _op="cross_attn")

        _attn = attn_np; _q_np = q_np; _k_np = k_np; _v_np = v_np
        _B, _Tq, _Tc, _nh, _hd, _sc = B, T_q, T_ctx, nh, hd, scale

        def _back():
            g = out_t.grad if out_t.grad is not None else np.ones(out_np.shape, np.float32)
            g_out = g.reshape(_B, _Tq, _nh, _hd).transpose(0, 2, 1, 3)  # (B, nh, Tq, hd)

            dV_np = _attn.transpose(0, 1, 3, 2) @ g_out                  # (B, nh, Tc, hd)
            dA_np = g_out @ _v_np.transpose(0, 1, 3, 2)                   # (B, nh, Tq, Tc)
            dS_np = _attn * (dA_np - (dA_np * _attn).sum(axis=-1, keepdims=True))
            dS_np = dS_np * _sc

            dQ_np = dS_np @ _k_np                                          # (B, nh, Tq, hd)
            dK_np = dS_np.transpose(0, 1, 3, 2) @ _q_np                   # (B, nh, Tc, hd)

            dQ_flat = dQ_np.transpose(0,2,1,3).reshape(_B,_Tq,_nh*_hd).astype(np.float32)
            dK_flat = dK_np.transpose(0,2,1,3).reshape(_B,_Tc,_nh*_hd).astype(np.float32)
            dV_flat = dV_np.transpose(0,2,1,3).reshape(_B,_Tc,_nh*_hd).astype(np.float32)

            if Q.requires_grad:
                Q.grad = (Q.grad + dQ_flat) if Q.grad is not None else dQ_flat
            if K.requires_grad:
                K.grad = (K.grad + dK_flat) if K.grad is not None else dK_flat
            if V.requires_grad:
                V.grad = (V.grad + dV_flat) if V.grad is not None else dV_flat

        out_t._backward = _back
        return self.out_proj(out_t)

    def __repr__(self) -> str:
        return (f"CrossAttention(d_model={self.d_model}, n_heads={self.n_heads}, "
                f"n_params={self.n_params():,})")
