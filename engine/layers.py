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
