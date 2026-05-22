import numpy as np
from typing import Tuple, Optional, Union, List, Set
from scipy.special import erf


def _unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Sum out dimensions that were broadcast to reach grad.shape from target_shape."""
    if grad.shape == target_shape:
        return grad
    # Pad target_shape on the left with 1s to match grad ndim
    ndim_diff = grad.ndim - len(target_shape)
    padded = (1,) * ndim_diff + tuple(target_shape)
    # Sum over axes where target was 1 (i.e., was broadcast)
    axes_to_sum = tuple(i for i, (g, t) in enumerate(zip(grad.shape, padded)) if t == 1)
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)
    # Remove the leading dims we padded
    if ndim_diff > 0:
        grad = grad.reshape(target_shape)
    else:
        grad = grad.reshape(target_shape)
    return grad


class Tensor:
    """Tape-based reverse-mode automatic differentiation tensor."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: tuple = (),
        _op: str = "",
    ):
        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, np.ndarray) and data.dtype == np.float32:
            self.data = data
        else:
            self.data = np.asarray(data, dtype=np.float32)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad: bool = requires_grad
        self._backward = lambda: None
        self._prev: tuple = _children
        self._op: str = _op

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad,
                     _children=(self,), _op="reshape")
        orig_shape = self.data.shape

        def _back():
            if self.requires_grad:
                g = _unbroadcast(out.grad, out.data.shape) if out.grad is not None else np.zeros(out.data.shape)
                contrib = g.reshape(orig_shape)
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def transpose(self, axes=None) -> "Tensor":
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad,
                     _children=(self,), _op="transpose")
        if axes is None:
            inv_axes = None
        else:
            inv_axes = np.argsort(axes)

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.zeros(out.data.shape)
                contrib = np.transpose(g, inv_axes)
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def squeeze(self, axis=None) -> "Tensor":
        orig_shape = self.data.shape
        out = Tensor(np.squeeze(self.data, axis=axis), requires_grad=self.requires_grad,
                     _children=(self,), _op="squeeze")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.zeros(out.data.shape)
                contrib = g.reshape(orig_shape)
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def unsqueeze(self, axis) -> "Tensor":
        orig_shape = self.data.shape
        out = Tensor(np.expand_dims(self.data, axis=axis), requires_grad=self.requires_grad,
                     _children=(self,), _op="unsqueeze")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.zeros(out.data.shape)
                contrib = g.reshape(orig_shape)
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------
    def __add__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(np.full(self.data.shape, other, dtype=np.float32))
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="+")

        def _back():
            g = out.grad if out.grad is not None else np.ones(out.data.shape)
            if self.requires_grad:
                contrib = _unbroadcast(g, self.data.shape)
                self.grad = self.grad + contrib if self.grad is not None else contrib
            if other.requires_grad:
                contrib = _unbroadcast(g, other.data.shape)
                other.grad = other.grad + contrib if other.grad is not None else contrib

        out._backward = _back
        return out

    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad,
                     _children=(self,), _op="neg")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                contrib = -g
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def __sub__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(np.full(self.data.shape, other, dtype=np.float32))
        return self.__add__(-other)

    def __rsub__(self, other) -> "Tensor":
        return (-self).__add__(other)

    def __mul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(np.full(self.data.shape, other, dtype=np.float32))
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="*")

        def _back():
            g = out.grad if out.grad is not None else np.ones(out.data.shape)
            if self.requires_grad:
                contrib = _unbroadcast(g * other.data, self.data.shape)
                self.grad = self.grad + contrib if self.grad is not None else contrib
            if other.requires_grad:
                contrib = _unbroadcast(g * self.data, other.data.shape)
                other.grad = other.grad + contrib if other.grad is not None else contrib

        out._backward = _back
        return out

    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(np.full(self.data.shape, other, dtype=np.float32))
        out = Tensor(self.data / other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="/")

        def _back():
            g = out.grad if out.grad is not None else np.ones(out.data.shape)
            if self.requires_grad:
                contrib = _unbroadcast(g / other.data, self.data.shape)
                self.grad = self.grad + contrib if self.grad is not None else contrib
            if other.requires_grad:
                # d/d(other) of self/other = -self/other^2
                contrib = _unbroadcast(-g * self.data / (other.data ** 2), other.data.shape)
                other.grad = other.grad + contrib if other.grad is not None else contrib

        out._backward = _back
        return out

    def __rtruediv__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(np.full(self.data.shape, other, dtype=np.float32))
        return other.__truediv__(self)

    def __matmul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="@")

        def _back():
            g = out.grad if out.grad is not None else np.ones(out.data.shape)
            if self.requires_grad:
                # grad_self = g @ other.T (handle batch dims)
                if other.data.ndim == 1:
                    contrib = np.outer(g, other.data) if g.ndim > 0 else g * other.data
                elif g.ndim == 1 and other.data.ndim == 2:
                    contrib = np.outer(g, other.data.T).sum(axis=0) if False else g @ other.data.T
                else:
                    # swap last two dims of other
                    ax = list(range(other.data.ndim))
                    ax[-2], ax[-1] = ax[-1], ax[-2]
                    contrib = g @ np.transpose(other.data, ax)
                contrib = _unbroadcast(contrib, self.data.shape)
                self.grad = self.grad + contrib if self.grad is not None else contrib
            if other.requires_grad:
                if self.data.ndim == 1:
                    contrib = np.outer(self.data, g) if g.ndim > 0 else self.data * g
                elif g.ndim == 1 and self.data.ndim == 2:
                    contrib = self.data.T @ g
                else:
                    ax = list(range(self.data.ndim))
                    ax[-2], ax[-1] = ax[-1], ax[-2]
                    contrib = np.transpose(self.data, ax) @ g
                contrib = _unbroadcast(contrib, other.data.shape)
                other.grad = other.grad + contrib if other.grad is not None else contrib

        out._backward = _back
        return out

    def __pow__(self, exponent) -> "Tensor":
        out = Tensor(self.data ** exponent,
                     requires_grad=self.requires_grad,
                     _children=(self,), _op=f"**{exponent}")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                contrib = exponent * (self.data ** (exponent - 1)) * g
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op="sum")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis=axis)
                contrib = np.broadcast_to(g, self.data.shape).copy()
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        n = self.data.size if axis is None else self.data.shape[axis] if not hasattr(axis, '__len__') else np.prod([self.data.shape[a] for a in axis])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        result = self.data.max(axis=axis, keepdims=True)
        out = Tensor(result if keepdims else result.squeeze(axis=axis),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op="max")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                g_expanded = g if keepdims else np.expand_dims(g, axis=axis) if axis is not None else g.reshape(result.shape)
                # Distribute gradient equally among ties
                mask = (self.data == result).astype(np.float32)
                tie_counts = mask.sum(axis=axis, keepdims=True)
                contrib = mask * (g_expanded / np.maximum(tie_counts, 1.0))
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def norm(self, p: float = 2, axis=None, keepdims: bool = False) -> "Tensor":
        if p == 2:
            sq = self * self
            s = sq.sum(axis=axis, keepdims=keepdims)
            return (s + 1e-12) ** 0.5
        elif p == 1:
            return self.abs().sum(axis=axis, keepdims=keepdims)
        else:
            return (self.abs() ** p).sum(axis=axis, keepdims=keepdims) ** (1.0 / p)

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------
    def exp(self) -> "Tensor":
        out_data = np.exp(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _children=(self,), _op="exp")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                contrib = g * out_data
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def log(self) -> "Tensor":
        clipped = np.clip(self.data, 1e-12, None)
        out = Tensor(np.log(clipped), requires_grad=self.requires_grad,
                     _children=(self,), _op="log")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                contrib = g / clipped
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad,
                     _children=(self,), _op="tanh")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                contrib = g * (1.0 - t ** 2)
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, requires_grad=self.requires_grad,
                     _children=(self,), _op="sigmoid")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                contrib = g * s * (1.0 - s)
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def relu(self) -> "Tensor":
        mask = self.data > 0
        out = Tensor(self.data * mask, requires_grad=self.requires_grad,
                     _children=(self,), _op="relu")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                contrib = g * mask.astype(np.float32)
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def silu(self) -> "Tensor":
        # SiLU(x) = x * sigmoid(x)
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        val = self.data * s
        out = Tensor(val, requires_grad=self.requires_grad,
                     _children=(self,), _op="silu")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                # d/dx [x*sig(x)] = sig(x) + x*sig(x)*(1-sig(x)) = sig(x)*(1 + x*(1-sig(x)))
                contrib = g * s * (1.0 + self.data * (1.0 - s))
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def gelu(self) -> "Tensor":
        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        sqrt2 = np.sqrt(2.0)
        cdf = 0.5 * (1.0 + erf(self.data / sqrt2))
        val = self.data * cdf
        out = Tensor(val, requires_grad=self.requires_grad,
                     _children=(self,), _op="gelu")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                # d/dx[x*Phi(x)] = Phi(x) + x*phi(x), where phi is standard normal PDF
                pdf = np.exp(-0.5 * self.data ** 2) / np.sqrt(2.0 * np.pi)
                contrib = g * (cdf + self.data * pdf)
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad,
                     _children=(self,), _op="softmax")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                # Jacobian-vector product: g - (g*s).sum(axis) * s
                dot = (g * s).sum(axis=axis, keepdims=True)
                contrib = s * (g - dot)
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def abs(self) -> "Tensor":
        sign = np.sign(self.data)
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad,
                     _children=(self,), _op="abs")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                contrib = g * sign
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def clip(self, min_val, max_val) -> "Tensor":
        clipped = np.clip(self.data, min_val, max_val)
        mask = (self.data >= min_val) & (self.data <= max_val)
        out = Tensor(clipped, requires_grad=self.requires_grad,
                     _children=(self,), _op="clip")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                contrib = g * mask.astype(np.float32)
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def layer_norm(self, normalized_shape, weight: "Tensor" = None,
                   bias: "Tensor" = None, eps: float = 1e-5) -> "Tensor":
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        axis = tuple(range(-len(normalized_shape), 0))
        mu = self.data.mean(axis=axis, keepdims=True)
        var = self.data.var(axis=axis, keepdims=True)
        std = np.sqrt(var + eps)
        x_hat = (self.data - mu) / std

        w_data = weight.data if weight is not None else np.ones(normalized_shape)
        b_data = bias.data if bias is not None else np.zeros(normalized_shape)
        out_data = x_hat * w_data + b_data

        children = [self]
        if weight is not None:
            children.append(weight)
        if bias is not None:
            children.append(bias)

        out = Tensor(out_data,
                     requires_grad=self.requires_grad or
                     (weight is not None and weight.requires_grad) or
                     (bias is not None and bias.requires_grad),
                     _children=tuple(children), _op="layer_norm")

        def _back():
            g = out.grad if out.grad is not None else np.ones(out.data.shape)
            n = np.prod([self.data.shape[i] for i in range(self.data.ndim) if i not in [self.data.ndim + a for a in axis]])
            # Actually compute normalized_shape product
            norm_size = np.prod(normalized_shape)

            if weight is not None and weight.requires_grad:
                contrib_w = _unbroadcast(g * x_hat, normalized_shape)
                weight.grad = weight.grad + contrib_w if weight.grad is not None else contrib_w
            if bias is not None and bias.requires_grad:
                contrib_b = _unbroadcast(g, normalized_shape)
                bias.grad = bias.grad + contrib_b if bias.grad is not None else contrib_b

            if self.requires_grad:
                # Backprop through normalization
                g_xhat = g * w_data
                # d(x_hat)/d(x): standard layer norm backward
                # dL/dx = (1/std) * (g_xhat - mean(g_xhat) - x_hat * mean(g_xhat * x_hat))
                mean_g_xhat = g_xhat.mean(axis=axis, keepdims=True)
                mean_g_xhat_xhat = (g_xhat * x_hat).mean(axis=axis, keepdims=True)
                contrib = (g_xhat - mean_g_xhat - x_hat * mean_g_xhat_xhat) / std
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def __getitem__(self, idx) -> "Tensor":
        out = Tensor(self.data[idx], requires_grad=self.requires_grad,
                     _children=(self,), _op="getitem")

        def _back():
            if self.requires_grad:
                g = out.grad if out.grad is not None else np.ones(out.data.shape)
                contrib = np.zeros_like(self.data)
                np.add.at(contrib, idx, g)
                self.grad = self.grad + contrib if self.grad is not None else contrib

        out._backward = _back
        return out

    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, grad: np.ndarray = None):
        if not self.requires_grad:
            return
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() called on non-scalar without grad argument")
            grad = np.ones_like(self.data)

        self.grad = self.grad + grad if self.grad is not None else grad.copy()

        # Topological sort
        topo: List["Tensor"] = []
        visited: Set[int] = set()

        def build_topo(t: "Tensor"):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)
        for t in reversed(topo):
            t._backward()

        # Free the computation graph so intermediate tensors are GC-eligible.
        for t in topo:
            t._prev = set()
            t._backward = lambda: None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def detach(self) -> "Tensor":
        t = Tensor(self.data.copy(), requires_grad=False)
        return t

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def item(self):
        return self.data.item()

    def zero_grad(self):
        self.grad = None

    def __repr__(self) -> str:
        return f"Tensor({self.data}, grad_fn={self._op!r})" if self._op else f"Tensor({self.data})"

    def __len__(self) -> int:
        return len(self.data)

    def __float__(self) -> float:
        return float(self.data)

    def __int__(self) -> int:
        return int(self.data)


class Parameter(Tensor):
    """A Tensor that is always a leaf requiring gradient."""

    def __init__(self, data, name: str = ""):
        if isinstance(data, np.ndarray):
            d = data.astype(np.float32)
        else:
            d = np.asarray(data, dtype=np.float32)
        super().__init__(d, requires_grad=True)
        self.name = name

    def __repr__(self) -> str:
        return f"Parameter({self.data}, name={self.name!r})"


# ------------------------------------------------------------------
# Factory functions
# ------------------------------------------------------------------

def zeros(shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def randn(shape, requires_grad: bool = False, scale: float = 1.0) -> Tensor:
    return Tensor(np.random.randn(*shape).astype(np.float32) * scale, requires_grad=requires_grad)


def rand(shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad)


def cat(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Concatenate tensors along axis with correct backward."""
    data_list = [t.data for t in tensors]
    out_data = np.concatenate(data_list, axis=axis)
    rg = any(t.requires_grad for t in tensors)
    out = Tensor(out_data, requires_grad=rg, _children=tuple(tensors), _op="cat")

    # Precompute split indices
    split_sizes = [t.data.shape[axis] for t in tensors]
    split_indices = np.cumsum(split_sizes[:-1])

    def _back():
        g = out.grad if out.grad is not None else np.ones(out_data.shape)
        g_splits = np.split(g, split_indices, axis=axis)
        for t, gs in zip(tensors, g_splits):
            if t.requires_grad:
                t.grad = t.grad + gs if t.grad is not None else gs.copy()

    out._backward = _back
    return out


def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Stack tensors along a new axis with correct backward."""
    data_list = [t.data for t in tensors]
    out_data = np.stack(data_list, axis=axis)
    rg = any(t.requires_grad for t in tensors)
    out = Tensor(out_data, requires_grad=rg, _children=tuple(tensors), _op="stack")

    def _back():
        g = out.grad if out.grad is not None else np.ones(out_data.shape)
        for i, t in enumerate(tensors):
            if t.requires_grad:
                idx = [slice(None)] * g.ndim
                idx[axis] = i
                gs = g[tuple(idx)]
                t.grad = t.grad + gs if t.grad is not None else gs.copy()

    out._backward = _back
    return out
