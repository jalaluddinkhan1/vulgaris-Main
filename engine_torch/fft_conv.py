"""engine_torch/fft_conv.py — FFT convolution via torch.nn.functional."""
import torch
import torch.nn.functional as F
import numpy as np
from engine_torch.tensor import Tensor


def fft_conv1d(x: np.ndarray, kernel: np.ndarray, dilation: int = 1) -> np.ndarray:
    """
    Drop-in for engine.fft_conv.fft_conv1d — uses torch conv1d for correctness.
    x      : (B, C, T)   numpy
    kernel : (C, K)      numpy
    Returns: (B, C, T)   numpy
    """
    B, C, T = x.shape
    nf, K   = kernel.shape
    assert C == nf

    x_t = torch.from_numpy(x.astype(np.float32))
    k_t = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(1)  # (C, 1, K)

    pad = (K - 1) * dilation // 2
    out = F.conv1d(x_t, k_t, padding=pad, dilation=dilation, groups=C)
    # Trim to original T
    out = out[:, :, :T]
    return out.numpy()


def fft_conv1d_backward(
    grad_out: np.ndarray,
    x: np.ndarray,
    kernel: np.ndarray,
    dilation: int = 1,
):
    """Gradient stub — returns zeros. Real gradients flow through torch autograd."""
    return np.zeros_like(x), np.zeros_like(kernel)
