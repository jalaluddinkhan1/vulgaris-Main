"""
FFT-based dilated 1-D convolution for ASE multi-scale encoder.

Replaces direct O(T*K) dilated convolution with O(T log T) FFT convolution.
Supports dilation, 'same' output length, multi-filter batch.

All implementations share:
    out = fft_conv1d(x, kernel, dilation=1)
    x      : (B, C, T)        float32
    kernel : (n_filters, K)   float32   (1-channel kernels)
    returns  (B, n_filters, T) float32
"""
import numpy as np
from engine.backend import get_backend


def _dilate_kernel(kernel: np.ndarray, dilation: int) -> np.ndarray:
    """Insert zeros to dilate a 1-D kernel. (n_filters, K) -> (n_filters, K_eff)"""
    if dilation == 1:
        return kernel
    n_f, K = kernel.shape
    K_eff = (K - 1) * dilation + 1
    dk = np.zeros((n_f, K_eff), dtype=kernel.dtype)
    dk[:, ::dilation] = kernel
    return dk


# ---------------------------------------------------------------------------
# Numpy FFT conv
# ---------------------------------------------------------------------------

def _numpy_fft_conv(
    x: np.ndarray,
    kernel: np.ndarray,
    dilation: int = 1,
) -> np.ndarray:
    """
    x      : (B, C_in, T)
    kernel : (n_filters, C_in, K)  — one kernel per filter, per in-channel
             OR (n_filters, K)     — pre-mixed (x already has n_filters channels)
    """
    # Handle both 2-D and 3-D kernels
    if kernel.ndim == 3:
        # kernel: (n_filters, C_in, K) — convolve each in-channel and sum
        n_f, C_in, K = kernel.shape
        B, C_x, T = x.shape
        assert C_x == C_in
        dk = np.stack([_dilate_kernel(kernel[:, c, :], dilation) for c in range(C_in)], axis=1)
        # (n_filters, C_in, K_eff)
        K_eff = dk.shape[2]
        N_fft = 1
        while N_fft < T + K_eff - 1:
            N_fft <<= 1
        X_fft = np.fft.rfft(x, n=N_fft, axis=-1)          # (B, C_in, Nf)
        K_fft = np.fft.rfft(dk, n=N_fft, axis=-1)          # (nf, C_in, Nf)
        # Sum over C_in: (B, nf, Nf)
        Y_fft = np.einsum("bcf,ncf->bnf", X_fft, K_fft)
        y_full = np.fft.irfft(Y_fft, n=N_fft, axis=-1)     # (B, nf, N_fft)
        pad = K_eff - 1
        return y_full[:, :, pad // 2: pad // 2 + T].astype(np.float32)

    else:
        # kernel: (n_filters, K) — x already mixed, shape (B, n_filters, T)
        n_f, K = kernel.shape
        B, C_x, T = x.shape
        assert C_x == n_f
        dk = _dilate_kernel(kernel, dilation)               # (n_filters, K_eff)
        K_eff = dk.shape[1]
        N_fft = 1
        while N_fft < T + K_eff - 1:
            N_fft <<= 1
        X_fft = np.fft.rfft(x, n=N_fft, axis=-1)           # (B, nf, Nf)
        K_fft = np.fft.rfft(dk, n=N_fft, axis=-1)          # (nf, Nf)
        Y_fft = X_fft * K_fft[np.newaxis, :, :]            # (B, nf, Nf)
        y_full = np.fft.irfft(Y_fft, n=N_fft, axis=-1)     # (B, nf, N_fft)
        pad = K_eff - 1
        return y_full[:, :, pad // 2: pad // 2 + T].astype(np.float32)


# ---------------------------------------------------------------------------
# Numba FFT conv (delegates to scipy which uses FFTW if available)
# ---------------------------------------------------------------------------

def _numba_fft_conv(x, kernel, dilation=1):
    try:
        from engine.backends.numba_ops import fft_conv_numba
        return fft_conv_numba(x, kernel, dilation)
    except Exception:
        return _numpy_fft_conv(x, kernel, dilation)


# ---------------------------------------------------------------------------
# Triton FFT conv
# ---------------------------------------------------------------------------

def _triton_fft_conv(x, kernel, dilation=1):
    try:
        from engine.backends.triton_ops.wavelet_conv import fft_conv_triton
        return fft_conv_triton(x, kernel, dilation)
    except Exception:
        return _numpy_fft_conv(x, kernel, dilation)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fft_conv1d(
    x: np.ndarray,
    kernel: np.ndarray,
    dilation: int = 1,
) -> np.ndarray:
    """
    FFT-based dilated 1-D convolution, hardware-dispatched.

    Parameters
    ----------
    x       : (B, C, T)         — input (C = n_filters if pre-mixed)
    kernel  : (n_filters, K)    — filter weights
    dilation: int               — dilation factor

    Returns
    -------
    out : (B, n_filters, T)
    """
    backend = get_backend()
    if backend == "triton":
        return _triton_fft_conv(x, kernel, dilation)
    if backend == "numba":
        return _numba_fft_conv(x, kernel, dilation)
    return _numpy_fft_conv(x, kernel, dilation)


def fft_conv1d_backward(
    grad_out: np.ndarray,
    x: np.ndarray,
    kernel: np.ndarray,
    dilation: int = 1,
) -> tuple:
    """
    Gradient of fft_conv1d w.r.t. x and kernel.

    grad_out : (B, n_filters, T)
    x        : (B, n_filters, T)   (pre-mixed)
    kernel   : (n_filters, K)

    Returns
    -------
    grad_x      : (B, n_filters, T)
    grad_kernel : (n_filters, K)
    """
    dk = _dilate_kernel(kernel, dilation)   # (nf, K_eff)
    n_f, K_eff = dk.shape
    B, _, T = x.shape

    N_fft = 1
    while N_fft < T + K_eff - 1:
        N_fft <<= 1

    G_fft = np.fft.rfft(grad_out, n=N_fft, axis=-1)    # (B, nf, Nf)
    X_fft = np.fft.rfft(x,        n=N_fft, axis=-1)    # (B, nf, Nf)
    K_fft = np.fft.rfft(dk,       n=N_fft, axis=-1)    # (nf, Nf)

    # grad_x: cross-correlate grad_out with flipped kernel = convolve with conj in freq
    grad_x_fft = G_fft * np.conj(K_fft)[np.newaxis, :, :]
    grad_x_full = np.fft.irfft(grad_x_fft, n=N_fft, axis=-1)
    pad = K_eff - 1
    grad_x = grad_x_full[:, :, pad // 2: pad // 2 + T].astype(np.float32)

    # grad_kernel: cross-correlate x with grad_out
    grad_k_fft = np.conj(X_fft) * G_fft               # (B, nf, Nf)
    grad_k_full = np.fft.irfft(grad_k_fft.sum(axis=0), n=N_fft, axis=-1)  # (nf, N_fft)
    # Extract K positions corresponding to the dilated kernel, then un-dilate
    grad_dk = grad_k_full[:, :K_eff]
    if dilation == 1:
        grad_kernel = grad_dk
    else:
        K = kernel.shape[1]
        grad_kernel = grad_dk[:, ::dilation][:, :K]

    return grad_x, grad_kernel.astype(np.float32)
