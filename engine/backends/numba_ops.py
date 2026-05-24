"""
Numba JIT-compiled hot-path ops for VULGARIS.
Falls back silently if numba is not installed.
"""
import numpy as np

try:
    import numba
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False


if _NUMBA:
    @njit(parallel=True, cache=True, fastmath=True)
    def _ssm_scan_core(fa, fb):
        """
        In-place Hillis-Steele parallel prefix scan.
        fa, fb : (B, T, D) float32 — modified in place
        """
        B, T, D = fa.shape
        stride = 1
        while stride < T:
            fa_prev = fa.copy()
            fb_prev = fb.copy()
            for b in prange(B):
                for t in range(stride, T):
                    s = t - stride
                    for d in range(D):
                        new_fa = fa_prev[b, t, d] * fa_prev[b, s, d]
                        new_fb = fa_prev[b, t, d] * fb_prev[b, s, d] + fb_prev[b, t, d]
                        fa[b, t, d] = new_fa
                        fb[b, t, d] = new_fb
            stride <<= 1
        return fb

    def ssm_scan_numba(a, b, h_init=None):
        B, T, D = a.shape
        fa = a.astype(np.float32, copy=True)
        fb = b.astype(np.float32, copy=True)
        if h_init is not None:
            fb[:, 0, :] += fa[:, 0, :] * h_init.astype(np.float32)
        return _ssm_scan_core(fa, fb)

    @njit(parallel=True, cache=True, fastmath=True)
    def _fft_conv_numba_core(x, dk, T, pad_left):
        """x: (B,nf,T), dk: (nf,K_eff) — returns (B,nf,T)"""
        from numpy.fft import rfft, irfft
        B, nf, _ = x.shape
        N_fft = x.shape[2]
        out = np.zeros((B, nf, T), dtype=np.float32)
        K_fft = np.fft.rfft(dk, n=N_fft, axis=-1)
        for b in prange(B):
            X_fft = np.fft.rfft(x[b], n=N_fft, axis=-1)
            Y_fft = X_fft * K_fft
            y_full = np.fft.irfft(Y_fft, n=N_fft, axis=-1)
            out[b] = y_full[:, pad_left: pad_left + T]
        return out

    def fft_conv_numba(x, kernel, dilation=1):
        from engine.fft_conv import _dilate_kernel
        dk = _dilate_kernel(kernel, dilation)
        K_eff = dk.shape[-1]
        B, nf, T = x.shape
        N_fft = 1
        while N_fft < T + K_eff - 1:
            N_fft <<= 1
        x_pad = np.zeros((B, nf, N_fft), dtype=np.float32)
        x_pad[:, :, :T] = x
        return _fft_conv_numba_core(x_pad, dk, T, (K_eff - 1) // 2)

else:
    def ssm_scan_numba(a, b, h_init=None):
        from engine.parallel_scan import _numpy_scan
        return _numpy_scan(a, b, h_init)

    def fft_conv_numba(x, kernel, dilation=1):
        from engine.fft_conv import _numpy_fft_conv
        return _numpy_fft_conv(x, kernel, dilation)
