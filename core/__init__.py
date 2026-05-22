import os
import ctypes

_dir = os.path.dirname(__file__)


def _try_load(name):
    path = os.path.join(_dir, 'kernels', f'{name}.so')
    if os.path.exists(path):
        try:
            return ctypes.CDLL(path)
        except Exception:
            return None
    return None


_ssm_lib = _try_load('ssm_scan')
_wavelet_lib = _try_load('wavelet_conv')

CUDA_AVAILABLE = _ssm_lib is not None


def ssm_scan_cuda(a, b):
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not compiled. Run cmake in core/.")
    import numpy as np
    assert a.shape == b.shape and a.dtype == np.float32
    B, T, N = a.shape
    h = np.empty_like(a)
    _ssm_lib.ssm_scan_cuda(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B), ctypes.c_int(T), ctypes.c_int(N)
    )
    return h


def wavelet_conv_cuda(x, filters, dilations):
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not compiled.")
    import numpy as np
    B, C, T = x.shape
    K, FL = filters.shape
    S = len(dilations)
    out = np.empty((B, K * S, T), dtype=np.float32)
    _wavelet_lib.wavelet_conv_cuda(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        filters.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B), ctypes.c_int(C), ctypes.c_int(T),
        ctypes.c_int(K), ctypes.c_int(S), ctypes.c_int(FL),
        dilations.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    return out
