from __future__ import annotations

import numpy as np

from engine.tensor import Tensor, Parameter, zeros
from engine.module import Module
from engine.layers import Linear, RMSNorm
from engine.ops import conv1d_forward


class AdaptiveSignalEmbedding(Module):
    """
    Projects raw multi-rate telemetry signals into a unified latent manifold
    using learnable Morlet-style continuous-time filter banks at multiple scales.

    x: (batch, in_channels, T) -> (batch, T, latent_dim)

    Improvements over baseline:
    - Filter cache: kernels recomputed only when wavelet params change.
    - Linear interpolation: missing positions interpolated instead of
      zero-padded, eliminating the frequency bias in FFT convolution.
    """

    def __init__(self, in_channels: int, n_filters: int, n_scales: int,
                 filter_len: int, latent_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters   = n_filters
        self.n_scales    = n_scales
        self.filter_len  = filter_len
        self.latent_dim  = latent_dim

        self.log_A     = Parameter(np.zeros(n_filters, dtype=np.float32), name="log_A")
        self.log_sigma = Parameter(np.zeros(n_filters, dtype=np.float32), name="log_sigma")
        self.omega     = Parameter(np.linspace(1.0, 8.0, n_filters, dtype=np.float32), name="omega")
        self.phi       = Parameter(np.zeros(n_filters, dtype=np.float32), name="phi")

        self.channel_mix = Parameter(
            np.random.randn(n_filters, in_channels + 1).astype(np.float32) * 0.02,
            name="channel_mix",
        )

        total_channels = n_scales * n_filters
        self.proj = Linear(total_channels, latent_dim)
        self.norm = RMSNorm(latent_dim)

        # Filter cache — invalidated when any wavelet param changes
        object.__setattr__(self, "_filter_cache", None)
        object.__setattr__(self, "_filter_fp",    None)

    # ------------------------------------------------------------------

    def _wavelet_fingerprint(self) -> tuple:
        return (
            float(self.log_A.data.sum()),
            float(self.log_sigma.data.sum()),
            float(self.omega.data.sum()),
            float(self.phi.data.sum()),
        )

    def _build_filters(self) -> np.ndarray:
        """
        Materialise the (n_filters, 1, filter_len) Morlet wavelet kernel.
        Result is cached and only recomputed when wavelet parameters change.

        psi_k(t) = A_k * exp(-0.5*(t/sigma_k)^2) * cos(omega_k*t + phi_k)
        """
        fp     = self._wavelet_fingerprint()
        cached = object.__getattribute__(self, "_filter_cache")
        if cached is not None and fp == object.__getattribute__(self, "_filter_fp"):
            return cached

        fl    = self.filter_len
        t     = np.linspace(-fl // 2, fl // 2, fl, dtype=np.float32) / fl

        A     = np.exp(self.log_A.data)
        sigma = np.exp(self.log_sigma.data)
        omega = self.omega.data
        phi   = self.phi.data

        t_s     = t[None, :] / sigma[:, None]
        gauss   = np.exp(-0.5 * t_s ** 2)
        carrier = np.cos(omega[:, None] * t[None, :] + phi[:, None])
        kernels = A[:, None] * gauss * carrier

        energy  = np.sqrt((kernels ** 2).sum(axis=1, keepdims=True) + 1e-8)
        kernels = (kernels / energy).reshape(self.n_filters, 1, fl)

        object.__setattr__(self, "_filter_cache", kernels)
        object.__setattr__(self, "_filter_fp",    fp)
        return kernels

    # ------------------------------------------------------------------

    @staticmethod
    def _linear_interp_masked(x_np: np.ndarray,
                              missing: np.ndarray) -> np.ndarray:
        """
        Replace missing positions with linear interpolation between nearest
        valid neighbours.  Eliminates the zero-padding frequency bias that
        distorts FFT-based wavelet convolution (Khayati et al. VLDB 2020).

        x_np    : (B, C, T)
        missing : (B, C, T)  True = missing / invalid
        Returns : (B, C, T)  with gaps filled
        """
        result = x_np.copy()
        B, C, T = x_np.shape
        for b in range(B):
            for c in range(C):
                m = missing[b, c]
                if not m.any():
                    continue
                valid = np.where(~m)[0]
                if len(valid) == 0:
                    continue
                result[b, c, np.where(m)[0]] = np.interp(
                    np.where(m)[0], valid, x_np[b, c, valid]
                )
        return result

    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        timestamps: Tensor | None = None,
        mask: np.ndarray | None = None,
    ) -> Tensor:
        """
        x          : (batch, in_channels, T)
        timestamps : optional (batch, T) — normalize to [0,1] and append as extra channel
        mask       : optional bool array (batch, T) or (batch, in_channels, T).
                     True = valid, False = missing.  Missing positions are
                     linearly interpolated before convolution.
        Returns    : (batch, T, latent_dim)
        """
        from engine.fft_conv import fft_conv1d, fft_conv1d_backward

        B, C, T = x.data.shape

        # ── Missing-value handling ────────────────────────────────────────
        if mask is not None:
            mask_np = np.asarray(mask, dtype=bool)
            if mask_np.ndim == 2:
                mask_np = np.broadcast_to(mask_np[:, None, :], (B, C, T)).copy()
            missing = ~mask_np
            x_np    = self._linear_interp_masked(x.data.copy(), missing)
        else:
            x_np = x.data

        # ── Optional timestamp channel ────────────────────────────────────
        if timestamps is not None:
            ts_np   = timestamps.data
            ts_min  = ts_np.min(axis=1, keepdims=True)
            ts_max  = ts_np.max(axis=1, keepdims=True)
            ts_norm = (ts_np - ts_min) / (ts_max - ts_min + 1e-8)
            x_np    = np.concatenate([x_np, ts_norm[:, None, :]], axis=1)
            actual_in = C + 1
        else:
            actual_in = C

        # ── Wavelet kernels (cached) ──────────────────────────────────────
        kernels_1ch = self._build_filters()           # (n_filters, 1, fl)
        kernels_2d  = kernels_1ch[:, 0, :]            # (n_filters, fl)

        cm      = self.channel_mix.data[:, :actual_in]
        x_mixed = np.einsum("fi,bit->bft", cm, x_np)

        # ── Channel-independent extraction ────────────────────────────────
        scale_outputs_per_channel = []
        for c_idx in range(actual_in):
            x_c   = x_np[:, c_idx:c_idx+1, :]
            cm_c  = cm[:, c_idx]
            x_exp = np.broadcast_to(
                x_c, (x_c.shape[0], self.n_filters, x_c.shape[2])
            ).copy() * cm_c[None, :, None]
            ch_scales = [
                fft_conv1d(x_exp, kernels_2d, dilation=2 ** s)
                for s in range(self.n_scales)
            ]
            scale_outputs_per_channel.append(ch_scales)

        scale_outputs = [
            sum(scale_outputs_per_channel[ci][s] for ci in range(actual_in))
            for s in range(self.n_scales)
        ]

        multi_scale   = np.concatenate(scale_outputs, axis=1)
        multi_scale_t = multi_scale.transpose(0, 2, 1).astype(np.float32)

        # ── Autograd-connected Tensor ─────────────────────────────────────
        total_ch  = self.n_scales * self.n_filters
        ms_tensor = Tensor(
            multi_scale_t,
            requires_grad=x.requires_grad or self.channel_mix.requires_grad,
            _children=(x, self.channel_mix, self.log_A, self.log_sigma,
                       self.omega, self.phi),
            _op="ase_multiscale",
        )

        _x_np    = x_np.copy()
        _cm_data = cm.copy()
        _k2d     = kernels_2d.copy()
        _x_mixed = x_mixed.copy()
        _T, _B, _ai, _nf = T, B, actual_in, self.n_filters

        # Pre-compute quantities needed for wavelet param gradients.
        # We capture these from _build_filters to avoid recomputing inside backward.
        _fl    = self.filter_len
        _t_grid = np.linspace(-_fl // 2, _fl // 2, _fl, dtype=np.float32) / _fl
        _A     = np.exp(self.log_A.data).copy()       # (n_filters,)
        _sigma = np.exp(self.log_sigma.data).copy()   # (n_filters,)
        _omega = self.omega.data.copy()               # (n_filters,)
        _phi   = self.phi.data.copy()                 # (n_filters,)
        # kernels (unit-energy normalised) already stored in _k2d: (n_filters, fl)

        def _ase_back():
            g = ms_tensor.grad
            if g is None:
                return
            g_bct = g.transpose(0, 2, 1)

            # Accumulate grad_kernel across all scales (needed for wavelet params)
            g_kernel_total = np.zeros((_nf, _fl), dtype=np.float32)

            if self.channel_mix.requires_grad:
                g_cm     = np.zeros_like(self.channel_mix.data)
                g_xmixed = np.zeros((_B, _nf, _T), dtype=np.float32)
                for s in range(self.n_scales):
                    g_s = g_bct[:, s * _nf:(s + 1) * _nf, :]
                    g_xm_s, g_k_s = fft_conv1d_backward(
                        g_s.astype(np.float32),
                        _x_mixed.astype(np.float32),
                        _k2d.astype(np.float32),
                        dilation=2 ** s,
                    )
                    g_xmixed += g_xm_s
                    g_kernel_total += g_k_s
                g_cm[:, :_ai] = np.einsum("bft,bit->fi", g_xmixed, _x_np)
                self.channel_mix.grad = (
                    self.channel_mix.grad + g_cm
                    if self.channel_mix.grad is not None else g_cm
                )

            if x.requires_grad:
                g_xmixed2 = np.zeros((_B, _nf, _T), dtype=np.float32)
                for s in range(self.n_scales):
                    g_s = g_bct[:, s * _nf:(s + 1) * _nf, :]
                    g_xm_s, g_k_s = fft_conv1d_backward(
                        g_s.astype(np.float32),
                        _x_mixed.astype(np.float32),
                        _k2d.astype(np.float32),
                        dilation=2 ** s,
                    )
                    g_xmixed2 += g_xm_s
                    if not self.channel_mix.requires_grad:
                        g_kernel_total += g_k_s
                g_x    = np.einsum("fi,bft->bit", _cm_data, g_xmixed2)
                contrib = g_x[:, :C, :]
                x.grad  = x.grad + contrib if x.grad is not None else contrib

            # ── Wavelet parameter gradients ─────────────────────────────────
            # Chain rule through _build_filters():
            #   kernels[f,τ] = A[f]*gauss[f,τ]*cos(ω[f]*t[τ]+φ[f]) / energy[f]
            #
            # g_kernel_total[f,τ] = d(loss)/d(kernels[f,τ])
            # We backprop through: kernels → (log_A, log_sigma, omega, phi)
            # Approximation: treat energy normalization as constant (stop-gradient
            # on energy) — standard trick for normalized filterbanks.
            if (self.log_A.requires_grad or self.log_sigma.requires_grad or
                    self.omega.requires_grad or self.phi.requires_grad):
                t  = _t_grid                              # (fl,)
                ts = t[None, :] / _sigma[:, None]         # (n_filters, fl)
                gauss   = np.exp(-0.5 * ts ** 2)          # (n_filters, fl)
                arg     = _omega[:, None] * t + _phi[:, None]
                cos_arg = np.cos(arg)                      # (n_filters, fl)
                sin_arg = np.sin(arg)                      # (n_filters, fl)

                # d(kernels)/d(log_A[f])   = kernels[f,:] (since A=exp(log_A))
                if self.log_A.requires_grad:
                    g_logA = (_k2d * g_kernel_total).sum(axis=1)  # (n_filters,)
                    self.log_A.grad = (
                        self.log_A.grad + g_logA
                        if self.log_A.grad is not None else g_logA
                    )

                # d(kernels)/d(log_sigma[f]) = kernels[f,:] * (t/sigma[f])^2
                if self.log_sigma.requires_grad:
                    g_logS = (_k2d * ts ** 2 * g_kernel_total).sum(axis=1)
                    self.log_sigma.grad = (
                        self.log_sigma.grad + g_logS
                        if self.log_sigma.grad is not None else g_logS
                    )

                # d(kernels)/d(omega[f])  ∝ -t * sin(ω*t+φ) * A*gauss/energy
                if self.omega.requires_grad:
                    d_omega = _A[:, None] * gauss * (-t * sin_arg)
                    energy  = np.sqrt((_k2d ** 2).sum(axis=1, keepdims=True) + 1e-8)
                    d_omega = d_omega / energy
                    g_omega = (d_omega * g_kernel_total).sum(axis=1)
                    self.omega.grad = (
                        self.omega.grad + g_omega
                        if self.omega.grad is not None else g_omega
                    )

                # d(kernels)/d(phi[f])    ∝ -sin(ω*t+φ) * A*gauss/energy
                if self.phi.requires_grad:
                    d_phi = _A[:, None] * gauss * (-sin_arg)
                    energy = np.sqrt((_k2d ** 2).sum(axis=1, keepdims=True) + 1e-8)
                    d_phi  = d_phi / energy
                    g_phi  = (d_phi * g_kernel_total).sum(axis=1)
                    self.phi.grad = (
                        self.phi.grad + g_phi
                        if self.phi.grad is not None else g_phi
                    )

        ms_tensor._backward = _ase_back

        out = self.proj(ms_tensor)
        out = self.norm(out)
        return out
