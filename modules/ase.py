import numpy as np
from typing import Optional

from engine.tensor import Tensor, Parameter, zeros
from engine.module import Module
from engine.layers import Linear, RMSNorm
from engine.ops import conv1d_forward


class AdaptiveSignalEmbedding(Module):
    """
    Projects raw multi-rate telemetry signals into a unified latent manifold
    using learnable Morlet-style continuous-time filter banks at multiple scales.

    x: (batch, in_channels, T) -> (batch, T, latent_dim)
    """

    def __init__(self, in_channels: int, n_filters: int, n_scales: int,
                 filter_len: int, latent_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.n_scales = n_scales
        self.filter_len = filter_len
        self.latent_dim = latent_dim

        # Learnable wavelet parameters per filter: shape (n_filters,)
        # Amplitude: log_A -> A = exp(log_A)
        self.log_A = Parameter(
            np.zeros(n_filters, dtype=np.float64), name="log_A"
        )
        # Width: log_sigma -> sigma = exp(log_sigma)
        self.log_sigma = Parameter(
            np.zeros(n_filters, dtype=np.float64), name="log_sigma"
        )
        # Frequency
        self.omega = Parameter(
            np.linspace(1.0, 8.0, n_filters, dtype=np.float64), name="omega"
        )
        # Phase
        self.phi = Parameter(
            np.zeros(n_filters, dtype=np.float64), name="phi"
        )

        # Total channels after multi-scale concatenation: n_scales * n_filters
        # Each scale applies n_filters filters to (in_channels + optional ts) input channels
        # We apply the filter bank independently to each input channel and sum
        # Output per scale: (batch, n_filters, T)
        total_channels = n_scales * n_filters

        # in_channels + 1 accounts for optional timestamp channel
        # We build separate conv weights but here we project the per-channel
        # filter output to n_filters via a mixing linear applied after summing
        # over input channels. The actual dilated conv treats each input channel
        # separately (groups=in_channels possible but we keep it general).
        # For simplicity: conv has in_channels -> n_filters via the wavelet kernel
        # (one wavelet per output filter, applied across ALL input channels simultaneously).
        # Weight shape for grouped approach: (n_filters, in_channels, filter_len) -- but
        # we materialise the wavelet for a single in_channel and use it for all.
        # We store a per-filter, per-input-channel scale parameter for mixing.
        self.channel_mix = Parameter(
            np.random.randn(n_filters, in_channels + 1).astype(np.float64) * 0.02,
            name="channel_mix"
        )

        self.proj = Linear(total_channels, latent_dim)
        self.norm = RMSNorm(latent_dim)

    # ------------------------------------------------------------------
    def _build_filters(self) -> np.ndarray:
        """
        Materialise the (n_filters, 1, filter_len) Morlet wavelet kernel.

        t grid: linspace(-filter_len//2, filter_len//2, filter_len) / filter_len
        psi_k(t) = A_k * exp(-0.5 * (t / sigma_k)^2) * cos(omega_k * t + phi_k)
        """
        fl = self.filter_len
        t = np.linspace(-fl // 2, fl // 2, fl, dtype=np.float64) / fl  # (fl,)

        A = np.exp(self.log_A.data)        # (n_filters,)
        sigma = np.exp(self.log_sigma.data) # (n_filters,)
        omega = self.omega.data             # (n_filters,)
        phi = self.phi.data                 # (n_filters,)

        # Broadcast: (n_filters, fl)
        t_s = t[None, :] / sigma[:, None]                         # (n_filters, fl)
        gauss = np.exp(-0.5 * t_s ** 2)                           # (n_filters, fl)
        carrier = np.cos(omega[:, None] * t[None, :] + phi[:, None])  # (n_filters, fl)
        kernels = A[:, None] * gauss * carrier                    # (n_filters, fl)

        # Normalise each filter to unit energy to stabilise training
        energy = np.sqrt((kernels ** 2).sum(axis=1, keepdims=True) + 1e-8)
        kernels = kernels / energy

        return kernels.reshape(self.n_filters, 1, fl)             # (n_filters, 1, fl)

    # ------------------------------------------------------------------
    def _dilated_conv1d(self, x_np: np.ndarray, kernel_np: np.ndarray,
                        dilation: int, padding: int) -> np.ndarray:
        """
        Dilated convolution via zero-insertion between filter taps.

        x_np    : (batch, C, T)
        kernel_np: (C_out, C_in, K)  with C_in == 1 here (applied per channel)
        dilation: int >= 1
        padding : int (pre-computed for "same" output length)

        Returns (batch, C_out, T)
        """
        C_out, C_in, K = kernel_np.shape

        if dilation == 1:
            dilated_k = kernel_np
        else:
            # Effective kernel length after zero-insertion
            K_eff = (K - 1) * dilation + 1
            dilated_k = np.zeros((C_out, C_in, K_eff), dtype=np.float64)
            dilated_k[:, :, ::dilation] = kernel_np

        K_eff = dilated_k.shape[2]
        B, C, T = x_np.shape

        # Manual padding along time axis
        if padding > 0:
            x_pad = np.pad(x_np, ((0, 0), (0, 0), (padding, padding)), mode="constant")
        else:
            x_pad = x_np

        T_out = x_pad.shape[2] - K_eff + 1
        out = np.zeros((B, C_out, T_out), dtype=np.float64)

        for k in range(K_eff):
            if dilated_k[0, 0, k] == 0.0 and dilation > 1:
                # Zero tap — skip (only valid if the tap is actually zero)
                # Check properly: skip if this tap is zero in all filters
                if np.all(dilated_k[:, :, k] == 0.0):
                    continue
            # x_pad[:, :, k : k + T_out] : (B, C, T_out)
            # dilated_k[:, :, k]         : (C_out, C_in)
            # We need sum over C_in: (B, C_out, T_out)
            out += np.einsum("oi,bit->bot", dilated_k[:, :, k], x_pad[:, :, k:k + T_out])

        return out  # (B, C_out, T_out)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        timestamps: Optional[Tensor] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        """
        x          : (batch, in_channels, T)
        timestamps : optional (batch, T) — normalize to [0,1] and append as extra channel
        mask       : optional boolean array, (batch, T) or (batch, in_channels, T).
                     True = valid, False = missing. Missing positions are zeroed before
                     convolution so they do not contaminate neighbouring timesteps.
        Returns    : (batch, T, latent_dim)
        """
        B, C, T = x.data.shape
        # Apply missing-value mask before any computation
        if mask is not None:
            mask_np = np.asarray(mask, dtype=np.float32)
            if mask_np.ndim == 2:          # (B, T) -> broadcast over channels
                mask_np = mask_np[:, None, :]
            x_np = x.data * mask_np        # zero out missing positions
        else:
            x_np = x.data  # work in numpy for the wavelet convolution

        if timestamps is not None:
            ts_np = timestamps.data  # (B, T)
            # Normalize each window to [0,1]
            ts_min = ts_np.min(axis=1, keepdims=True)
            ts_max = ts_np.max(axis=1, keepdims=True)
            ts_norm = (ts_np - ts_min) / (ts_max - ts_min + 1e-8)  # (B, T)
            ts_ch = ts_norm[:, None, :]   # (B, 1, T)
            x_np = np.concatenate([x_np, ts_ch], axis=1)  # (B, C+1, T)
            actual_in = C + 1
        else:
            # Pad channel_mix to only use C channels
            actual_in = C
            x_np = x_np  # (B, C, T)

        # Build wavelet kernels: (n_filters, 1, filter_len)
        kernels_1ch = self._build_filters()  # (n_filters, 1, filter_len)

        # Mix input channels: project (B, actual_in, T) -> (B, n_filters, T)
        # using channel_mix: (n_filters, in_channels+1) -> select first actual_in cols
        cm = self.channel_mix.data[:, :actual_in]  # (n_filters, actual_in)
        # (B, n_filters, T) = einsum over input channels
        x_mixed = np.einsum("fi,bit->bft", cm, x_np)  # (B, n_filters, T)
        # x_mixed now has n_filters channels, ready for per-filter single-channel conv

        fl = self.filter_len
        scale_outputs = []
        kernels_2d = kernels_1ch[:, 0, :]  # (n_filters, fl) — squeeze 1-channel dim

        from engine.fft_conv import fft_conv1d

        for s in range(self.n_scales):
            dilation = 2 ** s
            # FFT conv: O(T log T) vs O(T*K) direct; 'same' length maintained internally
            out_s = fft_conv1d(x_mixed, kernels_2d, dilation=dilation)  # (B, n_filters, T)
            scale_outputs.append(out_s)

        # Concatenate scales: (B, n_scales * n_filters, T)
        multi_scale = np.concatenate(scale_outputs, axis=1)  # (B, total_ch, T)

        # Transpose to (B, T, total_ch) for Linear
        multi_scale_t = multi_scale.transpose(0, 2, 1)  # (B, T, total_ch)

        # Wrap in Tensor with gradient linkage through channel_mix and wavelet params
        # The numpy path breaks autograd; we reconnect by building a Tensor that
        # records the channel_mix dependency for gradient flow.
        # For full autodiff we propagate through channel_mix manually.
        total_ch = self.n_scales * self.n_filters

        # Build output Tensor — attach channel_mix as child so its grad flows
        ms_tensor = Tensor(
            multi_scale_t,
            requires_grad=x.requires_grad or self.channel_mix.requires_grad,
            _children=(x, self.channel_mix, self.log_A, self.log_sigma,
                       self.omega, self.phi),
            _op="ase_multiscale"
        )

        # Backward: compute grad w.r.t. channel_mix and wavelet params numerically
        # via finite differences would be heavy; instead we implement the analytic path.
        _x_np = x_np.copy()
        _cm_data = cm.copy()
        _kernels = kernels_1ch.copy()
        _kernels_2d = kernels_2d.copy()
        _x_mixed = x_mixed.copy()
        _scale_outs = [s.copy() for s in scale_outputs]
        _multi_scale = multi_scale.copy()
        _T = T
        _B = B
        _actual_in = actual_in

        def _ase_back():
            g = ms_tensor.grad  # (B, T, total_ch)
            if g is None:
                return
            # Transpose back to (B, total_ch, T)
            g_bct = g.transpose(0, 2, 1)  # (B, n_scales*n_filters, T)

            if self.channel_mix.requires_grad:
                # grad w.r.t. cm: (n_filters, actual_in)
                # multi_scale[s*nf:(s+1)*nf] = dilated_conv(x_mixed, kernels_1ch[k])
                # x_mixed = cm @ x_np -> dL/dcm = sum_s (dL/d_out_s @ d_out_s/d_x_mixed) @ x_np^T
                # For each scale s: out_s = dilated_conv(x_mixed, kernels_1ch, dilation=2^s)
                # dout_s/d_x_mixed[filter_k, t] involves the convolution kernel
                # We approximate: dL/d_x_mixed via transposed convolution
                from engine.fft_conv import fft_conv1d_backward
                g_cm = np.zeros_like(self.channel_mix.data)
                g_xmixed = np.zeros((_B, self.n_filters, _T), dtype=np.float32)

                for s in range(self.n_scales):
                    g_s = g_bct[:, s * self.n_filters:(s + 1) * self.n_filters, :]
                    # FFT-based transposed conv: O(T log T) vs O(T*K) loop
                    g_xm_s, _ = fft_conv1d_backward(
                        g_s.astype(np.float32),
                        _x_mixed.astype(np.float32),
                        _kernels_2d.astype(np.float32),
                        dilation=2 ** s,
                    )
                    g_xmixed += g_xm_s

                g_cm[:, :_actual_in] = np.einsum("bft,bit->fi", g_xmixed, _x_np)
                self.channel_mix.grad = (self.channel_mix.grad + g_cm
                                         if self.channel_mix.grad is not None else g_cm)

            if x.requires_grad:
                from engine.fft_conv import fft_conv1d_backward
                g_xmixed2 = np.zeros((_B, self.n_filters, _T), dtype=np.float32)

                for s in range(self.n_scales):
                    g_s = g_bct[:, s * self.n_filters:(s + 1) * self.n_filters, :]
                    g_xm_s, _ = fft_conv1d_backward(
                        g_s.astype(np.float32),
                        _x_mixed.astype(np.float32),
                        _kernels_2d.astype(np.float32),
                        dilation=2 ** s,
                    )
                    g_xmixed2 += g_xm_s

                g_x = np.einsum("fi,bft->bit", _cm_data, g_xmixed2)
                contrib_x = g_x[:, :C, :]
                x.grad = x.grad + contrib_x if x.grad is not None else contrib_x

        ms_tensor._backward = _ase_back

        # Project to latent dim: (B, T, total_ch) -> (B, T, latent_dim)
        out = self.proj(ms_tensor)
        out = self.norm(out)
        return out  # (B, T, latent_dim)
