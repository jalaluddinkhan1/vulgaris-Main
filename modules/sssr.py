import numpy as np
from collections import OrderedDict
from typing import List, Optional, Tuple

from engine.tensor import Tensor, Parameter, zeros, ones, cat
from engine.module import Module
from engine.layers import Linear, RMSNorm, Conv1d
from config import SSSRConfig


class SSSRHead(Module):
    """Single SSM head with Hebbian online adaptation."""

    def __init__(self, d_model: int, state_dim: int,
                 dt_min: float, dt_max: float,
                 hebbian_lr: float, stability_eps: float):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.hebbian_lr = hebbian_lr
        self.stability_eps = stability_eps

        # log_A: init near log(0.5) + small noise so A ≈ 0.5
        log_a_init = np.full(state_dim, np.log(0.5), dtype=np.float64)
        log_a_init += np.random.randn(state_dim) * 0.01
        self.log_A = Parameter(log_a_init, name="log_A")

        self.B_proj = Linear(d_model, state_dim, bias=False)
        self.C_proj = Linear(d_model, state_dim, bias=False)

        # D skip-connection, one scalar per output channel (d_model)
        self.D = Parameter(np.ones(d_model, dtype=np.float64), name="D")

        # dt projection + bias
        self.dt_proj = Linear(d_model, 1, bias=True)
        # initialise dt_proj bias so softplus output lands near dt_min
        # softplus^{-1}(dt_min) = log(exp(dt_min) - 1)
        dt_init_bias = np.log(np.exp(dt_min) - 1.0 + 1e-8)
        self.dt_proj.bias.data[:] = dt_init_bias
        self.dt_bias = Parameter(np.zeros(1, dtype=np.float64), name="dt_bias")

    def _compute_dt(self, x: Tensor) -> Tensor:
        """
        x: (batch, T, d_model) or (batch, d_model)
        Returns dt: (batch, T, 1) or (batch, 1), clamped to [dt_min, dt_max]
        """
        raw = self.dt_proj(x) + self.dt_bias  # (..., 1)
        # softplus
        sp_data = np.log1p(np.exp(np.clip(raw.data, -20, 20)))
        sp = Tensor(sp_data, requires_grad=raw.requires_grad,
                    _children=(raw,), _op="softplus")

        def _sp_back():
            if raw.requires_grad and sp.grad is not None:
                sig = 1.0 / (1.0 + np.exp(-np.clip(raw.data, -500, 500)))
                raw.grad = (raw.grad + sp.grad * sig
                            if raw.grad is not None else sp.grad * sig)

        sp._backward = _sp_back
        return sp.clip(self.dt_min, self.dt_max)

    def forward(self, x: Tensor, h_prev: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        """
        x    : (batch, T, d_model)
        Returns (y, h_last):
            y      : (batch, T, state_dim)  — raw SSM output, projected back externally
            h_last : (batch, state_dim)
        """
        B, T, D = x.shape

        if h_prev is None:
            h_prev = zeros((B, self.state_dim))

        dt = self._compute_dt(x)       # (B, T, 1)
        B_t = self.B_proj(x)           # (B, T, state_dim)
        C_t = self.C_proj(x)           # (B, T, state_dim)

        # A_t = exp(-exp(log_A) * dt)  — ZOH discretisation
        # exp(log_A): (state_dim,)  ->  (1, 1, state_dim)
        decay = self.log_A.exp().reshape(1, 1, self.state_dim)  # always positive
        A_t = (decay * dt * (-1.0)).exp()                        # (B, T, state_dim)

        # Sequential scan storing all h_t for Hebbian update
        h_t = h_prev
        ys = []
        h_all = []  # list of (B, state_dim) tensors, T entries

        for t in range(T):
            a_t = A_t[:, t, :]   # (B, state_dim)
            b_t = B_t[:, t, :]   # (B, state_dim)
            c_t = C_t[:, t, :]   # (B, state_dim)

            h_t = a_t * h_t + b_t  # h_t = A_t ⊙ h_{t-1} + B_t ⊙ x_t
            h_all.append(h_t)

            # y_t = C_t · h_t + D ⊙ x_t  (dot then add skip)
            y_t = (c_t * h_t).sum(axis=-1, keepdims=True)  # (B, 1)
            # Add D ⊙ x_t contribution: D is per d_model but y_t is scalar per head
            # We sum D⊙x over d_model to produce scalar skip
            x_t_raw = x[:, t, :]                  # (B, d_model)
            skip = (self.D * x_t_raw).sum(axis=-1, keepdims=True)  # (B, 1)
            y_t = y_t + skip                       # (B, 1)
            ys.append(y_t.unsqueeze(1))            # (B, 1, 1)

        y = cat(ys, axis=1)                        # (B, T, 1)
        h_last = h_t                               # (B, state_dim)

        # Hebbian update during training (numpy in-place, not autograd)
        if self.training:
            h_np = np.stack([hh.data for hh in h_all], axis=1)  # (B, T, state_dim)
            self.hebbian_update(h_np)

        return y, h_last

    def step(self, x_t: Tensor, h_prev: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Single-step inference.
        x_t   : (batch, d_model)
        h_prev: (batch, state_dim)
        Returns (y_t, h_new):
            y_t  : (batch, 1)
            h_new: (batch, state_dim)
        """
        x_t_3d = x_t.unsqueeze(1)                # (batch, 1, d_model)
        dt = self._compute_dt(x_t_3d)             # (batch, 1, 1)
        dt_2d = dt.squeeze(1)                     # (batch, 1)

        B_t = self.B_proj(x_t)                    # (batch, state_dim)
        C_t = self.C_proj(x_t)                    # (batch, state_dim)

        decay = self.log_A.exp().reshape(1, self.state_dim)
        A_t = (decay * dt_2d * (-1.0)).exp()      # (batch, state_dim)

        h_new = A_t * h_prev + B_t
        y_t = (C_t * h_new).sum(axis=-1, keepdims=True)  # (batch, 1)
        skip = (self.D * x_t).sum(axis=-1, keepdims=True)
        y_t = y_t + skip

        return y_t, h_new

    def hebbian_update(self, h_states: np.ndarray):
        """
        h_states: (batch, T, state_dim)
        ΔA_log = η * mean_{batch,time}(h_t ⊙ h_{t-1} - h_t^2)
        Clamp log_A to [-5, 0].
        """
        if h_states.shape[1] < 2:
            return
        h_prev_np = h_states[:, :-1, :]   # (B, T-1, N)
        h_curr_np = h_states[:, 1:, :]    # (B, T-1, N)
        delta = (h_curr_np * h_prev_np - h_curr_np ** 2).mean(axis=(0, 1))  # (N,)
        self.log_A.data += self.hebbian_lr * delta
        np.clip(self.log_A.data, -5.0, 0.0, out=self.log_A.data)


class SelectiveSSR(Module):
    """Full SSSR block: multi-head SSM with expansion, causal conv, gating, and skip."""

    def __init__(self, d_model: int, config: SSSRConfig):
        super().__init__()
        self.d_model = d_model
        self.d_inner = config.d_inner
        self.n_heads = config.n_heads
        self.state_dim = config.state_dim
        self.dt_min = config.dt_min
        self.dt_max = config.dt_max

        # Expansion branch
        self.x_proj = Linear(d_model, config.d_inner)
        # Gating branch
        self.z_proj = Linear(d_model, d_model)

        # Causal depthwise conv on expansion: kernel=4, causal padding = kernel-1 on left
        self.conv_kernel = 4
        self.depthwise_conv = Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=self.conv_kernel,
            padding=0,      # we do manual left padding
            groups=config.d_inner,
            bias=True,
        )

        # n_heads independent SSM heads; each head gets d_inner features
        # and produces 1-dim output; we project d_inner channels through heads
        # head_dim = d_inner // n_heads  (features per head)
        assert config.d_inner % config.n_heads == 0, \
            "d_inner must be divisible by n_heads"
        self.head_dim = config.d_inner // config.n_heads
        head_state = max(1, config.state_dim // config.n_heads)

        for i in range(config.n_heads):
            setattr(self, f"head_{i}",
                    SSSRHead(self.head_dim, head_state,
                             config.dt_min, config.dt_max,
                             config.hebbian_lr, config.stability_eps))

        # Project concatenated head outputs (n_heads scalar outputs per timestep)
        # Each head produces 1 scalar -> total n_heads per timestep
        self.y_proj = Linear(config.n_heads, d_model)

        # Skip connection
        self.skip_proj = Linear(d_model, d_model)

        # Output norm
        self.norm = RMSNorm(d_model)

    def _get_head(self, i: int) -> SSSRHead:
        return getattr(self, f"head_{i}")

    def _causal_conv(self, u: Tensor) -> Tensor:
        """
        u: (B, T, d_inner)
        Applies depthwise conv1d with causal (left) padding.
        Returns (B, T, d_inner).
        """
        B, T, C = u.shape
        # Conv1d expects (B, C, L); transpose
        u_t = u.transpose((0, 2, 1))    # (B, d_inner, T)

        # Manual left padding: pad kernel_size-1 zeros on the left
        pad = self.conv_kernel - 1
        pad_data = np.zeros((B, C, pad), dtype=np.float64)
        pad_t = Tensor(pad_data, requires_grad=False)

        # Concatenate [pad | u_t] along time axis
        u_padded_data = np.concatenate([pad_t.data, u_t.data], axis=2)  # (B, C, T+pad)
        u_padded = Tensor(
            u_padded_data,
            requires_grad=u_t.requires_grad,
            _children=(u_t,),
            _op="causal_pad"
        )

        def _pad_back():
            if u_t.requires_grad and u_padded.grad is not None:
                # Gradient only flows to the right (non-pad) portion
                u_t.grad = (u_t.grad + u_padded.grad[:, :, pad:]
                            if u_t.grad is not None
                            else u_padded.grad[:, :, pad:].copy())

        u_padded._backward = _pad_back

        # Apply conv (no padding inside Conv1d) -> output length = T
        conv_out = self.depthwise_conv(u_padded)   # (B, d_inner, T)

        # Transpose back: (B, T, d_inner)
        out = conv_out.transpose((0, 2, 1))
        return out

    def forward(self, x: Tensor, h_states: Optional[List] = None
                ) -> Tuple[Tensor, List]:
        """
        x: (batch, T, d_model)
        Returns (output, h_states_new) where output: (batch, T, d_model)
        """
        B, T, D = x.shape

        if h_states is None:
            h_states = [None] * self.n_heads

        # Expansion and causal conv
        u = self.x_proj(x)               # (B, T, d_inner)
        u = self._causal_conv(u)          # (B, T, d_inner)

        # Gating branch
        z = self.z_proj(x)               # (B, T, d_model)

        # Multi-head SSM: split u into n_heads chunks along feature dim
        head_outs = []
        h_states_new = []

        for i in range(self.n_heads):
            head = self._get_head(i)
            # Slice the i-th chunk: (B, T, head_dim)
            start = i * self.head_dim
            end = start + self.head_dim
            u_i_data = u.data[:, :, start:end]
            u_i = Tensor(u_i_data, requires_grad=u.requires_grad,
                         _children=(u,), _op=f"head_slice_{i}")

            _start, _end, _u, _u_i = start, end, u, u_i

            def _slice_back(ui=_u_i, parent=_u, s=_start, e=_end):
                if parent.requires_grad and ui.grad is not None:
                    if parent.grad is None:
                        parent.grad = np.zeros_like(parent.data)
                    parent.grad[:, :, s:e] += ui.grad

            u_i._backward = _slice_back

            y_i, h_new_i = head(u_i, h_states[i])   # y_i: (B, T, 1)
            head_outs.append(y_i)
            h_states_new.append(h_new_i)

        # Concatenate head outputs: (B, T, n_heads)
        y_heads = cat(head_outs, axis=-1)   # (B, T, n_heads)

        # Project to d_model
        y_ssm = self.y_proj(y_heads)        # (B, T, d_model)

        # Gated output: y_ssm * silu(z)
        gated = y_ssm * z.silu()            # (B, T, d_model)

        # Skip connection
        skip = self.skip_proj(x)            # (B, T, d_model)

        out = self.norm(gated + skip)
        return out, h_states_new

    def step(self, x_t: Tensor, h_states: List) -> Tuple[Tensor, List]:
        """
        Single-step streaming inference.
        x_t    : (batch, d_model)
        h_states: list of h_prev per head, each (batch, head_state_dim)
        Returns (output_t, h_states_new)  output_t: (batch, d_model)
        """
        B, D = x_t.shape

        # Expand
        u_t = self.x_proj(x_t)            # (B, d_inner)

        # For streaming conv: we just apply with single sample (no history buffer here)
        # Treat as T=1 with causal padding
        u_t_3d = u_t.unsqueeze(1)                       # (B, 1, d_inner)
        u_conv = self._causal_conv(u_t_3d).squeeze(1)   # (B, d_inner)

        # Gating
        z_t = self.z_proj(x_t)            # (B, d_model)

        # Multi-head SSM step
        head_outs = []
        h_states_new = []

        for i in range(self.n_heads):
            head = self._get_head(i)
            start = i * self.head_dim
            end = start + self.head_dim
            u_i_data = u_conv.data[:, start:end]
            u_i = Tensor(u_i_data, requires_grad=u_conv.requires_grad,
                         _children=(u_conv,), _op=f"step_slice_{i}")

            _u_conv, _u_i, _s, _e = u_conv, u_i, start, end

            def _back(ui=_u_i, p=_u_conv, s=_s, e=_e):
                if p.requires_grad and ui.grad is not None:
                    if p.grad is None:
                        p.grad = np.zeros_like(p.data)
                    p.grad[:, s:e] += ui.grad

            u_i._backward = _back

            y_i, h_new_i = head.step(u_i, h_states[i])   # y_i: (B, 1)
            head_outs.append(y_i)
            h_states_new.append(h_new_i)

        # (B, n_heads)
        y_heads = cat(head_outs, axis=-1)

        # Project
        y_ssm = self.y_proj(y_heads)       # (B, d_model)

        gated = y_ssm * z_t.silu()
        skip = self.skip_proj(x_t)
        out = self.norm(gated + skip)

        return out, h_states_new
