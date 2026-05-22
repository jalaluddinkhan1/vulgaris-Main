import numpy as np
from typing import List, Optional, Tuple

from engine.tensor import Tensor, Parameter, zeros
from engine.module import Module
from engine.layers import Linear
from config import HTDConfig


class HTDLevel(Module):
    """
    Single SSM level with ZOH discretisation.
    dt is constrained to be near the level's time constant tau.

    State: h_t = A_bar * h_{t-1} + (1 - A_bar) * B_proj(x) + bias
    Output: y_t = C_proj(h_t)
    """

    def __init__(self, d_model: int, state_dim: int, tau: float):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.tau = tau

        # Diagonal SSM decay: log_A initialised to log(0.5) -> A ~ 0.5
        self.log_A = Parameter(
            np.full(state_dim, np.log(0.5), dtype=np.float64), name="log_A"
        )
        self.B_proj = Linear(d_model, state_dim)
        self.C_proj = Linear(state_dim, d_model)
        self.dt_proj = Linear(d_model, 1)

    def forward(self, x: Tensor, h_prev: Optional[Tensor] = None,
                bias: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        x     : (batch, T, d_model) or (batch, d_model) for single step
        h_prev: (batch, state_dim) or None  [zeros if None]
        bias  : (batch, state_dim) optional slow->fast feedback, broadcast over T

        Returns (y, h_last):
            y      : same leading dims as x but last dim = d_model
            h_last : (batch, state_dim)
        """
        squeeze_T = x.ndim == 2
        if squeeze_T:
            x = x.unsqueeze(1)  # (batch, 1, d_model)

        B_sz, T, D = x.shape

        if h_prev is None:
            h_prev = zeros((B_sz, self.state_dim))

        # dt: (batch, T, 1) -> constrained around tau
        dt_raw = self.dt_proj(x)                    # (batch, T, 1)
        dt = dt_raw.sigmoid() * (2.0 * self.tau)    # dt in [0, 2*tau]

        # A_bar = exp(-exp(log_A) * dt)  — broadcast state_dim
        # exp(log_A): (state_dim,) ; dt: (batch, T, 1)
        A_decay = (self.log_A.exp()).reshape(1, 1, self.state_dim)  # (1,1,N)
        A_bar = (A_decay * dt * (-1.0)).exp()                       # (batch, T, N)

        B_in = self.B_proj(x)   # (batch, T, state_dim)

        # Recurrent scan over T — sequential for correctness
        h_t = h_prev  # (batch, state_dim)
        ys = []

        for t in range(T):
            a_t = A_bar[:, t, :]    # (batch, state_dim)
            b_t = B_in[:, t, :]     # (batch, state_dim)

            # h_t = a_t * h_{t-1} + (1 - a_t) * b_t
            h_t = a_t * h_t + (Tensor(np.ones(a_t.shape)) - a_t) * b_t

            if bias is not None:
                h_t = h_t + bias

            y_t = self.C_proj(h_t)   # (batch, d_model)
            ys.append(y_t.unsqueeze(1))  # (batch, 1, d_model)

        # Stack along time
        # Use cat instead of stack to preserve gradient
        from engine.tensor import cat as tcat
        y = tcat(ys, axis=1)  # (batch, T, d_model)
        h_last = h_t          # (batch, state_dim)

        if squeeze_T:
            y = y.squeeze(1)

        return y, h_last


class HierarchicalTimescaleDecomposition(Module):
    """
    n_levels nested SSMs at different time constants.
    Level i subsamples input by factor 2^i and communicates via bottleneck layers.

    Fast->Slow: slow input augmented by bottleneck_down(h_fast)
    Slow->Fast: fast receives additive bias from bottleneck_up(h_slow)
    """

    def __init__(self, d_model: int, state_dim: int, config: HTDConfig):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.n_levels = config.n_levels
        self.taus = config.time_constants

        # SSM level modules — stored as numbered attributes for Module registration
        for i, tau in enumerate(self.taus):
            setattr(self, f"level_{i}", HTDLevel(d_model, state_dim, tau))

        # Bottleneck projections: fast(level i) -> slow(level i+1) input
        for i in range(self.n_levels - 1):
            setattr(self, f"bottleneck_down_{i}",
                    Linear(state_dim, d_model))

        # Slow(level i+1) -> fast(level i) bias feedback
        for i in range(self.n_levels - 1):
            setattr(self, f"bottleneck_up_{i}",
                    Linear(state_dim, state_dim))

        # Output: concatenate d_model from each level, project back to d_model
        self.output_proj = Linear(d_model * self.n_levels, d_model)

    def _get_level(self, i: int) -> HTDLevel:
        return getattr(self, f"level_{i}")

    def _get_bn_down(self, i: int) -> Linear:
        return getattr(self, f"bottleneck_down_{i}")

    def _get_bn_up(self, i: int) -> Linear:
        return getattr(self, f"bottleneck_up_{i}")

    def forward(self, x: Tensor,
                states: Optional[List[Optional[Tensor]]] = None
                ) -> Tuple[Tensor, List[Tensor]]:
        """
        x     : (batch, T, d_model)
        states: list of (batch, state_dim) h_prev per level, or None

        Returns (output, new_states):
            output    : (batch, T, d_model)
            new_states: list of (batch, state_dim), one per level
        """
        B, T, D = x.shape

        if states is None:
            states = [None] * self.n_levels

        # ----------------------------------------------------------------
        # Bottom-up pass: compute each level's output
        # Level i operates on x subsampled by stride 2^i
        # Fast->Slow coupling adds bottleneck_down(h_fast) to slow input
        # ----------------------------------------------------------------
        level_inputs: List[Optional[Tensor]] = [None] * self.n_levels
        level_inputs[0] = x  # level 0 gets full-rate input

        # Pre-compute subsampled inputs (before coupling; coupling added below)
        for i in range(1, self.n_levels):
            stride = 2 ** i
            # Subsample: take every stride-th timestep
            indices = list(range(0, T, stride))
            if len(indices) == 0:
                indices = [0]
            # Build subsampled tensor
            x_sub_np = x.data[:, indices, :]           # (B, T_i, D)
            level_inputs[i] = Tensor(
                x_sub_np,
                requires_grad=x.requires_grad,
                _children=(x,),
                _op=f"subsample_{i}"
            )
            # Attach backward for subsampling
            _i = i
            _indices = indices
            _T = T
            _sub = level_inputs[i]
            _x = x

            def _sub_back(sub=_sub, inp=_x, idx=_indices, t_full=_T):
                if inp.requires_grad and sub.grad is not None:
                    contrib = np.zeros_like(inp.data)
                    np.add.at(contrib, (slice(None), idx, slice(None)), sub.grad)
                    inp.grad = inp.grad + contrib if inp.grad is not None else contrib

            level_inputs[_i]._backward = _sub_back

        # Now run levels with coupling
        # We do two passes: first collect h_last from each level independently,
        # then on a second pass inject the cross-level feedback.
        # For efficiency we do a single pass: process level 0 first, use its
        # hidden state to augment level 1, and so on (causal).

        level_outputs: List[Optional[Tensor]] = [None] * self.n_levels
        new_states: List[Optional[Tensor]] = [None] * self.n_levels
        level_h_last: List[Optional[Tensor]] = [None] * self.n_levels

        # Forward sweep: level 0 -> n_levels-1
        for i in range(self.n_levels):
            xi = level_inputs[i]   # (B, T_i, D)
            h_prev = states[i]

            # Fast->Slow: augment xi with bottleneck_down of previous (faster) level
            if i > 0 and level_h_last[i - 1] is not None:
                aug = self._get_bn_down(i - 1)(level_h_last[i - 1])  # (B, D)
                # Broadcast aug over time dimension of xi
                T_i = xi.shape[1]
                aug_expanded = aug.unsqueeze(1)  # (B, 1, D)
                aug_np = np.broadcast_to(aug_expanded.data, (B, T_i, D)).copy()
                aug_broad = Tensor(
                    aug_np,
                    requires_grad=aug.requires_grad,
                    _children=(aug,),
                    _op="broadcast_aug"
                )
                _aug = aug
                _aug_broad = aug_broad
                _T_i = T_i

                def _aug_back(ab=_aug_broad, a=_aug, ti=_T_i):
                    if a.requires_grad and ab.grad is not None:
                        contrib = ab.grad.sum(axis=1)   # (B, D)
                        a.grad = a.grad + contrib if a.grad is not None else contrib

                aug_broad._backward = _aug_back
                xi = xi + aug_broad

            # Slow->Fast: feedback bias from slower (coarser) level
            slow_bias = None
            if i < self.n_levels - 1 and level_h_last[i + 1] is not None:
                slow_bias = self._get_bn_up(i)(level_h_last[i + 1])  # (B, state_dim)

            yi, h_last_i = self._get_level(i).forward(xi, h_prev, slow_bias)
            level_outputs[i] = yi           # (B, T_i, D)
            level_h_last[i] = h_last_i      # (B, state_dim)
            new_states[i] = h_last_i

        # ----------------------------------------------------------------
        # Upsample all levels to T and concatenate
        # ----------------------------------------------------------------
        upsampled: List[Tensor] = []
        for i in range(self.n_levels):
            yi = level_outputs[i]    # (B, T_i, D)
            T_i = yi.shape[1]

            if T_i == T:
                upsampled.append(yi)
            else:
                # Nearest-neighbour upsample: repeat each timestep by (T // T_i)
                repeat = T // T_i
                # np.repeat along axis=1
                yi_up_np = np.repeat(yi.data, repeat, axis=1)  # (B, T_i*repeat, D)
                # Crop or pad to exactly T
                if yi_up_np.shape[1] > T:
                    yi_up_np = yi_up_np[:, :T, :]
                elif yi_up_np.shape[1] < T:
                    pad_len = T - yi_up_np.shape[1]
                    yi_up_np = np.pad(yi_up_np, ((0, 0), (0, pad_len), (0, 0)),
                                      mode="edge")

                yi_up = Tensor(
                    yi_up_np,
                    requires_grad=yi.requires_grad,
                    _children=(yi,),
                    _op=f"upsample_{i}"
                )
                _yi = yi
                _yi_up = yi_up
                _repeat = repeat
                _T_i2 = T_i

                def _up_back(up=_yi_up, src=_yi, r=_repeat, ti=_T_i2):
                    if src.requires_grad and up.grad is not None:
                        # Sum gradients from repeated positions back to source
                        g = up.grad[:, :ti * r, :]  # trim to clean repeat region
                        # reshape (B, T_i, r, D) then sum over r
                        g_folded = g.reshape(g.shape[0], ti, r, g.shape[2])
                        contrib = g_folded.sum(axis=2)   # (B, T_i, D)
                        src.grad = src.grad + contrib if src.grad is not None else contrib

                yi_up._backward = _up_back
                upsampled.append(yi_up)

        # Concatenate along feature dim: (B, T, D * n_levels)
        from engine.tensor import cat as tcat
        concat = tcat(upsampled, axis=2)  # (B, T, D * n_levels)

        output = self.output_proj(concat)  # (B, T, D)

        return output, new_states
