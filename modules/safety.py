import numpy as np
import warnings
from typing import List, Optional, Callable

from engine.tensor import Tensor, Parameter, zeros, ones, randn
from engine.module import Module
from engine.layers import Linear
from config import SafetyConfig


class CBFLayer(Module):
    """Differentiable Control Barrier Function.

    h_net: state_dim -> 64 -> n_constraints
    h(s) >= 0 defines the safe set S.
    """

    def __init__(self, state_dim: int, n_constraints: int, gamma: float):
        super().__init__()
        self.state_dim = state_dim
        self.n_constraints = n_constraints
        self.gamma = gamma

        self.fc1 = Linear(state_dim, 64)
        self.fc2 = Linear(64, n_constraints)

    def forward(self, state: Tensor) -> Tensor:
        """Returns h(s): (batch, n_constraints)."""
        hidden = self.fc1(state).tanh()
        return self.fc2(hidden)

    def violation(self, state: Tensor) -> Tensor:
        """Returns relu(-h(s)): positive where constraint violated."""
        h = self.forward(state)
        return (-h).relu()


class SpectralNormLinear(Module):
    """Linear layer with power-iteration spectral normalisation.

    Weight is scaled so σ_max(W) <= L_max (Lipschitz bound per layer).
    """

    def __init__(self, in_features: int, out_features: int,
                 n_power_iter: int = 3, L_max: float = 10.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_power_iter = n_power_iter
        self.L_max = L_max

        bound = np.sqrt(1.0 / in_features)
        w_data = np.random.uniform(-bound, bound, (out_features, in_features))
        self.weight = Parameter(w_data, name="weight")
        b_data = np.random.uniform(-bound, bound, (out_features,))
        self.bias = Parameter(b_data, name="bias")

        # Singular vector estimates (not Parameters — updated by power iteration)
        u_init = np.random.randn(out_features)
        u_init /= (np.linalg.norm(u_init) + 1e-12)
        v_init = np.random.randn(in_features)
        v_init /= (np.linalg.norm(v_init) + 1e-12)
        object.__setattr__(self, "u_vec", u_init)
        object.__setattr__(self, "v_vec", v_init)

    def _spectral_norm(self) -> Tensor:
        """Power iteration to estimate σ_max(W); returns spectral-normalised W as Tensor.

        Updates u_vec and v_vec in-place (no grad tracking on singular vectors).
        Scales W if σ > L_max.
        """
        W = self.weight.data  # (out, in)
        u = self.u_vec.copy()
        v = self.v_vec.copy()

        for _ in range(self.n_power_iter):
            # v = W.T @ u / ||W.T @ u||
            v_new = W.T @ u
            v_new_norm = np.linalg.norm(v_new) + 1e-12
            v = v_new / v_new_norm

            # u = W @ v / ||W @ v||
            u_new = W @ v
            u_new_norm = np.linalg.norm(u_new) + 1e-12
            u = u_new / u_new_norm

        # σ = u.T @ W @ v
        sigma = float(u @ W @ v)

        # Update estimates (no gradient needed)
        object.__setattr__(self, "u_vec", u)
        object.__setattr__(self, "v_vec", v)

        if sigma > self.L_max:
            # Scale W: W_norm = W * (L_max / sigma)
            # We build a differentiable scaled Tensor from the Parameter
            scale = self.L_max / sigma
            W_scaled = self.weight * scale
        else:
            W_scaled = self.weight

        return W_scaled

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using spectral-normalised weight."""
        W_norm = self._spectral_norm()
        out = x @ W_norm.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out


class SafetyPolicyHead(Module):
    """Safety-Critical Policy Head with CBF-based safety filter and Lipschitz bounds."""

    def __init__(self, d_model: int, action_dim: int, config: SafetyConfig):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.config = config
        self.gamma = config.cbf_gamma
        self.eps_qp = 1e-6  # numerical stability in QP projection

        # Policy network: 2 SpectralNormLinear layers
        self.policy_fc1 = SpectralNormLinear(d_model, 128,
                                             L_max=config.lipschitz_bound)
        self.policy_fc2 = SpectralNormLinear(128, action_dim,
                                             L_max=config.lipschitz_bound)

        # CBF layer
        self.cbf = CBFLayer(d_model, config.n_constraints, config.cbf_gamma)

    def forward(self, state: Tensor) -> Tensor:
        """
        1. Compute nominal action u_nom = policy_net(state)
        2. Compute h = cbf(state)
        3. Apply safety filter to get u_safe
        Returns u_safe: (batch, action_dim)
        """
        # Nominal action
        hidden = self.policy_fc1(state).tanh()
        u_nom = self.policy_fc2(hidden)

        # CBF values
        h = self.cbf(state)   # (batch, n_constraints)

        # Safety filter
        u_safe = self._safety_filter(u_nom, h, state)
        return u_safe

    def _safety_filter(self, u_nom: Tensor, h: Tensor, state: Tensor) -> Tensor:
        """Differentiable CBF projection (closed-form QP via projected gradient).

        For each constraint h_i, compute:
            ∇h_i(s)·u_nom + γ*h_i(s)  — safety margin
        If margin < 0: apply correction to u_nom.

        correction = max(0, -∇h·u_nom - γ*h) / (||∇h||^2 + ε) * ∇h

        Uses finite differences on state.data for ∇h_i.
        """
        state_np = state.data   # (batch, d_model)
        u_np = u_nom.data       # (batch, action_dim)
        h_np = h.data           # (batch, n_constraints)

        batch = state_np.shape[0]
        fd_eps = 1e-4

        # Compute ∇h_i w.r.t. state via central finite differences
        # grad_h: (batch, n_constraints, d_model)
        grad_h = np.zeros((batch, self.config.n_constraints, self.d_model),
                          dtype=np.float64)

        for j in range(self.d_model):
            s_plus = state_np.copy()
            s_minus = state_np.copy()
            s_plus[:, j] += fd_eps
            s_minus[:, j] -= fd_eps

            h_plus = self.cbf(Tensor(s_plus, requires_grad=False)).data
            h_minus = self.cbf(Tensor(s_minus, requires_grad=False)).data
            # Central difference: (h(s+ε*e_j) - h(s-ε*e_j)) / (2ε)
            grad_h[:, :, j] = (h_plus - h_minus) / (2.0 * fd_eps)

        # For each batch element, accumulate corrections across all constraints
        correction_np = np.zeros_like(u_np)

        for c in range(self.config.n_constraints):
            grad_h_c = grad_h[:, c, :]   # (batch, d_model)
            h_c = h_np[:, c]             # (batch,)

            # ∇h_c · u_nom: project gradient onto action space via u_nom dot
            # We interpret ∇h_c(s) ∈ R^{d_model} as a direction in action space
            # by truncating/padding to action_dim for the projection
            action_dim = self.action_dim
            if self.d_model >= action_dim:
                grad_h_act = grad_h_c[:, :action_dim]  # (batch, action_dim)
            else:
                pad = np.zeros((batch, action_dim - self.d_model))
                grad_h_act = np.concatenate([grad_h_c, pad], axis=1)

            # Lie derivative proxy: ∇h · u_nom
            lie_deriv = np.sum(grad_h_act * u_np, axis=1)  # (batch,)

            # CBF constraint margin: Lie_deriv + γ*h_c >= 0
            margin = lie_deriv + self.gamma * h_c    # (batch,)
            violation_mask = margin < 0.0             # (batch,)

            if not np.any(violation_mask):
                continue

            # Correction magnitude: max(0, -margin) / (||∇h_act||^2 + ε)
            norm_sq = np.sum(grad_h_act ** 2, axis=1) + self.eps_qp  # (batch,)
            corr_mag = np.maximum(0.0, -margin) / norm_sq            # (batch,)

            # correction vector: corr_mag * grad_h_act
            corr_vec = corr_mag[:, None] * grad_h_act                 # (batch, action_dim)
            corr_vec *= violation_mask[:, None].astype(np.float64)

            if h_c.min() < -1e-4:
                idx = np.where(h_c < -1e-4)[0]
                warnings.warn(
                    f"CBFLayer: constraint {c} violated at {len(idx)} states "
                    f"(min h={h_c.min():.4f})"
                )

            correction_np += corr_vec

        # Build corrected action as Tensor, preserving gradient chain through u_nom
        corr_t = Tensor(correction_np, requires_grad=False)
        return u_nom + corr_t

    def cbf_loss(self, state: Tensor, next_state: Tensor) -> Tensor:
        """Penalise CBF constraint violations between consecutive states.

        L_cbf = Σ_i relu(-(h(s') - (1-γ)*h(s)))
        """
        h_curr = self.cbf(state)       # (batch, n_constraints)
        h_next = self.cbf(next_state)  # (batch, n_constraints)

        # (1-γ)*h(s)
        decay = Tensor(np.array([1.0 - self.gamma]), requires_grad=False)
        rhs = h_curr * decay

        # relu(-(h(s') - rhs)) = relu(rhs - h(s'))
        violation = (rhs - h_next).relu()
        return violation.sum()

    def is_safe(self, state_np: np.ndarray) -> bool:
        """Returns True if all CBF constraints satisfied (h_i(s) >= 0)."""
        if state_np.ndim == 1:
            state_np = state_np[None, :]
        state_t = Tensor(state_np, requires_grad=False)
        h = self.cbf(state_t).data   # (batch, n_constraints)
        return bool(np.all(h >= 0.0))
