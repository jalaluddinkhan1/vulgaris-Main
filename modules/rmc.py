"""
Regime Mixture Core (RMC) — soft Mixture-of-Experts over operating regimes.

Each operating regime (startup, steady-state, fault, CIP, …) has distinct
signal statistics. A single linear layer cannot adapt optimally to all of them.
RMC routes each latent token to a soft convex combination of K expert networks,
each specialising on one regime cluster.

Architecture
------------
    scores  = gate_proj(z)                        (B, T, K)
    weights = softmax(scores / τ)                 (B, T, K)  — routing weights
    experts = [E_k(z) for k in range(K)]          K × (B, T, d_model)
    z_out   = Σ_k  weights[:,:,k:k+1] * E_k(z)   (B, T, d_model)

Load-balancing loss (Switch Transformer style)
----------------------------------------------
    f_k = mean_{B,T}[ I(argmax(weights) == k) ]   fraction of tokens to expert k
    P_k = mean_{B,T}[ weights[:,:,k] ]             mean gate probability

    L_balance = K · Σ_k  f_k · P_k               → 0 when uniform routing

This loss is added to aux["rmc_balance_loss"] and multiplied by lambda_rmc
(default 0.01) in VulgarisLoss.

Usage
-----
    rmc = RegimeMixtureCore(d_model=256, n_experts=4)
    z_out, balance_loss = rmc(z)   # z: (B, T, d_model)
    # wire as:  z = z + z_out  in Vulgaris.forward()
from __future__ import annotations

"""


import numpy as np

from engine.tensor import Tensor, Parameter
from engine.module import Module
from engine.layers import Linear


class RegimeMixtureCore(Module):
    """
    Soft Mixture-of-Experts residual block for regime-aware processing.

    Parameters
    ----------
    d_model   : latent dimension (must match Vulgaris d_model)
    n_experts : number of regime expert networks (default 4)
    tau       : gating temperature — lower values → harder routing (default 1.0)
    """

    def __init__(self, d_model: int, n_experts: int = 4, tau: float = 1.0,
                 top_k: int = 0):
        """
        top_k : number of experts activated per token (0 = all experts, soft routing).
                top_k=2 gives Switch-Transformer-style sparse routing with straight-through
                gradient so inactive experts still receive gradients.
        """
        super().__init__()
        self.d_model   = d_model
        self.n_experts = n_experts
        self.tau       = tau
        self.top_k     = top_k if 0 < top_k <= n_experts else n_experts

        # Gating network: d_model → K logits
        self.gate_proj = Linear(d_model, n_experts)
        self.last_regime_weights: np.ndarray | None = None

        for k in range(n_experts):
            setattr(self, f"expert_{k}", Linear(d_model, d_model))

    def _experts(self):
        return [getattr(self, f"expert_{k}") for k in range(self.n_experts)]

    # ──────────────────────────────────────────────────────────────────────

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        z : (B, T, d_model)
        Returns:
            z_out          : (B, T, d_model)  — weighted expert mixture
            balance_loss   : scalar Tensor    — Switch-Transformer load-balancing
            regime weights are stored on self.last_regime_weights for callers
            that need them, such as CRG regime-conditioned routing.
        """
        B, T, D = z.data.shape
        K = self.n_experts

        # Flatten to (B*T, D) for linear operations
        z_flat = z.reshape(B * T, D)   # (N, D),  N = B*T

        # ── Gating ────────────────────────────────────────────────────────
        scores_flat = self.gate_proj(z_flat)   # (N, K)

        # Softmax with temperature
        scores_np = scores_flat.data / self.tau
        scores_np = scores_np - scores_np.max(axis=-1, keepdims=True)   # stability
        exp_s = np.exp(scores_np)
        denom = exp_s.sum(axis=-1, keepdims=True) + 1e-8
        weights_np = exp_s / denom   # (N, K) — soft routing probabilities

        # Straight-through sparse top-k: keep full softmax weights for gradient
        # computation (so inactive experts still train), but zero out non-top-k
        # contributions to the forward output.
        weights_np_full = weights_np.copy()   # (N, K) — pre-mask, used in backward

        if self.top_k < K:
            topk_idx = np.argsort(weights_np, axis=-1)[:, -self.top_k:]  # (N, top_k)
            mask = np.zeros_like(weights_np)
            np.put_along_axis(mask, topk_idx, 1.0, axis=-1)
            weights_np = weights_np * mask
            denom_topk = weights_np.sum(axis=-1, keepdims=True) + 1e-8
            weights_np = weights_np / denom_topk   # re-normalise sparse weights

        weights = Tensor(
            weights_np,
            requires_grad=scores_flat.requires_grad,
            _children=(scores_flat,),
            _op="rmc_softmax"
        )
        _sf = scores_flat
        _tau = self.tau
        _w_full = weights_np_full   # straight-through: grad uses full softmax weights

        def _softmax_back():
            if _sf.requires_grad and weights.grad is not None:
                g = weights.grad                                       # (N, K)
                # Straight-through: propagate gradient as if no top-k mask was applied
                sg = (g * _w_full).sum(axis=-1, keepdims=True)        # (N, 1)
                d_scores = _w_full * (g - sg) / _tau
                _sf.grad = _sf.grad + d_scores if _sf.grad is not None else d_scores

        weights._backward = _softmax_back

        # ── Expert outputs ─────────────────────────────────────────────────
        # Each expert produces (N, D); sum weighted contributions
        z_out_np = np.zeros((B * T, D), dtype=np.float32)
        expert_outputs = []
        for k, expert in enumerate(self._experts()):
            e_k = expert(z_flat)           # (N, D)
            w_k = weights_np[:, k:k + 1]  # (N, 1)
            z_out_np += w_k * e_k.data
            expert_outputs.append((e_k, weights_np[:, k]))

        # Build output tensor with backward that routes gradients to each expert
        # and back through weights
        all_expert_out = [e for e, _ in expert_outputs]
        z_out_flat = Tensor(
            z_out_np,
            requires_grad=z_flat.requires_grad,
            _children=tuple([weights] + all_expert_out),
            _op="rmc_mix"
        )
        _weights = weights
        _expert_outputs_list = expert_outputs
        _K = K

        def _mix_back():
            if z_out_flat.grad is None:
                return
            g_out = z_out_flat.grad   # (N, D)
            # Gradient to each expert_k output: g_e_k = weights_k * g_out
            # Gradient to weights_k: g_w_k = sum_d(g_out * e_k_data)
            g_weights_np = np.zeros((_weights.data.shape[0], _K), dtype=np.float32)
            for k, (e_k_t, w_k_np) in enumerate(_expert_outputs_list):
                g_ek = w_k_np[:, None] * g_out       # (N, D)
                if e_k_t.requires_grad:
                    e_k_t.grad = (e_k_t.grad + g_ek
                                  if e_k_t.grad is not None else g_ek.copy())
                g_weights_np[:, k] = (g_out * e_k_t.data).sum(axis=-1)  # (N,)
            if _weights.requires_grad:
                _weights.grad = (_weights.grad + g_weights_np
                                 if _weights.grad is not None else g_weights_np.copy())

        z_out_flat._backward = _mix_back

        # Reshape back to (B, T, D)
        z_out = z_out_flat.reshape(B, T, D)

        # ── Load-balancing loss ────────────────────────────────────────────
        # f_k: hard assignment fraction (straight-through, no gradient)
        hard_assign = np.argmax(weights_np, axis=-1)   # (N,) ∈ {0,...,K-1}
        f_k = np.array([np.mean(hard_assign == k) for k in range(K)],
                       dtype=np.float32)               # (K,)

        # P_k: mean soft weight per expert (differentiable)
        P_k_np = weights_np.mean(axis=0)               # (K,)

        P_k = Tensor(
            P_k_np,
            requires_grad=weights.requires_grad,
            _children=(weights,),
            _op="rmc_Pk"
        )
        _w = weights
        _N = B * T

        def _Pk_back():
            if _w.requires_grad and P_k.grad is not None:
                g_Pk = P_k.grad                                # (K,)
                g_w = np.broadcast_to(g_Pk[None, :] / _N,
                                      _w.data.shape).copy()   # (N, K)
                _w.grad = _w.grad + g_w if _w.grad is not None else g_w

        P_k._backward = _Pk_back

        # balance_loss = K * sum_k(f_k * P_k)
        # f_k is a constant (no gradient); only P_k carries gradient
        balance_np = np.array([[K * float(np.dot(f_k, P_k_np))]], dtype=np.float32)
        balance_loss = Tensor(
            balance_np,
            requires_grad=P_k.requires_grad,
            _children=(P_k,),
            _op="rmc_balance"
        )
        _fk = f_k
        _Pk_t = P_k

        def _balance_back():
            if _Pk_t.requires_grad and balance_loss.grad is not None:
                g_scalar = float(balance_loss.grad.sum())
                g_Pk = g_scalar * K * _fk    # (K,)
                _Pk_t.grad = (_Pk_t.grad + g_Pk
                              if _Pk_t.grad is not None else g_Pk.copy())

        balance_loss._backward = _balance_back

        # Return mean per-regime routing weight so callers (e.g. CRG) never
        # need to re-run gate_proj to extract regime context.
        regime_weights = weights_np_full.mean(axis=0).astype(np.float32)   # (K,)
        self.last_regime_weights = regime_weights
        return z_out, balance_loss

    def regime_assignments(self, z: Tensor) -> np.ndarray:
        """
        Returns the hard expert assignment per token — useful for visualising
        which regime each timestep is assigned to.

        Returns: (B, T) int array of expert indices (0 … K-1).
        """
        B, T, D = z.data.shape
        z_flat = z.reshape(B * T, D)
        scores_flat = self.gate_proj(z_flat)
        assignments = np.argmax(scores_flat.data, axis=-1).reshape(B, T)
        return assignments
