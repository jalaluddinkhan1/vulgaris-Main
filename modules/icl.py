"""In-Context Learning adapter for VULGARIS.

Enables zero-shot adaptation at inference time by conditioning the model on a
small set of labelled reference examples — no gradient updates required.

Usage
-----
context = [
    (x_ref1, y_ref1),   # x: (B, C, T_ref), y: (B, output_dim) numpy arrays
    (x_ref2, y_ref2),
]
output, aux = model(x_query, context=context)

How it works
------------
1. ContextEncoder encodes each (x_ref, y_ref) pair:
   - x_ref passes through RevIN + ASE to produce z_ref (B, T_ref, d_model).
   - z_ref is mean-pooled to a single vector (B, d_model).
   - y_ref is projected to d_model and added to the pooled z_ref.
   - Result: one context vector per example, (B, d_model).

2. InContextAdapter applies cross-attention:
   - Query  : the model's latent z after SSSR  (B, T, d_model)
   - Context: stacked context vectors           (B, n_ctx, d_model)
   - Output : context-conditioned residual      (B, T, d_model)

3. The residual is added back to z before CRG/HMB/output.
from __future__ import annotations

"""

from typing import TYPE_CHECKING
import numpy as np

from engine.tensor import Tensor, Parameter
from engine.module import Module
from engine.layers import Linear, RMSNorm, CrossAttention

if TYPE_CHECKING:
    from memory.episodic import EpisodicMemory


class ContextEncoder(Module):
    """
    Encodes a single (x_ref, y_ref) pair into a single context vector.

    x_ref : (B, C, T_ref)  — reference input window
    y_ref : (B, output_dim) — known label for that window

    Returns (B, d_model) Tensor.
    """

    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.y_proj   = Linear(output_dim, d_model, bias=True)
        # Learned query for attention pooling — replaces mean-pool
        self.pool_q   = Linear(d_model, 1, bias=False)
        self.norm     = RMSNorm(d_model)

    def forward(self, z_ref: Tensor, y_ref: Tensor) -> Tensor:
        """
        z_ref : (B, T_ref, d_model)  — already ASE-encoded reference latent
        y_ref : (B, output_dim)
        Returns (B, d_model).
        """
        B, _, D = z_ref.data.shape

        # Attention pooling: learned scalar score per timestep via pool_q,
        # then softmax-weighted sum. Outperforms mean-pool by letting the
        # encoder focus on the most informative timesteps.
        scores_3d = self.pool_q(z_ref)                        # (B, T_ref, 1)
        scores = scores_3d.data[:, :, 0] / np.sqrt(D)         # (B, T_ref)
        scores_shifted = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores_shifted)
        weights = exp_s / exp_s.sum(axis=1, keepdims=True)    # (B, T_ref)

        pool_np = np.einsum("bt,btd->bd", weights, z_ref.data).astype(np.float32)
        pool_t = Tensor(
            pool_np,
            requires_grad=z_ref.requires_grad or scores_3d.requires_grad,
            _children=(z_ref, scores_3d),
            _op="attn_pool"
        )
        _z, _s3, _w = z_ref, scores_3d, weights

        def _pool_back():
            if pool_t.grad is None:
                return
            g = pool_t.grad  # (B, D)
            # Gradient through direct weighted-sum path: d(pool)/d(z_t) += w_t
            if _z.requires_grad:
                g_direct = _w[:, :, None] * g[:, None, :]     # (B, T_ref, D)
                _z.grad = _z.grad + g_direct if _z.grad is not None else g_direct
            # Gradient through scores (for pool_q weights):
            # dL/ds_t = w_t * (g · z_t − g · pool)
            if _s3.requires_grad:
                gz_t  = np.einsum("bd,btd->bt", g, _z.data)   # (B, T_ref)
                g_pool = np.einsum("bd,bd->b", g, pool_np)[:, None]  # (B, 1)
                g_s = (_w * (gz_t - g_pool) / np.sqrt(D))[:, :, None].astype(np.float32)
                _s3.grad = _s3.grad + g_s if _s3.grad is not None else g_s

        pool_t._backward = _pool_back

        # Project label and fuse
        y_emb = self.y_proj(y_ref)          # (B, d_model)
        ctx_vec = pool_t + y_emb            # (B, d_model)
        return self.norm(ctx_vec)           # (B, d_model)


class InContextAdapter(Module):
    """
    Conditions model latent z on a set of encoded context vectors via
    cross-attention, then returns a residual (B, T, d_model).

    Parameters
    ----------
    d_model    : model dimension
    n_heads    : attention heads for cross-attention
    max_ctx    : maximum number of context examples (used only for repr)
    """

    def __init__(self, d_model: int, n_heads: int = 4, max_ctx: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_ctx = max_ctx

        self.cross_attn = CrossAttention(d_model=d_model, n_heads=n_heads)
        # Gate: learnable scalar per-channel, initialised near zero so the
        # adapter starts dormant and grows only when it helps.
        self.gate = Parameter(
            np.zeros(d_model, dtype=np.float32) + 0.1, name="icl_gate"
        )

    def forward(self, z: Tensor, ctx_vecs: Tensor) -> Tensor:
        """
        z        : (B, T, d_model)   — current model latent
        ctx_vecs : (B, n_ctx, d_model) — stacked context vectors
        Returns  : (B, T, d_model)   — gated residual
        """
        residual = self.cross_attn(z, ctx_vecs)   # (B, T, d_model)
        # Apply learnable gate (sigmoid so it's bounded [0, 1])
        gate_np = 1.0 / (1.0 + np.exp(-np.clip(self.gate.data, -20, 20)))  # (d_model,)
        gate_t = Tensor(
            gate_np.reshape(1, 1, -1).astype(np.float32),
            requires_grad=self.gate.requires_grad,
            _children=(self.gate,),
            _op="icl_gate_sigmoid"
        )

        _gate_raw = self.gate
        _gate_np = gate_np

        def _gate_back():
            if _gate_raw.requires_grad and gate_t.grad is not None:
                g = gate_t.grad.reshape(-1)           # (d_model,)
                dsigmoid = _gate_np * (1.0 - _gate_np)
                contrib = (g * dsigmoid).astype(np.float32)
                _gate_raw.grad = (_gate_raw.grad + contrib
                                  if _gate_raw.grad is not None else contrib)

        gate_t._backward = _gate_back

        return residual * gate_t   # element-wise scale

    def __repr__(self) -> str:
        return (f"InContextAdapter(d_model={self.d_model}, n_heads={self.n_heads}, "
                f"n_params={self.n_params():,})")


class InContextLearning(Module):
    """
    Full ICL pipeline: encodes context pairs then applies cross-attention
    conditioning to the model latent.

    Used internally by Vulgaris.forward() when `context` is provided.

    Parameters
    ----------
    d_model         : model dimension
    output_dim      : dimensionality of y labels in context pairs
    n_heads         : cross-attention heads
    episodic_memory : optional EpisodicMemory — when provided, successful context
                      encodings are stored and retrieved automatically at inference
                      time, giving ICL persistent memory across sessions.
    """

    def __init__(self, d_model: int, output_dim: int, n_heads: int = 4,
                 episodic_memory: "EpisodicMemory | None" = None):
        super().__init__()
        self.encoder        = ContextEncoder(d_model=d_model, output_dim=output_dim)
        self.adapter        = InContextAdapter(d_model=d_model, n_heads=n_heads)
        self.d_model        = d_model
        self.output_dim     = output_dim
        # EpisodicMemory instance (not a Module — no parameters, pure numpy state)
        self.episodic_memory = episodic_memory

    def encode_context(
        self,
        context_latents: list[Tensor],   # each (B, T_ref, d_model) — pre-encoded
        context_labels:  list[Tensor],   # each (B, output_dim)
        retrieve_k: int = 0,             # if > 0, augment with top-k episodes from memory
    ) -> Tensor:
        """
        Encodes each (z_ref, y_ref) pair and stacks into (B, n_ctx, d_model).

        When `episodic_memory` is attached and `retrieve_k > 0`, the method:
          1. Retrieves top-k similar past episodes from memory using the first
             provided context as the query.
          2. Prepends those retrieved vectors to the current context set so
             the cross-attention sees both retrieved history and the current context.
          3. Stores each newly encoded context vector in episodic memory for
             future retrieval.
        """
        ctx_vecs = []
        for z_ref, y_ref in zip(context_latents, context_labels):
            vec = self.encoder(z_ref, y_ref)   # (B, d_model)
            ctx_vecs.append(vec)

            # Store into episodic memory (batch item 0 as the representative key)
            if self.episodic_memory is not None:
                key_np   = vec.data[0]                         # (d_model,) float32
                label_np = self.encoder.y_proj(y_ref).data[0]  # (d_model,) projected label
                self.episodic_memory.store(key_np, label_np)

        if not ctx_vecs:
            raise ValueError("context must contain at least one example")

        # Episodic retrieval: prepend retrieved past contexts to the current set
        if self.episodic_memory is not None and retrieve_k > 0 and len(self.episodic_memory) > 0:
            query_np = ctx_vecs[0].data[0]   # (d_model,) — use first encoded vec as query
            ret_keys, ret_vals, _, _ = self.episodic_memory.retrieve(query_np, top_k=retrieve_k)
            # ret_keys: (k, d_model) — inject as additional context vectors (no grad)
            import numpy as _np
            B_ctx = ctx_vecs[0].data.shape[0]
            for i in range(len(ret_keys)):
                # Broadcast retrieved key across the batch
                ret_np = _np.tile(ret_keys[i:i+1], (B_ctx, 1)).astype(_np.float32)
                ret_t  = Tensor(ret_np, requires_grad=False)
                ctx_vecs.insert(0, ret_t)   # prepend — most similar first

        # Stack: list of (B, d_model) -> (B, n_ctx, d_model)
        n_ctx = len(ctx_vecs)
        B, D = ctx_vecs[0].data.shape
        stacked_np = np.stack([v.data for v in ctx_vecs], axis=1).astype(np.float32)

        stacked_t = Tensor(
            stacked_np,
            requires_grad=any(v.requires_grad for v in ctx_vecs),
            _children=tuple(ctx_vecs),
            _op="ctx_stack"
        )

        _ctx_vecs = ctx_vecs
        _n = n_ctx

        def _stack_back():
            if stacked_t.grad is None:
                return
            for i, v in enumerate(_ctx_vecs):
                if v.requires_grad:
                    g = stacked_t.grad[:, i, :].astype(np.float32)
                    v.grad = v.grad + g if v.grad is not None else g

        stacked_t._backward = _stack_back
        return stacked_t

    def forward(self, z: Tensor, ctx_stack: "Tensor | None") -> Tensor:
        """
        z         : (B, T, d_model)
        ctx_stack : (B, n_ctx, d_model) or None — when None, returns z unchanged.
        Returns   : (B, T, d_model) — z conditioned on context
        """
        if ctx_stack is None:
            return z
        return self.adapter(z, ctx_stack)
