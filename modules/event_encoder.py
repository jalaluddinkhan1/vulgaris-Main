from __future__ import annotations

import numpy as np

from engine.tensor import Tensor, Parameter
from engine.module import Module
from engine.layers import Linear, RMSNorm


class EventEncoder(Module):
    """
    Encodes a sequence of discrete events (alarm codes, log levels, deployment
    IDs, protocol events) into a continuous latent representation compatible
    with CMLA's multi-modal fusion path.

    Input  : (B, L)        — integer token IDs, L = event sequence length
    Output : (B, T, d_model) — event context broadcast over T signal timesteps

    Architecture
    ------------
    1. Embedding lookup     (vocab_size, d_embed)
    2. Attention pooling    learns which events matter most
    3. Linear projection    d_embed → d_model
    4. Broadcast            (B, d_model) → (B, T, d_model)

    Usage
    -----
        enc = EventEncoder(vocab_size=512, d_model=256)

        # From raw token IDs
        events = np.array([[3, 7, 0, 12, ...]])  # (B, L) int32

        # Produce modality tensor for CMLA
        z_events = enc(events, T=64)    # (B, T=64, d_model)

        # Multi-modal fusion
        fused, loss, _ = cmla([z_sensors, z_events])

    Domain examples
    ---------------
    OT    : alarm codes (HIGH_PRESSURE=3, VALVE_FAULT=7), process step IDs
    IT    : log severity (DEBUG/INFO/WARN/ERROR), Kubernetes event types
    Telecom: alarm categories (CELL_DOWN=1, HO_FAIL=2), protocol error codes
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_embed: int | None = None,
        max_len: int = 512,
        pad_id: int = 0,
    ):
        """
        vocab_size : number of distinct event types (include PAD=0 by convention)
        d_model    : output dimension (must match the model's d_model)
        d_embed    : internal embedding dimension (default = d_model // 2)
        max_len    : maximum event sequence length for positional encoding
        pad_id     : token ID used for padding (masked in attention pool)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.d_embed    = d_embed or max(d_model // 2, 32)
        self.max_len    = max_len
        self.pad_id     = pad_id

        # Embedding table: (vocab_size, d_embed)
        scale = (self.d_embed ** -0.5)
        self.embedding = Parameter(
            (np.random.randn(vocab_size, self.d_embed) * scale).astype(np.float32),
            name="event_embedding",
        )

        # Learnable positional encoding: (max_len, d_embed)
        self.pos_enc = Parameter(
            (np.random.randn(max_len, self.d_embed) * 0.02).astype(np.float32),
            name="event_pos_enc",
        )

        # Attention pool: linear score per position → softmax weights
        self.pool_q = Linear(self.d_embed, 1)

        # Projection to d_model
        self.proj = Linear(self.d_embed, d_model)
        self.norm = RMSNorm(d_model)

    # ──────────────────────────────────────────────────────────────────────

    def forward(self, event_ids: np.ndarray, T: int) -> Tensor:
        """
        event_ids : (B, L) int32/int64 — token IDs (0 = pad)
        T         : int — signal sequence length to broadcast event context over

        Returns   : (B, T, d_model)
        """
        B, L = event_ids.shape
        L    = min(L, self.max_len)
        ids  = event_ids[:, :L].clip(0, self.vocab_size - 1)  # (B, L)

        # ── 1. Embedding lookup ───────────────────────────────────────────
        emb_np = self.embedding.data[ids]          # (B, L, d_embed) — numpy gather
        pos_np = self.pos_enc.data[:L, :][None]    # (1, L, d_embed)
        h_np   = (emb_np + pos_np).astype(np.float32)

        # Build Tensor with gradient connected to embedding and pos_enc
        h = Tensor(
            h_np,
            requires_grad=self.embedding.requires_grad or self.pos_enc.requires_grad,
            _children=(self.embedding, self.pos_enc),
            _op="event_embed",
        )
        _emb, _pos, _ids, _L = self.embedding, self.pos_enc, ids, L

        def _embed_back():
            if h.grad is None:
                return
            g = h.grad   # (B, L, d_embed)
            if _emb.requires_grad:
                g_emb = np.zeros_like(_emb.data)
                np.add.at(g_emb, _ids, g)
                _emb.grad = _emb.grad + g_emb if _emb.grad is not None else g_emb
            if _pos.requires_grad:
                g_pos = g.sum(axis=0)   # (L, d_embed)
                g_full = np.zeros_like(_pos.data)
                g_full[:_L] = g_pos
                _pos.grad = _pos.grad + g_full if _pos.grad is not None else g_full

        h._backward = _embed_back

        # ── 2. Attention pool over event sequence ─────────────────────────
        # Mask padding tokens so they don't influence the context vector
        pad_mask = (ids == self.pad_id)[:, :, None].astype(np.float32)   # (B, L, 1)

        # pool_q: (B, L, d_embed) → (B, L, 1) → softmax
        scores = self.pool_q(h)                                           # (B, L, 1)
        scores_np = scores.data.copy()
        scores_np[pad_mask.squeeze(-1) == 1] = -1e9   # mask pad positions
        exp_s  = np.exp(scores_np - scores_np.max(axis=1, keepdims=True))
        alpha  = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-8)       # (B, L, 1)

        # Weighted sum: (B, d_embed)
        ctx_np = (alpha * h.data).sum(axis=1).astype(np.float32)

        ctx = Tensor(
            ctx_np,
            requires_grad=h.requires_grad,
            _children=(h,),
            _op="event_pool",
        )
        _h, _alpha = h, alpha

        def _pool_back():
            if _h.requires_grad and ctx.grad is not None:
                g_ctx = ctx.grad                              # (B, d_embed)
                g_h   = (_alpha * g_ctx[:, None, :]).astype(np.float32)  # (B, L, d_embed)
                _h.grad = _h.grad + g_h if _h.grad is not None else g_h

        ctx._backward = _pool_back

        # ── 3. Project to d_model ─────────────────────────────────────────
        out = self.norm(self.proj(ctx))   # (B, d_model)

        # ── 4. Broadcast over signal time axis ────────────────────────────
        out_np = np.broadcast_to(
            out.data[:, None, :], (B, T, self.d_model)
        ).copy().astype(np.float32)

        out_t = Tensor(
            out_np,
            requires_grad=out.requires_grad,
            _children=(out,),
            _op="event_broadcast",
        )
        _out, _T = out, T

        def _broadcast_back():
            if _out.requires_grad and out_t.grad is not None:
                g = out_t.grad.sum(axis=1)   # (B, d_model) — reduce time axis
                _out.grad = _out.grad + g if _out.grad is not None else g

        out_t._backward = _broadcast_back
        return out_t   # (B, T, d_model)

    def encode_events(self, event_ids: np.ndarray, T: int) -> Tensor:
        """Alias for forward() — preferred name in user-facing code."""
        return self.forward(event_ids, T)
