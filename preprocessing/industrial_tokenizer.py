"""
Industrial Tokenizer — multi-rate typed token stream for VULGARIS.

Converts heterogeneous sensor channels (continuous signals, discrete events,
control actions, regime labels) into a unified (B, T_target, d_model) sequence
suitable for the VULGARIS encoder.

Token types
-----------
CONTINUOUS : real-valued sensor readings at a fixed Hz (temperature, pressure, …)
EVENT      : sparse binary/integer event flags (alarm triggers, state changes)
ACTION     : control output channels (setpoints, actuator commands)
REGIME     : categorical operating mode labels (encoded as one-hot)

Multi-rate alignment
--------------------
Each channel may arrive at a different sample rate. The tokenizer resamples
all channels to `target_hz` using nearest-neighbour interpolation (safe for
both continuous and discrete channels — avoids creating phantom intermediate
values for events/regimes).

Architecture
------------
For each token type a dedicated Linear projection maps the (1,) or (n_classes,)
channel value to (d_model,).  A learnable token-type embedding (one per type)
is added before the projections are concatenated across the time axis.

Usage
-----
    spec = [
        ChannelSpec(idx=0, token_type=TokenType.CONTINUOUS, hz=100.0,  name="temp"),
        ChannelSpec(idx=1, token_type=TokenType.CONTINUOUS, hz=100.0,  name="pressure"),
        ChannelSpec(idx=2, token_type=TokenType.EVENT,      hz=10.0,   name="alarm"),
        ChannelSpec(idx=3, token_type=TokenType.ACTION,     hz=50.0,   name="setpoint"),
        ChannelSpec(idx=4, token_type=TokenType.REGIME,     hz=1.0,    name="mode",
                    n_classes=4),
    ]
    tok = IndustrialTokenizer(spec, d_model=256, target_hz=100.0)
    z = tok(x_dict)   # x_dict: {channel_idx: (B, T_ch) np.ndarray}
    # z: (B, T_target, d_model)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from engine.tensor import Tensor, Parameter
from engine.module import Module
from engine.layers import Linear


# ──────────────────────────────────────────────────────────────────────────────
# Token type enum
# ──────────────────────────────────────────────────────────────────────────────

class TokenType(Enum):
    CONTINUOUS = "continuous"   # real-valued, Hz-sampled
    EVENT      = "event"        # sparse binary/integer flags
    ACTION     = "action"       # control output / actuator command
    REGIME     = "regime"       # categorical operating mode (one-hot)


# ──────────────────────────────────────────────────────────────────────────────
# Channel specification
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChannelSpec:
    """
    Specification for one input channel.

    Parameters
    ----------
    idx        : column index in the raw input array
    token_type : TokenType
    hz         : sample rate of this channel (Hz)
    name       : human-readable label (used in metadata / logging)
    n_classes  : number of categories (REGIME channels only)
    """
    idx:        int
    token_type: TokenType
    hz:         float
    name:       str   = ""
    n_classes:  int   = 2   # for REGIME; ignored otherwise


# ──────────────────────────────────────────────────────────────────────────────
# Nearest-neighbour resampler
# ──────────────────────────────────────────────────────────────────────────────

def _nn_resample(signal: np.ndarray, src_len: int, dst_len: int) -> np.ndarray:
    """
    Nearest-neighbour resampling along the last axis.
    signal: (..., src_len)
    Returns (..., dst_len)
    """
    if src_len == dst_len:
        return signal
    idx = np.round(np.linspace(0, src_len - 1, dst_len)).astype(int)
    return signal[..., idx]


# ──────────────────────────────────────────────────────────────────────────────
# Industrial Tokenizer
# ──────────────────────────────────────────────────────────────────────────────

class IndustrialTokenizer(Module):
    """
    Multi-rate typed token stream tokenizer.

    Accepts a dict of {channel_idx: (B, T_ch) ndarray} and produces a
    single (B, T_target, d_model) Tensor that VULGARIS can consume directly
    in place of (or concatenated with) the raw ASE output.

    Parameters
    ----------
    specs      : list of ChannelSpec objects — one per input channel
    d_model    : output latent dimension (must match Vulgaris d_model)
    target_hz  : Hz to resample all channels to before tokenizing
    window_s   : window duration in seconds — determines T_target = target_hz * window_s
                 (pass None to infer from the first channel's length)
    """

    _TYPE_ORDER = [TokenType.CONTINUOUS, TokenType.EVENT,
                   TokenType.ACTION, TokenType.REGIME]

    def __init__(
        self,
        specs:     List[ChannelSpec],
        d_model:   int,
        target_hz: float = 100.0,
        window_s:  Optional[float] = None,
    ):
        super().__init__()
        self.specs      = specs
        self.d_model    = d_model
        self.target_hz  = target_hz
        self.window_s   = window_s

        # Per-type value projections
        # CONTINUOUS / EVENT / ACTION: scalar (1,) → d_model
        # REGIME: (n_classes,) one-hot → d_model (max n_classes across regime specs)
        max_regime_classes = max(
            (s.n_classes for s in specs if s.token_type == TokenType.REGIME),
            default=2,
        )
        self.proj_continuous = Linear(1, d_model)
        self.proj_event      = Linear(1, d_model)
        self.proj_action     = Linear(1, d_model)
        self.proj_regime     = Linear(max_regime_classes, d_model)
        self._max_regime     = max_regime_classes

        # Learnable token-type embeddings: shape (4, d_model)
        # Index: 0=CONTINUOUS, 1=EVENT, 2=ACTION, 3=REGIME
        self.type_embed = Parameter(
            np.random.randn(4, d_model).astype(np.float64) * 0.02,
            name="type_embed",
        )

        # Output projection: aggregates all channels → d_model
        n_channels = len(specs)
        self.out_proj = Linear(d_model * n_channels, d_model)

    # ──────────────────────────────────────────────────────────────────────

    def _proj_for_type(self, t: TokenType) -> Linear:
        if t == TokenType.CONTINUOUS: return self.proj_continuous
        if t == TokenType.EVENT:      return self.proj_event
        if t == TokenType.ACTION:     return self.proj_action
        return self.proj_regime

    def _type_idx(self, t: TokenType) -> int:
        return self._TYPE_ORDER.index(t)

    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x_dict: Dict[int, np.ndarray],
        window_len: Optional[int] = None,
    ) -> Tensor:
        """
        x_dict    : {channel_idx: (B, T_ch) ndarray}
        window_len: target T in samples (overrides window_s inference)

        Returns (B, T_target, d_model) Tensor.
        """
        # Infer T_target
        if window_len is not None:
            T_target = window_len
        elif self.window_s is not None:
            T_target = int(self.target_hz * self.window_s)
        else:
            # Infer from the highest-rate channel
            max_len = max(v.shape[-1] for v in x_dict.values())
            # Scale to target_hz using the spec with max hz
            max_hz = max(s.hz for s in self.specs)
            T_target = int(max_len * self.target_hz / max_hz)
        T_target = max(T_target, 1)

        # Infer batch size
        B = next(iter(x_dict.values())).shape[0] if x_dict else 1

        channel_tokens: List[Tensor] = []

        for spec in self.specs:
            raw = x_dict.get(spec.idx)
            if raw is None:
                # Missing channel: fill with zeros
                raw = np.zeros((B, T_target), dtype=np.float32)
            else:
                raw = np.asarray(raw, dtype=np.float32)

            # Resample to T_target
            src_len = raw.shape[-1]
            resampled = _nn_resample(raw, src_len, T_target)   # (B, T_target)

            if spec.token_type == TokenType.REGIME:
                # One-hot encode integer class indices
                n_cls = spec.n_classes
                cls_idx = np.clip(resampled.astype(int), 0, n_cls - 1)  # (B, T_target)
                one_hot = np.eye(n_cls, dtype=np.float32)[cls_idx]       # (B, T, n_cls)
                # Pad to max_regime_classes if needed
                if n_cls < self._max_regime:
                    pad = np.zeros((B, T_target, self._max_regime - n_cls),
                                   dtype=np.float32)
                    one_hot = np.concatenate([one_hot, pad], axis=-1)
                val_t = Tensor(one_hot.reshape(B * T_target, self._max_regime).astype(np.float64))
            else:
                # Scalar channel: (B, T_target, 1)
                val_t = Tensor(resampled.reshape(B * T_target, 1).astype(np.float64))

            proj = self._proj_for_type(spec.token_type)
            tok = proj(val_t)   # (B*T, d_model)

            # Add type embedding
            type_emb = self.type_embed.data[self._type_idx(spec.token_type)]  # (d_model,)
            tok_data = tok.data + type_emb[None, :]

            tok_out = Tensor(
                tok_data,
                requires_grad=tok.requires_grad,
                _children=(tok,),
                _op=f"type_emb_{spec.name}"
            )
            _tok, _emb_idx = tok, self._type_idx(spec.token_type)
            _te = self.type_embed

            def _te_back(t=_tok, t_out=tok_out, ti=_emb_idx, te=_te):
                if t.requires_grad and t_out.grad is not None:
                    t.grad = (t.grad + t_out.grad
                              if t.grad is not None else t_out.grad.copy())
                if te.requires_grad and t_out.grad is not None:
                    if te.grad is None:
                        te.grad = np.zeros_like(te.data)
                    te.grad[ti] += t_out.grad.sum(axis=0)

            tok_out._backward = _te_back

            # Reshape to (B, T_target, d_model)
            tok_bt = Tensor(
                tok_out.data.reshape(B, T_target, self.d_model),
                requires_grad=tok_out.requires_grad,
                _children=(tok_out,),
                _op="reshape_token"
            )
            _tok_out = tok_out

            def _rsh_back(src=_tok_out, dst=tok_bt):
                if src.requires_grad and dst.grad is not None:
                    src.grad = (src.grad + dst.grad.reshape(src.data.shape)
                                if src.grad is not None
                                else dst.grad.reshape(src.data.shape))

            tok_bt._backward = _rsh_back
            channel_tokens.append(tok_bt)

        # Concatenate all channel tokens along last dim: (B, T, d_model * C)
        stacked = np.concatenate([t.data for t in channel_tokens], axis=-1)
        stacked_t = Tensor(
            stacked,
            requires_grad=any(t.requires_grad for t in channel_tokens),
            _children=tuple(channel_tokens),
            _op="tok_concat"
        )
        _tokens = channel_tokens

        def _cat_back():
            if stacked_t.grad is None:
                return
            d = self.d_model
            for i, ct in enumerate(_tokens):
                if ct.requires_grad:
                    g = stacked_t.grad[:, :, i * d:(i + 1) * d]
                    ct.grad = ct.grad + g if ct.grad is not None else g.copy()

        stacked_t._backward = _cat_back

        # Reshape to (B*T, d_model*C) for out_proj
        B_, T_, DC = stacked_t.data.shape
        flat = Tensor(
            stacked_t.data.reshape(B_ * T_, DC),
            requires_grad=stacked_t.requires_grad,
            _children=(stacked_t,),
            _op="flat_for_proj"
        )
        _st = stacked_t

        def _flat_back():
            if _st.requires_grad and flat.grad is not None:
                _st.grad = (_st.grad + flat.grad.reshape(B_, T_, DC)
                            if _st.grad is not None
                            else flat.grad.reshape(B_, T_, DC))

        flat._backward = _flat_back

        out_flat = self.out_proj(flat)   # (B*T, d_model)
        out = Tensor(
            out_flat.data.reshape(B_, T_, self.d_model),
            requires_grad=out_flat.requires_grad,
            _children=(out_flat,),
            _op="tok_out_reshape"
        )
        _of = out_flat

        def _out_rsh_back():
            if _of.requires_grad and out.grad is not None:
                _of.grad = (_of.grad + out.grad.reshape(B_ * T_, self.d_model)
                            if _of.grad is not None
                            else out.grad.reshape(B_ * T_, self.d_model))

        out._backward = _out_rsh_back
        return out   # (B, T_target, d_model)

    # ──────────────────────────────────────────────────────────────────────

    def specs_summary(self) -> List[dict]:
        return [
            {"idx": s.idx, "type": s.token_type.value,
             "hz": s.hz, "name": s.name}
            for s in self.specs
        ]
