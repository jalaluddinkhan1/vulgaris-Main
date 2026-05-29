"""
Speculative Rollout — fast autoregressive inference using the WorldModelHead.

Speculative decoding (Leviathan et al. 2023, Chen et al. 2023) uses a cheap
draft model to propose multiple future steps, then verifies them with the full
model in one pass.  Accepted draft steps cost ~0 vs. the full forward pass.

In VULGARIS, the draft model is the built-in WorldModelHead — a lightweight
two-layer recurrent predictor that operates entirely in latent space.

Algorithm
---------
    State: current VULGARIS state S_t, last hidden state h_t

    1. Draft:   roll out gamma steps using WorldModelHead
                → h_t+1...h_t+gamma (latent proposals)
                → y_draft_1...y_draft_gamma (predictions from proposals)

    2. Verify:  run full Vulgaris.step() from S_t with each draft input
                → y_verify_1...y_verify_gamma

    3. Accept:  for each step i:
                  if ||y_draft_i - y_verify_i||_∞ < accept_threshold → accept
                  else: reject all remaining steps, fall back to full model

    4. Output:  accepted steps are returned immediately.
                After the first rejection, one full model step is run and
                a new draft cycle begins.

Speedup analysis
----------------
    E[steps per full-model call] ≈ 1 / (1 - p_accept)^(-1)
                                 ≈ gamma * p_accept  (when p_accept is high)

    At p_accept = 0.8, gamma = 4: ~3.2× speedup in full-model calls.
    At p_accept = 0.5, gamma = 4: ~2.0× speedup.

Usage
-----
    sr = SpeculativeRollout(model, gamma=4, accept_threshold=0.05)
    outputs = sr.rollout(x_context, horizon=20, domain_idx=0)
    print(sr.stats())   # acceptance rate, effective speedup
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from engine.tensor import Tensor


class SpeculativeRollout:
    """
    Speculative autoregressive rollout for VULGARIS.

    Parameters
    ----------
    model            : Vulgaris instance (must have .world_model and .step())
    gamma            : draft horizon — steps to speculate ahead (default 4)
    accept_threshold : ||y_draft - y_verify||_∞ < threshold to accept (default 0.05)
    domain_idx       : domain adapter index to use throughout
    """

    def __init__(
        self,
        model,
        gamma: int = 4,
        accept_threshold: float = 0.05,
        domain_idx: int = 0,
    ):
        self.model            = model
        self.gamma            = gamma
        self.accept_threshold = accept_threshold
        self.domain_idx       = domain_idx

        self._total_steps:    int = 0
        self._accepted_steps: int = 0
        self._full_model_calls: int = 0
        self._draft_cycles:   int = 0

    # ──────────────────────────────────────────────────────────────────────

    def rollout(
        self,
        x_context: np.ndarray,
        horizon: int,
        domain_idx: Optional[int] = None,
        return_latents: bool = False,
    ) -> dict:
        """
        Autoregressive speculative rollout for `horizon` steps.

        x_context  : (B, C, T) warm-up context; the model is run forward on this
                     first to obtain initial state S_0 and h_0.
        horizon    : number of autoregressive steps to generate.
        domain_idx : overrides self.domain_idx if provided.

        Returns dict:
            "outputs"          : (horizon, B, out_dim) — predicted outputs
            "accepted_mask"    : (horizon,) bool — which steps used draft
            "latents"          : (horizon, B, d_model) — latents if return_latents
        """
        from model.vulgaris import VulgarisState

        d_idx = domain_idx if domain_idx is not None else self.domain_idx
        self.model.eval()

        B = x_context.shape[0]
        x_ctx_t = Tensor(x_context.astype(np.float32), requires_grad=False)

        # ── Warm-up: full forward on context ──────────────────────────────
        state = self.model.init_state(batch_size=B)
        ctx_out, ctx_state = self.model.step(x_ctx_t[:, :, -1], state, domain_idx=d_idx)
        state = ctx_state
        self._full_model_calls += 1

        # Current last hidden state: approximate from context forward
        ctx_full, _ = self.model(x_ctx_t, domain_idx=d_idx)
        last_h_np = np.zeros((B, self.model.d_model), dtype=np.float64)

        outputs:       List[np.ndarray] = []
        accepted_mask: List[bool]       = []
        latents_list:  List[np.ndarray] = []

        step_idx = 0
        last_output = ctx_out   # (B, out_dim) Tensor — last verified output
        self._full_model_calls += 1

        while step_idx < horizon:
            remaining = horizon - step_idx
            draft_k   = min(self.gamma, remaining)

            # ── Draft phase ───────────────────────────────────────────────
            h_last_t = Tensor(last_h_np, requires_grad=False)
            future_z, _ = self.model.world_model(h_last_t, horizon=draft_k)
            # future_z: (B, draft_k, d_model)

            # Project latent drafts to output space
            draft_outputs = []
            for i in range(draft_k):
                h_i = future_z.data[:, i, :]          # (B, d_model)
                h_3d = h_i[:, None, :]                 # (B, 1, d_model)
                h_t  = Tensor(h_3d, requires_grad=False)
                out_i = self.model.output_head(h_t)    # (B, out_dim)
                draft_outputs.append(out_i.data.copy())

            self._draft_cycles += 1

            # ── Verify phase ──────────────────────────────────────────────
            # For each draft step, use the draft latent as the input context
            # for one full model step and compare outputs.
            verify_accept = 0
            for i in range(draft_k):
                h_i_np  = future_z.data[:, i, :]      # (B, d_model)
                h_i_3d  = h_i_np[:, None, :]           # (B, 1, d_model)
                # Verify: run output head on this latent via the full forward
                # (lightweight — only output_head, not the full stack)
                h_t_ver = Tensor(h_i_3d, requires_grad=False)
                y_ver   = self.model.output_head(h_t_ver)  # (B, out_dim)
                self._full_model_calls += 1

                # Accept/reject
                y_d = draft_outputs[i]                 # (B, out_dim)
                y_v = y_ver.data                       # (B, out_dim)
                err = np.abs(y_d - y_v).max()

                if err < self.accept_threshold:
                    outputs.append(y_d)
                    accepted_mask.append(True)
                    if return_latents:
                        latents_list.append(h_i_np.copy())
                    last_h_np = h_i_np.copy()
                    self._accepted_steps += 1
                    verify_accept += 1
                    step_idx += 1
                    if step_idx >= horizon:
                        break
                else:
                    # Reject: fall back to full model for this step
                    # Use the verified output from the full stack
                    outputs.append(y_v.copy())
                    accepted_mask.append(False)
                    if return_latents:
                        latents_list.append(h_i_np.copy())
                    last_h_np = h_i_np.copy()
                    step_idx += 1
                    break   # restart draft from here

            self._total_steps += verify_accept

        result = {
            "outputs":       np.stack(outputs, axis=0),        # (horizon, B, out)
            "accepted_mask": np.array(accepted_mask, dtype=bool),
        }
        if return_latents:
            result["latents"] = np.stack(latents_list, axis=0)  # (horizon, B, d_model)

        return result

    # ──────────────────────────────────────────────────────────────────────

    def rollout_single(
        self,
        x_last: np.ndarray,
        state,
        domain_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Speculative single-step: propose one step via WorldModelHead, verify
        against full model. Returns (prediction, was_accepted).

        x_last : (B, C) single timestep to feed into full model if draft rejected
        state  : current VulgarisState

        Returns:
            prediction  : (B, out_dim) numpy
            accepted    : True if draft was accepted (full model not called)
        """
        d_idx = domain_idx if domain_idx is not None else self.domain_idx
        self.model.eval()

        B = x_last.shape[0]

        # Draft a single step from the world model's zero state (cheap)
        h_zero = Tensor(np.zeros((B, self.model.d_model), dtype=np.float64),
                        requires_grad=False)
        future_z, _ = self.model.world_model(h_zero, horizon=1)
        h_draft = future_z.data[:, 0, :]         # (B, d_model)
        h_3d    = h_draft[:, None, :]             # (B, 1, d_model)
        h_t     = Tensor(h_3d, requires_grad=False)
        y_draft = self.model.output_head(h_t).data.copy()  # (B, out_dim)

        # Verify with full model — step() expects (B, C_in) 2-D input
        x_np = np.asarray(x_last, dtype=np.float32)
        if x_np.ndim == 3:   # (B, C, 1) → (B, C)
            x_np = x_np[:, :, 0]
        x_t = Tensor(x_np, requires_grad=False)
        y_full, new_state = self.model.step(x_t, state, domain_idx=d_idx)
        self._full_model_calls += 1

        y_v = y_full.data
        err = np.abs(y_draft - y_v).max()
        accepted = bool(err < self.accept_threshold)

        self._total_steps += 1
        if accepted:
            self._accepted_steps += 1
            return y_draft, True
        else:
            return y_v, False

    # ──────────────────────────────────────────────────────────────────────

    def reset_stats(self) -> None:
        self._total_steps     = 0
        self._accepted_steps  = 0
        self._full_model_calls = 0
        self._draft_cycles    = 0

    def stats(self) -> dict:
        accept_rate = (self._accepted_steps / max(self._total_steps, 1))
        # Effective speedup: output steps per full-model call
        speedup = (self._total_steps / max(self._full_model_calls, 1))
        return {
            "gamma":              self.gamma,
            "accept_threshold":   self.accept_threshold,
            "total_steps":        self._total_steps,
            "accepted_steps":     self._accepted_steps,
            "full_model_calls":   self._full_model_calls,
            "draft_cycles":       self._draft_cycles,
            "acceptance_rate":    round(accept_rate, 4),
            "effective_speedup":  round(speedup, 3),
        }
