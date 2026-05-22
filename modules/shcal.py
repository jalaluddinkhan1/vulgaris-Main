import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

from engine.tensor import Tensor, Parameter, zeros, ones, randn
from engine.module import Module
from engine.layers import Linear
from config import SHCALConfig


class SHCAL(Module):
    """Self-Healing & Continuous Adaptation Layer — online learning with stability-plasticity."""

    def __init__(self, monitored_modules: List[Module], config: SHCALConfig):
        super().__init__()
        self.config = config
        self.monitored_modules = monitored_modules

        # EWC state: keyed by "module_idx.weight" / "module_idx.bias"
        object.__setattr__(self, "fisher_diagonal", {})
        object.__setattr__(self, "reference_params", {})

        # Structural plasticity masks — populated lazily when first layer is seen
        object.__setattr__(self, "prune_masks", {})
        # Tracks how many consecutive steps each weight has been below threshold
        object.__setattr__(self, "prune_counter", {})
        # Tracks how many consecutive steps each gradient row has been above threshold
        object.__setattr__(self, "grow_counter", {})
        # Prune window: number of consecutive steps below threshold before pruning
        object.__setattr__(self, "prune_window", 10)

        # Conformal calibration
        object.__setattr__(self, "calibration_scores", deque(maxlen=1000))
        object.__setattr__(self, "trigger_adaptation", False)
        self.recalib_scale = Parameter(np.array([1.0]), name="recalib_scale")

        object.__setattr__(self, "step_counter", 0)
        object.__setattr__(self, "conformal_alpha", 0.1)  # target miscoverage

        # Initialize masks and counters for every Linear in monitored_modules
        for idx, mod in enumerate(monitored_modules):
            for attr_name in ("weight", "bias"):
                param = getattr(mod, attr_name, None)
                if param is not None and isinstance(param, Parameter):
                    key = f"{idx}.{attr_name}"
                    self.prune_masks[key] = np.ones(param.data.shape, dtype=bool)
                    self.prune_counter[key] = np.zeros(param.data.shape, dtype=np.int32)
                    self.grow_counter[key] = np.zeros(param.data.shape, dtype=np.int32)

    # ------------------------------------------------------------------
    # EWC
    # ------------------------------------------------------------------

    def compute_fisher(self, loss_fn, data_loader_iter, n_samples: int):
        """Estimate Fisher diagonal by squaring gradients over n_samples examples.

        loss_fn(x, y) must call backward() and leave grads on monitored params.
        After this call, reference_params is snapshotted from current weights.
        """
        # Zero-initialise accumulators
        accum: Dict[str, np.ndarray] = {}
        for idx, mod in enumerate(self.monitored_modules):
            for attr_name in ("weight", "bias"):
                param = getattr(mod, attr_name, None)
                if param is not None and isinstance(param, Parameter):
                    key = f"{idx}.{attr_name}"
                    accum[key] = np.zeros(param.data.shape, dtype=np.float64)

        count = 0
        while count < n_samples:
            try:
                x, y = next(data_loader_iter)
            except StopIteration:
                break

            # Clear gradients
            for mod in self.monitored_modules:
                for attr_name in ("weight", "bias"):
                    param = getattr(mod, attr_name, None)
                    if param is not None and isinstance(param, Parameter):
                        param.grad = None

            # Compute loss and backward
            loss = loss_fn(x, y)
            loss.backward()

            # Accumulate squared gradients
            for idx, mod in enumerate(self.monitored_modules):
                for attr_name in ("weight", "bias"):
                    param = getattr(mod, attr_name, None)
                    if param is not None and isinstance(param, Parameter):
                        key = f"{idx}.{attr_name}"
                        if param.grad is not None:
                            accum[key] += param.grad ** 2

            count += 1

        if count > 0:
            for key in accum:
                accum[key] /= count

        # Store Fisher diagonal
        object.__setattr__(self, "fisher_diagonal", accum)

        # Snapshot reference parameters θ*
        ref: Dict[str, np.ndarray] = {}
        for idx, mod in enumerate(self.monitored_modules):
            for attr_name in ("weight", "bias"):
                param = getattr(mod, attr_name, None)
                if param is not None and isinstance(param, Parameter):
                    ref[f"{idx}.{attr_name}"] = param.data.copy()
        object.__setattr__(self, "reference_params", ref)

    def ewc_loss(self, current_params: Dict[str, Tensor]) -> Tensor:
        """EWC penalty: (λ/2) * Σ_i F_i * (θ_i - θ*_i)^2"""
        if not self.fisher_diagonal:
            # Return zero scalar
            return Tensor(np.array([0.0]), requires_grad=False)

        penalty = Tensor(np.array([0.0]), requires_grad=False)
        lam = self.config.ewc_lambda

        for key, F_np in self.fisher_diagonal.items():
            if key not in current_params or key not in self.reference_params:
                continue
            theta = current_params[key]
            theta_star = Tensor(self.reference_params[key], requires_grad=False)
            F = Tensor(F_np, requires_grad=False)
            diff = theta - theta_star
            contrib = (F * diff * diff).sum()
            penalty = penalty + contrib

        return penalty * (lam / 2.0)

    # ------------------------------------------------------------------
    # Hebbian update with shadow mode validation
    # ------------------------------------------------------------------

    def hebbian_update(self, activations: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """Apply Δ W = η*(post⊗pre - decay*W) with shadow mode check.

        activations: {module_name: (pre, post)}
          pre : (batch, d_in)   — input to layer
          post: (batch, d_out)  — output of layer (before nonlinearity)
        module_name must be stringified index matching self.monitored_modules order.
        """
        eta = self.config.plasticity_rate
        decay = 1e-3  # weight decay term in Hebbian rule

        for idx, mod in enumerate(self.monitored_modules):
            key = str(idx)
            if key not in activations:
                continue
            if not isinstance(mod, Linear):
                continue

            pre_np, post_np = activations[key]
            # Average over batch dimension
            if pre_np.ndim > 1:
                pre_avg = pre_np.mean(axis=0)   # (d_in,)
                post_avg = post_np.mean(axis=0)  # (d_out,)
            else:
                pre_avg = pre_np
                post_avg = post_np

            param = mod.weight
            W = param.data  # (d_out, d_in)

            # Hebbian delta: ΔW_ij = η * (post_i * pre_j - decay * W_ij)
            dW = eta * (np.outer(post_avg, pre_avg) - decay * W)

            # Shadow mode validation
            mask_key = f"{idx}.weight"
            shadow_W = W.copy() + dW

            # Apply prune mask to shadow
            if mask_key in self.prune_masks:
                shadow_W *= self.prune_masks[mask_key]

            # Evaluate shadow loss proxy: Frobenius norm of shadow vs current
            # Use ||post - shadow_W @ pre|| vs ||post - W @ pre|| as proxy loss
            if pre_avg.ndim == 1 and post_avg.ndim == 1:
                current_residual = np.sum((post_avg - W @ pre_avg) ** 2)
                shadow_residual = np.sum((post_avg - shadow_W @ pre_avg) ** 2)
                loss_increased = shadow_residual > current_residual * 1.1
            else:
                loss_increased = False

            if not loss_increased:
                # Accept update
                param.data += dW
                if mask_key in self.prune_masks:
                    param.data *= self.prune_masks[mask_key]

    # ------------------------------------------------------------------
    # Conformal uncertainty calibration
    # ------------------------------------------------------------------

    def record_calibration(self, y_pred_np: np.ndarray, y_actual_np: np.ndarray,
                           sigma_np: np.ndarray):
        """Append normalised residual scores and update trigger_adaptation."""
        # score = ||y_pred - y_actual|| / σ_pred
        residual = np.linalg.norm(y_pred_np - y_actual_np, axis=-1)
        sigma = np.maximum(sigma_np, 1e-8)
        if sigma.ndim == 0:
            scores = residual / float(sigma)
            self.calibration_scores.append(float(scores))
        else:
            scores = residual / sigma
            for s in np.atleast_1d(scores).flatten():
                self.calibration_scores.append(float(s))

        # Conformal coverage check
        if len(self.calibration_scores) >= 20:
            alpha = self.conformal_alpha
            threshold = np.quantile(list(self.calibration_scores), 1.0 - alpha)
            covered = sum(1 for s in self.calibration_scores if s < threshold)
            coverage = covered / len(self.calibration_scores)
            object.__setattr__(self, "trigger_adaptation", coverage < (1.0 - alpha))

    # ------------------------------------------------------------------
    # Structural plasticity
    # ------------------------------------------------------------------

    def structural_update(self):
        """Prune and grow weights based on magnitude and gradient norms."""
        prune_thr = self.config.prune_threshold
        grow_thr = self.config.grow_threshold
        prune_win = self.prune_window

        for idx, mod in enumerate(self.monitored_modules):
            if not isinstance(mod, Linear):
                continue
            weight_key = f"{idx}.weight"
            param = mod.weight

            if weight_key not in self.prune_masks:
                continue

            W = param.data
            mask = self.prune_masks[weight_key]
            p_ctr = self.prune_counter[weight_key]
            g_ctr = self.grow_counter[weight_key]

            # --- Pruning ---
            below = (np.abs(W) < prune_thr) & mask
            p_ctr[below] += 1
            p_ctr[~below] = 0

            to_prune = (p_ctr >= prune_win) & mask
            mask[to_prune] = False
            W[to_prune] = 0.0
            p_ctr[to_prune] = 0

            # --- Growing ---
            if param.grad is not None:
                grad_norm_sq = param.grad ** 2
                above_grad = (grad_norm_sq > grow_thr) & (~mask)
                g_ctr[above_grad] += 1
                g_ctr[~above_grad & ~mask] = 0

                to_grow = (g_ctr >= 3) & (~mask)  # 3 consecutive steps
                mask[to_grow] = True
                # Reinitialise with small random value
                W[to_grow] = np.random.randn(int(to_grow.sum())) * 0.01
                g_ctr[to_grow] = 0

            self.prune_masks[weight_key] = mask
            self.prune_counter[weight_key] = p_ctr
            self.grow_counter[weight_key] = g_ctr

        object.__setattr__(self, "step_counter", self.step_counter + 1)

    # ------------------------------------------------------------------
    # State summary
    # ------------------------------------------------------------------

    def get_adaptation_state(self) -> dict:
        """Return summary of current adaptation state."""
        n_pruned = 0
        n_total = 0
        for mask in self.prune_masks.values():
            n_total += mask.size
            n_pruned += int((~mask).sum())

        n_grown = sum(int((ctr > 0).sum()) for ctr in self.grow_counter.values())

        ewc_norm = 0.0
        for key, F_np in self.fisher_diagonal.items():
            if key in self.reference_params:
                ewc_norm += float(np.sum(F_np))

        coverage = 0.0
        if len(self.calibration_scores) >= 2:
            alpha = self.conformal_alpha
            threshold = np.quantile(list(self.calibration_scores), 1.0 - alpha)
            covered = sum(1 for s in self.calibration_scores if s < threshold)
            coverage = covered / len(self.calibration_scores)

        return {
            "coverage": coverage,
            "n_pruned": n_pruned,
            "n_grown": n_grown,
            "n_total_weights": n_total,
            "sparsity": n_pruned / max(n_total, 1),
            "ewc_penalty_norm": ewc_norm,
            "trigger_adaptation": self.trigger_adaptation,
            "recalib_scale": float(self.recalib_scale.data[0]),
            "step_counter": self.step_counter,
            "n_calibration_scores": len(self.calibration_scores),
        }

    def forward(self, x: Tensor) -> Tensor:
        """Pass-through; SHCAL operates via side-effect methods."""
        return x
