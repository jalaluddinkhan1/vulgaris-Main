"""
Active Learning — pool-based uncertainty sampling for VULGARIS.

Selects the most informative unlabeled samples from a pool so that
labelling effort is concentrated where the model is most uncertain.

Acquisition strategies
----------------------
"uncertainty"  — variance of output vector (works for regression and classification)
"entropy"      — Shannon entropy of softmax output (classification)
"margin"       — difference between top-2 class probabilities (classification)
"random"       — baseline uniform random selection

Usage
-----
    learner = ActiveLearner(model, strategy="uncertainty")
    scores  = learner.score(x_pool, domain_idx=0)       # (N,) uncertainty scores
    indices = learner.query(x_pool, n_query=10)          # top-10 indices
    learner.mark_labeled(indices)                        # track labeled budget

    # Integration with TrainingPipeline:
    pipeline.active_step(x_pool, x_labeled, y_labeled, n_query=10)
from __future__ import annotations

"""


import numpy as np
from typing import List, Optional


class ActiveLearner:
    """
    Pool-based active learner using a VULGARIS model's prediction uncertainty.

    Parameters
    ----------
    model       : Vulgaris instance
    strategy    : acquisition function — "uncertainty", "entropy", "margin", "random"
    n_mc        : number of MC samples for dropout-based uncertainty (set >1 to enable)
    batch_size  : how many pool samples to score at once (memory control)
    """

    def __init__(
        self,
        model,
        strategy: str = "uncertainty",
        n_mc: int = 1,
        batch_size: int = 64,
    ):
        self.model      = model
        self.strategy   = strategy
        self.n_mc       = n_mc
        self.batch_size = batch_size

        self._labeled_mask: Optional[np.ndarray] = None   # bool (N,) — tracked externally
        self._query_history: List[np.ndarray] = []

    # ──────────────────────────────────────────────────────────────────────

    def score(
        self,
        x_pool: np.ndarray,
        domain_idx: int = 0,
    ) -> np.ndarray:
        """
        Compute acquisition scores for every sample in the pool.

        x_pool : (N, C, T) numpy array — unlabeled pool
        Returns (N,) float array — higher score = more informative.
        """
        from engine.tensor import Tensor

        N = x_pool.shape[0]
        scores = np.zeros(N, dtype=np.float64)

        self.model.eval()

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            x_batch = x_pool[start:end].astype(np.float32)
            x_t = Tensor(x_batch, requires_grad=False)

            if self.n_mc > 1:
                # MC uncertainty: multiple forward passes (model must support dropout)
                preds = []
                for _ in range(self.n_mc):
                    out, _ = self.model(x_t, domain_idx=domain_idx)
                    preds.append(out.data.copy())
                pred_stack = np.stack(preds, axis=0)      # (n_mc, B, out_dim)
                batch_scores = pred_stack.var(axis=0).mean(axis=-1)  # (B,)
            else:
                out, _ = self.model(x_t, domain_idx=domain_idx)
                preds_np = out.data                       # (B, out_dim)
                batch_scores = self._acquisition(preds_np)  # (B,)

            scores[start:end] = batch_scores

        return scores

    def _acquisition(self, preds: np.ndarray) -> np.ndarray:
        """Apply the selected acquisition function to a (B, out_dim) prediction."""
        if self.strategy == "uncertainty":
            # Variance across output dimensions — model spread
            return preds.var(axis=-1)

        elif self.strategy == "entropy":
            # Treat output as class probabilities (softmax already applied)
            p = np.clip(preds, 1e-10, 1.0)
            p = p / (p.sum(axis=-1, keepdims=True) + 1e-10)
            return -(p * np.log(p)).sum(axis=-1)

        elif self.strategy == "margin":
            # Margin = difference between top-2 probs; lower margin = more uncertain
            if preds.shape[-1] < 2:
                return preds.var(axis=-1)
            sorted_p = np.sort(preds, axis=-1)
            margin = sorted_p[:, -1] - sorted_p[:, -2]
            return -margin   # negate: smaller margin → higher score

        elif self.strategy == "random":
            return np.random.rand(preds.shape[0])

        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'. "
                             "Choose from: uncertainty, entropy, margin, random")

    def query(
        self,
        x_pool: np.ndarray,
        n_query: int,
        domain_idx: int = 0,
        exclude_labeled: bool = True,
    ) -> np.ndarray:
        """
        Return indices of the n_query most uncertain samples.

        x_pool         : (N, C, T) pool
        n_query        : how many samples to select
        exclude_labeled: skip already-labeled indices (if mark_labeled was called)

        Returns: (n_query,) int array of selected indices.
        """
        scores = self.score(x_pool, domain_idx=domain_idx)

        if exclude_labeled and self._labeled_mask is not None:
            n = len(scores)
            mask = self._labeled_mask[:n] if len(self._labeled_mask) >= n else \
                   np.pad(self._labeled_mask, (0, n - len(self._labeled_mask)))
            scores[mask] = -np.inf   # exclude labeled samples

        n_query = min(n_query, int(np.isfinite(scores).sum()))
        indices = np.argsort(scores)[::-1][:n_query]
        return indices.astype(np.int64)

    def mark_labeled(self, indices: np.ndarray, pool_size: Optional[int] = None) -> None:
        """
        Record that the given pool indices have been labeled and should be
        excluded from future queries.
        """
        if pool_size is None:
            pool_size = int(indices.max()) + 1 if len(indices) > 0 else 0

        if self._labeled_mask is None or len(self._labeled_mask) < pool_size:
            new_mask = np.zeros(max(pool_size, len(self._labeled_mask or [])),
                                dtype=bool)
            if self._labeled_mask is not None:
                new_mask[:len(self._labeled_mask)] = self._labeled_mask
            self._labeled_mask = new_mask

        self._labeled_mask[indices] = True
        self._query_history.append(indices.copy())

    def reset(self) -> None:
        """Clear the labeled mask and query history."""
        self._labeled_mask = None
        self._query_history = []

    def labeled_count(self) -> int:
        """Number of samples marked as labeled."""
        if self._labeled_mask is None:
            return 0
        return int(self._labeled_mask.sum())

    def stats(self) -> dict:
        return {
            "strategy":       self.strategy,
            "n_mc":           self.n_mc,
            "labeled_count":  self.labeled_count(),
            "query_rounds":   len(self._query_history),
        }
