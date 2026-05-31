from __future__ import annotations

import numpy as np


class EpisodicMemory:
    """
    Fixed-capacity ring buffer of (key, value) pairs with cosine-similarity retrieval.

    Used by InContextLearning to persist good reference episodes across inference
    sessions — enabling zero-shot adaptation without retraining.

    Parameters
    ----------
    capacity : maximum number of episodes to store (FIFO eviction)
    d_model  : embedding dimension for keys and values
    """

    def __init__(self, capacity: int = 512, d_model: int = 256):
        self.capacity  = capacity
        self.d_model   = d_model
        self._keys     = np.zeros((capacity, d_model), dtype=np.float32)
        self._values   = np.zeros((capacity, d_model), dtype=np.float32)
        self._metadata: list[dict] = [{}] * capacity
        self._write_ptr = 0
        self._filled    = 0

    # ------------------------------------------------------------------

    def store(
        self,
        key: np.ndarray,
        value: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        """
        Store one (key, value) pair.

        key      : (d_model,) — context embedding (e.g. pooled z_ref from ContextEncoder)
        value    : (d_model,) — outcome embedding (e.g. projected y_ref label)
        metadata : optional dict of arbitrary scalars (e.g. timestamp, anomaly_score)
        """
        idx = self._write_ptr % self.capacity
        self._keys[idx]     = key.reshape(-1)[: self.d_model].astype(np.float32)
        self._values[idx]   = value.reshape(-1)[: self.d_model].astype(np.float32)
        self._metadata[idx] = metadata or {}
        self._write_ptr    += 1
        self._filled        = min(self._filled + 1, self.capacity)

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """
        Return the top-k most similar stored episodes.

        query : (d_model,) — query embedding

        Returns
        -------
        keys      : (k, d_model) float32
        values    : (k, d_model) float32
        scores    : (k,)         float32 — cosine similarities in descending order
        metadata  : list[dict]   length k
        """
        if self._filled == 0:
            empty = np.zeros((0, self.d_model), dtype=np.float32)
            return empty, empty, np.zeros(0, dtype=np.float32), []

        k = min(top_k, self._filled)
        keys = self._keys[: self._filled]          # (N, d_model)
        q    = query.reshape(1, -1).astype(np.float32)

        q_norm = q / (np.linalg.norm(q) + 1e-8)
        k_norm = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8)
        scores = (k_norm @ q_norm.T).reshape(-1)   # (N,) cosine similarities

        top_idx = np.argsort(scores)[::-1][:k]
        meta    = [self._metadata[i] for i in top_idx]
        return (
            keys[top_idx],
            self._values[: self._filled][top_idx],
            scores[top_idx],
            meta,
        )

    def clear(self) -> None:
        """Reset the buffer — call between unrelated deployments."""
        self._write_ptr = 0
        self._filled    = 0
        self._metadata  = [{}] * self.capacity

    def __len__(self) -> int:
        return self._filled

    def __repr__(self) -> str:
        return (f"EpisodicMemory(capacity={self.capacity}, d_model={self.d_model}, "
                f"stored={self._filled})")
