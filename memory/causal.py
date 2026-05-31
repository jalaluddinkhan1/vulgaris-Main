from __future__ import annotations

from collections import defaultdict


class CausalMemory:
    """
    Append-only store of (cause_node, effect_node, confidence) triples.

    Written by CausalRoutingGraph during online structure discovery.
    Read by InContextLearning and RegimeMixtureCore for context-aware reasoning —
    e.g. "sensor 3 causes sensor 7 with confidence 0.82" informs which
    upstream nodes to watch when sensor 7 anomalies fire.

    Parameters
    ----------
    capacity : maximum triples to retain (FIFO eviction after capacity)
    """

    def __init__(self, capacity: int = 4096):
        self.capacity   = capacity
        self._triples:  list[dict] = []
        self._timestamp = 0

        # Inverted indices for O(1) lookup by cause or effect node
        self._by_cause:  dict[int, list[int]] = defaultdict(list)   # cause  → [triple_idx]
        self._by_effect: dict[int, list[int]] = defaultdict(list)   # effect → [triple_idx]

    # ------------------------------------------------------------------

    def record(self, cause: int, effect: int, confidence: float) -> None:
        """
        Append one causal triple.

        cause      : source node index
        effect     : downstream node index
        confidence : strength ∈ [0, 1] (e.g. |Granger correlation|)
        """
        if len(self._triples) >= self.capacity:
            self._evict()

        idx = len(self._triples)
        self._triples.append({
            "cause":      int(cause),
            "effect":     int(effect),
            "confidence": float(confidence),
            "t":          self._timestamp,
        })
        self._by_cause[int(cause)].append(idx)
        self._by_effect[int(effect)].append(idx)
        self._timestamp += 1

    def _evict(self) -> None:
        """Remove the oldest triple and rebuild inverted indices."""
        self._triples.pop(0)
        # Rebuild indices — O(N) but called infrequently (every `capacity` records)
        self._by_cause  = defaultdict(list)
        self._by_effect = defaultdict(list)
        for i, t in enumerate(self._triples):
            self._by_cause[t["cause"]].append(i)
            self._by_effect[t["effect"]].append(i)

    # ------------------------------------------------------------------

    def query_effects(
        self, cause: int, min_confidence: float = 0.1
    ) -> list[tuple[int, float]]:
        """
        Return [(effect_node, confidence)] for all known downstream effects
        of `cause`, sorted by confidence descending.
        """
        idxs    = self._by_cause.get(int(cause), [])
        results = [
            (self._triples[i]["effect"], self._triples[i]["confidence"])
            for i in idxs
            if i < len(self._triples) and self._triples[i]["confidence"] >= min_confidence
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def query_causes(
        self, effect: int, min_confidence: float = 0.1
    ) -> list[tuple[int, float]]:
        """
        Return [(cause_node, confidence)] for all known upstream causes
        of `effect`, sorted by confidence descending.
        """
        idxs    = self._by_effect.get(int(effect), [])
        results = [
            (self._triples[i]["cause"], self._triples[i]["confidence"])
            for i in idxs
            if i < len(self._triples) and self._triples[i]["confidence"] >= min_confidence
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_recent(self, n: int = 100) -> list[dict]:
        """Return the n most recent triples."""
        return self._triples[-n:]

    def strongest_edges(self, top_k: int = 20) -> list[dict]:
        """Return top-k triples by confidence."""
        return sorted(self._triples, key=lambda t: t["confidence"], reverse=True)[:top_k]

    def clear(self) -> None:
        self._triples   = []
        self._by_cause  = defaultdict(list)
        self._by_effect = defaultdict(list)
        self._timestamp = 0

    def __len__(self) -> int:
        return len(self._triples)

    def __repr__(self) -> str:
        return f"CausalMemory(capacity={self.capacity}, stored={len(self._triples)})"
