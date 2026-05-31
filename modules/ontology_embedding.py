"""
Ontology Embeddings for VULGARIS — domain-semantic context injection.

Maps industrial sensor ontology terms (e.g. "temperature", "vibration",
"pressure") to a dense (1, meta_dim) context vector that is injected
additively into the DAH hypernetwork domain embedding z, alongside the
optional RuleEncoder vector.

Result: adapters become ontology-aware — the same domain_idx with different
ontology terms (different physical quantities being measured) produces
meaningfully different adapter weights.

Ontology term vocabulary
------------------------
A built-in vocabulary covers ~80 industrial sensor/process terms, organised
into semantic groups.  Unknown terms are mapped to a trainable <UNK> token.

Usage
-----
    registry = OntologyRegistry()
    registry.register(domain_idx=0, terms=["temperature", "pressure", "vibration"])

    embed = OntologyEmbedding(meta_dim=128)
    z_onto = embed.encode(registry, domain_idx=0)   # (1, 128) Tensor

    # Wire into DAH:
    dah.attach_ontology_embedding(embed, registry)
from __future__ import annotations

"""


import numpy as np
from typing import Dict

from engine.tensor import Tensor, Parameter
from engine.module import Module
from engine.layers import Linear


# ──────────────────────────────────────────────────────────────────────────────
# Built-in vocabulary
# ──────────────────────────────────────────────────────────────────────────────

# Canonical industrial sensor / process terms, grouped by semantic cluster.
# The cluster index encodes a coarse semantic prior (same cluster → similar
# initial embeddings via Xavier init scaled by 1/n_clusters).
_ONTOLOGY_VOCAB: Dict[str, int] = {
    # Thermal (cluster 0)
    "temperature": 0, "temp": 0, "heat": 0, "thermal": 0,
    "exhaust_temp": 0, "coolant_temp": 0, "ambient_temp": 0,

    # Mechanical vibration (cluster 1)
    "vibration": 1, "acceleration": 1, "shock": 1, "rms_vibration": 1,
    "bearing_vibration": 1, "imbalance": 1,

    # Pressure (cluster 2)
    "pressure": 2, "vacuum": 2, "differential_pressure": 2,
    "inlet_pressure": 2, "outlet_pressure": 2, "gauge_pressure": 2,

    # Flow (cluster 3)
    "flow": 3, "flow_rate": 3, "mass_flow": 3, "volume_flow": 3,
    "coolant_flow": 3, "fuel_flow": 3,

    # Electrical (cluster 4)
    "voltage": 4, "current": 4, "power": 4, "frequency": 4,
    "harmonics": 4, "thd": 4, "phase_angle": 4, "reactive_power": 4,

    # Speed / rotation (cluster 5)
    "speed": 5, "rpm": 5, "angular_velocity": 5, "torque": 5,
    "shaft_speed": 5, "rotor_speed": 5,

    # Position / geometry (cluster 6)
    "position": 6, "displacement": 6, "level": 6, "thickness": 6,
    "gap": 6, "alignment": 6,

    # Chemical / quality (cluster 7)
    "ph": 7, "conductivity": 7, "turbidity": 7, "concentration": 7,
    "viscosity": 7, "density": 7, "humidity": 7,

    # Network / telecom (cluster 8)
    "latency": 8, "throughput": 8, "packet_loss": 8, "snr": 8,
    "rssi": 8, "ber": 8, "jitter": 8,

    # Semiconductor / fab (cluster 9)
    "etch_rate": 9, "deposition_rate": 9, "film_thickness": 9,
    "uniformity": 9, "particle_count": 9, "overlay": 9,

    # Control / action (cluster 10)
    "setpoint": 10, "valve_position": 10, "actuator": 10,
    "pid_output": 10, "control_action": 10,

    # Regime / state (cluster 11)
    "regime": 11, "mode": 11, "state": 11, "phase": 11,
    "operating_mode": 11, "fault_code": 11,
}

_N_CLUSTERS = 12
_UNK_TOKEN  = "<unk>"


# ──────────────────────────────────────────────────────────────────────────────
# Ontology Registry
# ──────────────────────────────────────────────────────────────────────────────

class OntologyRegistry:
    """
    Per-domain ontology term store.

    Usage
    -----
        registry = OntologyRegistry()
        registry.register(0, ["temperature", "vibration", "pressure"])
        terms = registry.get(0)
        clusters = registry.cluster_ids(0)
    """

    def __init__(self):
        self._store: dict[int, list[str]] = {}

    def register(self, domain_idx: int, terms: list[str]) -> None:
        """Register (replace) ontology terms for a domain."""
        self._store[domain_idx] = [t.lower().strip() for t in terms]

    def get(self, domain_idx: int) -> list[str]:
        return self._store.get(domain_idx, [])

    def has_terms(self, domain_idx: int) -> bool:
        return bool(self._store.get(domain_idx))

    def cluster_ids(self, domain_idx: int) -> list[int]:
        """Return cluster index for each registered term (UNK → -1)."""
        return [_ONTOLOGY_VOCAB.get(t, -1) for t in self.get(domain_idx)]

    def all_domains(self) -> list[int]:
        return list(self._store.keys())

    def encode_domain_vec(self, embed: "OntologyEmbedding",
                          domain_idx: int) -> "Tensor":
        """Convenience: forward through embed for this domain's terms."""
        return embed.encode(self, domain_idx)

    def vocab_coverage(self, domain_idx: int) -> dict:
        terms = self.get(domain_idx)
        known = [t for t in terms if t in _ONTOLOGY_VOCAB]
        unknown = [t for t in terms if t not in _ONTOLOGY_VOCAB]
        return {"known": known, "unknown": unknown,
                "coverage": len(known) / max(len(terms), 1)}


# ──────────────────────────────────────────────────────────────────────────────
# Ontology Embedding Module
# ──────────────────────────────────────────────────────────────────────────────

class OntologyEmbedding(Module):
    """
    Encodes a domain's ontology term set into a (1, meta_dim) context vector.

    Architecture
    ------------
    1. Each term → cluster_id → look up cluster embedding (n_clusters, meta_dim)
    2. Unknown terms (cluster_id = -1) → learnable <UNK> embedding
    3. Mean-pool over all terms → (1, meta_dim)
    4. Output linear projection: (meta_dim,) → (meta_dim,)

    The cluster embeddings are initialised with distinct Xavier-scaled random
    values per cluster so different sensor types start with distinguishable
    representations.

    Parameters
    ----------
    meta_dim : must match DAHConfig.meta_dim
    """

    def __init__(self, meta_dim: int, n_clusters: int = _N_CLUSTERS):
        super().__init__()
        self.meta_dim   = meta_dim
        self.n_clusters = n_clusters

        # Cluster embedding table: (n_clusters + 1, meta_dim)
        # Last row = UNK embedding
        n_rows = n_clusters + 1
        scale  = np.sqrt(2.0 / (n_rows + meta_dim))
        init   = np.random.randn(n_rows, meta_dim).astype(np.float64) * scale
        self.cluster_embed = Parameter(init, name="onto_cluster_embed")

        # Output projection
        self.out_proj = Linear(meta_dim, meta_dim)

    # ──────────────────────────────────────────────────────────────────────

    def _cluster_ids_for_terms(self, terms: list[str]) -> list[int]:
        ids = []
        for t in terms:
            cid = _ONTOLOGY_VOCAB.get(t.lower().strip(), -1)
            ids.append(cid if cid >= 0 else self.n_clusters)   # UNK row
        return ids

    def forward(self, terms: list[str]) -> Tensor:
        """
        terms: list of ontology term strings (from OntologyRegistry.get())
        Returns (1, meta_dim) Tensor.
        """
        if not terms:
            # No terms → zero vector (no gradient needed)
            return Tensor(np.zeros((1, self.meta_dim), dtype=np.float64),
                          requires_grad=False)

        cluster_ids = self._cluster_ids_for_terms(terms)
        # Look up embeddings: (n_terms, meta_dim)
        emb_np = self.cluster_embed.data[cluster_ids]   # (T, meta_dim)

        # Wrap with grad connection to cluster_embed
        emb = Tensor(
            emb_np,
            requires_grad=self.cluster_embed.requires_grad,
            _children=(self.cluster_embed,),
            _op="onto_lookup"
        )
        _ce = self.cluster_embed
        _ids = cluster_ids

        def _lookup_back():
            if _ce.requires_grad and emb.grad is not None:
                if _ce.grad is None:
                    _ce.grad = np.zeros_like(_ce.data)
                for local_i, gid in enumerate(_ids):
                    _ce.grad[gid] += emb.grad[local_i]

        emb._backward = _lookup_back

        # Mean pool: (1, meta_dim)
        n = len(terms)
        pooled_np = emb_np.mean(axis=0, keepdims=True)   # (1, meta_dim)
        pooled = Tensor(
            pooled_np,
            requires_grad=emb.requires_grad,
            _children=(emb,),
            _op="onto_pool"
        )
        _emb = emb

        def _pool_back():
            if _emb.requires_grad and pooled.grad is not None:
                contrib = np.repeat(pooled.grad, n, axis=0) / n  # (T, meta_dim)
                _emb.grad = (_emb.grad + contrib
                             if _emb.grad is not None else contrib)

        pooled._backward = _pool_back
        return self.out_proj(pooled)   # (1, meta_dim)

    def encode(self, registry: OntologyRegistry, domain_idx: int) -> Tensor:
        """Convenience wrapper: look up terms and forward."""
        return self.forward(registry.get(domain_idx))

    def term_similarity(self, term_a: str, term_b: str) -> float:
        """
        Cosine similarity between two term embeddings (uses current weights).
        Useful for inspecting learned semantic structure.
        """
        id_a = _ONTOLOGY_VOCAB.get(term_a.lower(), self.n_clusters)
        id_b = _ONTOLOGY_VOCAB.get(term_b.lower(), self.n_clusters)
        ea = self.cluster_embed.data[id_a]
        eb = self.cluster_embed.data[id_b]
        return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))

    def nearest_terms(self, query_term: str, top_k: int = 5) -> list[tuple]:
        """
        Return top-k nearest vocabulary terms to `query_term` by cosine similarity.
        Returns list of (term, similarity) sorted descending.
        """
        q_id  = _ONTOLOGY_VOCAB.get(query_term.lower(), self.n_clusters)
        q_emb = self.cluster_embed.data[q_id]
        q_norm = np.linalg.norm(q_emb) + 1e-8

        scores = []
        for term, cid in _ONTOLOGY_VOCAB.items():
            if term == query_term:
                continue
            e = self.cluster_embed.data[cid]
            sim = float(np.dot(q_emb, e) / (q_norm * (np.linalg.norm(e) + 1e-8)))
            scores.append((term, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
