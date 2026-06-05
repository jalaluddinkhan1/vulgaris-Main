"""
VULGARIS model building blocks.

Each module is independently usable and composable.
"""

from __future__ import annotations

# ── Signal embedding ───────────────────────────────────────────────────────
from .ase              import AdaptiveSignalEmbedding
from .revin            import RevIN

# ── State-space recurrence ─────────────────────────────────────────────────
from .sssr             import SelectiveSSR, SSSRHead

# ── Hierarchical timescales ────────────────────────────────────────────────
from .htd              import HierarchicalTimescaleDecomposition, HTDLevel

# ── Causal routing ─────────────────────────────────────────────────────────
from .crg              import CausalRoutingGraph

# ── Memory ─────────────────────────────────────────────────────────────────
from .hmb              import HierarchicalMemoryBank, MemoryVAE
from .episodic_memory  import EpisodicMemory
from .causal_memory    import CausalMemory

# ── Continual adaptation ───────────────────────────────────────────────────
from .shcal            import SHCAL

# ── Domain adaptation ──────────────────────────────────────────────────────
from .dah              import DomainAdaptiveHypernetwork, AdapterLayer

# ── Explainability ─────────────────────────────────────────────────────────
from .ese              import ExplainabilityEngine, CARTExtractor, DecisionNode

# ── Multi-modal alignment ──────────────────────────────────────────────────
from .cmla             import CrossModalLatentAlignment, ModalityEncoder

# ── Safety ─────────────────────────────────────────────────────────────────
from .safety           import SafetyPolicyHead, CBFLayer, SpectralNormLinear

# ── Output heads ───────────────────────────────────────────────────────────
from .multitask_head   import MultiTaskHead

# ── In-context learning ────────────────────────────────────────────────────
from .icl              import InContextLearning, InContextAdapter, ContextEncoder

# ── Regime mixture ─────────────────────────────────────────────────────────
from .rmc              import RegimeMixtureCore

# ── Ontology ───────────────────────────────────────────────────────────────
from .ontology_embedding import OntologyEmbedding, OntologyRegistry

# ── Rule engine ────────────────────────────────────────────────────────────
from .rule_engine      import (Rule, RuleRegistry, RuleEncoder,
                               RuleConditionLoss, RuleDistiller,
                               RuleLifecycleManager)

# ── Test-time training ─────────────────────────────────────────────────────
from .ttt              import TestTimeTrainer, TTTConfig
from .event_encoder    import EventEncoder
from .weibull_head     import WeibullHead


__all__ = [
    # Signal embedding
    "AdaptiveSignalEmbedding", "RevIN",
    # State-space recurrence
    "SelectiveSSR", "SSSRHead",
    # Hierarchical timescales
    "HierarchicalTimescaleDecomposition", "HTDLevel",
    # Causal routing
    "CausalRoutingGraph",
    # Memory
    "HierarchicalMemoryBank", "MemoryVAE", "EpisodicMemory", "CausalMemory",
    # Continual adaptation
    "SHCAL",
    # Domain adaptation
    "DomainAdaptiveHypernetwork", "AdapterLayer",
    # Explainability
    "ExplainabilityEngine", "CARTExtractor", "DecisionNode",
    # Multi-modal alignment
    "CrossModalLatentAlignment", "ModalityEncoder",
    # Safety
    "SafetyPolicyHead", "CBFLayer", "SpectralNormLinear",
    # Output heads
    "MultiTaskHead",
    # In-context learning
    "InContextLearning", "InContextAdapter", "ContextEncoder",
    # Regime mixture
    "RegimeMixtureCore",
    # Ontology
    "OntologyEmbedding", "OntologyRegistry",
    # Rule engine
    "Rule", "RuleRegistry", "RuleEncoder",
    "RuleConditionLoss", "RuleDistiller", "RuleLifecycleManager",
    # Test-time training
    "TestTimeTrainer", "TTTConfig",
    # Universal domain heads
    "EventEncoder", "WeibullHead",
]
