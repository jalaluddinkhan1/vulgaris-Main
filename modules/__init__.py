"""
VULGARIS model building blocks.

Each module is independently usable and composable.
"""

from __future__ import annotations

from .ase    import AdaptiveSignalEmbedding
from .sssr   import SelectiveSSR
from .crg    import CausalRoutingGraph
from .hmb    import HierarchicalMemoryBank
from .shcal  import SHCAL
from .dah    import DomainAdaptiveHypernetwork
from .ese    import ExplainabilityEngine
from .cmla   import CrossModalLatentAlignment
from .htd    import HierarchicalTimescaleDecomposition
from .safety import SafetyPolicyHead
from .multitask_head import MultiTaskHead

__all__ = [
    "AdaptiveSignalEmbedding",
    "SelectiveSSR",
    "CausalRoutingGraph",
    "HierarchicalMemoryBank",
    "SHCAL",
    "DomainAdaptiveHypernetwork",
    "ExplainabilityEngine",
    "CrossModalLatentAlignment",
    "HierarchicalTimescaleDecomposition",
    "SafetyPolicyHead",
    "MultiTaskHead",
]
