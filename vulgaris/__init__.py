"""VULGARIS — Foundational Industrial AI Model for streaming telemetry, IoT, and time-series.

Quick start
-----------
    import vulgaris

    model  = vulgaris.Vulgaris(vulgaris.ModelConfig(input_dim=32, output_dim=1))
    engine = vulgaris.StreamingInference(model, batch_size=1)

    result = engine.step(sensor_reading)   # (1, 32) numpy array
    # → {'prediction': ..., 'uncertainty': ..., 'latency_ms': ..., 'skipped': ...}

Install extras
--------------
    pip install vulgaris            # core only (numpy + scipy)
    pip install "vulgaris[serve]"   # + FastAPI REST server
    pip install "vulgaris[train]"   # + tqdm + rich progress
    pip install "vulgaris[all]"     # everything
"""

from __future__ import annotations

__version__ = "0.4.0"
__author__  = "VULGARIS Contributors"
__license__ = "Apache-2.0"

# ── Config ────────────────────────────────────────────────────────────────────
from vulgaris.config import (
    ModelConfig, ASEConfig, SSSRConfig, CRGConfig, HMBConfig,
    SHCALConfig, DAHConfig, ESEConfig, HTDConfig, SafetyConfig,
    TrainingConfig, MultiTaskConfig,
)

# ── Core model ────────────────────────────────────────────────────────────────
from model.vulgaris import Vulgaris, VulgarisState, OutputHead

# ── Autograd engine ───────────────────────────────────────────────────────────
from engine.tensor import Tensor, Parameter, zeros, ones, randn, rand, cat, stack
from engine.module import Module
from engine.layers import Linear, LayerNorm, RMSNorm

# ── Training ──────────────────────────────────────────────────────────────────
from training.pipeline    import TrainingPipeline, TimeSeriesAugment, CurriculumSchedule
from training.loss        import VulgarisLoss
from training.optimizer   import SpectralAdamW, CosineSchedule, MuonOptimizer
from training.conformal   import NonStationaryConformal
from training.distillation import DistillationLoss, DistillationTrainer
from training.active_learning import ActiveLearner

# ── Inference ─────────────────────────────────────────────────────────────────
from inference.streaming    import StreamingInference
from inference.speculative  import SpeculativeRollout
from inference.event_buffer import EventBuffer

# ── Serving ───────────────────────────────────────────────────────────────────
from serve.degradation import DegradationController, DeploymentMode, CanaryController
from serve.audit       import AuditLogger
from serve.migration   import migrate_checkpoint, HotSwapAdapter, get_checkpoint_version

# ── Modules (importable individually) ────────────────────────────────────────
from modules.ase              import AdaptiveSignalEmbedding
from modules.sssr             import SelectiveSSR
from modules.htd              import HierarchicalTimescaleDecomposition
from modules.crg              import CausalRoutingGraph
from modules.hmb              import HierarchicalMemoryBank
from modules.dah              import DomainAdaptiveHypernetwork
from modules.ese              import ExplainabilityEngine
from modules.safety           import SafetyPolicyHead
from modules.shcal            import SHCAL
from modules.cmla             import CrossModalLatentAlignment
from modules.icl              import InContextLearning
from modules.rmc              import RegimeMixtureCore
from modules.ontology_embedding import OntologyEmbedding, OntologyRegistry
from modules.rule_engine      import RuleRegistry, RuleEncoder, Rule

# ── Preprocessing ─────────────────────────────────────────────────────────────
from preprocessing.industrial_tokenizer import (
    IndustrialTokenizer, ChannelSpec, TokenType,
)
from preprocessing.log_encoder import LogEncoder

# ── Distributed training ──────────────────────────────────────────────────────
from training.distributed import (
    init_process_group, destroy_process_group,
    is_initialized, get_rank, get_world_size,
    is_main_process, barrier,
    broadcast_parameters, allreduce_gradients, allreduce_scalar,
)
from training.distributed_sampler import DistributedSampler

# ── Monitoring ────────────────────────────────────────────────────────────────
from monitoring.drift import DriftDetector

# ── Utilities ─────────────────────────────────────────────────────────────────
from vulgaris.utils      import set_seed, get_rng
from vulgaris.pretrained import from_pretrained, save_pretrained


__all__ = [
    # Version
    "__version__",

    # Config
    "ModelConfig", "ASEConfig", "SSSRConfig", "CRGConfig", "HMBConfig",
    "SHCALConfig", "DAHConfig", "ESEConfig", "HTDConfig", "SafetyConfig",
    "TrainingConfig", "MultiTaskConfig",

    # Core model
    "Vulgaris", "VulgarisState", "OutputHead",

    # Engine
    "Tensor", "Parameter", "Module", "Linear", "LayerNorm", "RMSNorm",
    "zeros", "ones", "randn", "rand", "cat", "stack",

    # Training
    "TrainingPipeline", "TimeSeriesAugment", "CurriculumSchedule",
    "VulgarisLoss", "SpectralAdamW", "CosineSchedule", "MuonOptimizer",
    "NonStationaryConformal", "DistillationLoss", "DistillationTrainer",
    "ActiveLearner",

    # Inference
    "StreamingInference", "SpeculativeRollout", "EventBuffer",

    # Serving
    "DegradationController", "DeploymentMode", "CanaryController", "AuditLogger",
    "migrate_checkpoint", "HotSwapAdapter", "get_checkpoint_version",

    # Distributed
    "init_process_group", "destroy_process_group",
    "is_initialized", "get_rank", "get_world_size",
    "is_main_process", "barrier",
    "broadcast_parameters", "allreduce_gradients", "allreduce_scalar",
    "DistributedSampler",

    # Modules
    "AdaptiveSignalEmbedding", "SelectiveSSR", "HierarchicalTimescaleDecomposition",
    "CausalRoutingGraph", "HierarchicalMemoryBank", "DomainAdaptiveHypernetwork",
    "ExplainabilityEngine", "SafetyPolicyHead", "SHCAL", "CrossModalLatentAlignment",
    "InContextLearning", "RegimeMixtureCore", "OntologyEmbedding", "OntologyRegistry",
    "Rule", "RuleRegistry", "RuleEncoder",

    # Preprocessing
    "IndustrialTokenizer", "ChannelSpec", "TokenType", "LogEncoder",

    # Monitoring
    "DriftDetector",

    # Utilities
    "set_seed", "get_rng", "from_pretrained", "save_pretrained",
]
