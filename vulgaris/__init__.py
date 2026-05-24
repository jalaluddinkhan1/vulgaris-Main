"""VULGARIS — Foundational Industrial AI Model for streaming telemetry, IoT, and time-series."""

from __future__ import annotations

__version__ = "0.3.0"
__author__  = "VULGARIS Contributors"
__license__ = "Apache-2.0"

from vulgaris.config import ModelConfig, ASEConfig, SSSRConfig, CRGConfig, HMBConfig
from vulgaris.config import SHCALConfig, DAHConfig, ESEConfig, HTDConfig, SafetyConfig
from vulgaris.config import TrainingConfig, MultiTaskConfig

from model.vulgaris import Vulgaris, VulgarisState, OutputHead

from engine.tensor import Tensor, Parameter, zeros, ones, randn, rand, cat, stack
from engine.module import Module

from training.pipeline  import TrainingPipeline
from training.loss      import VulgarisLoss
from training.optimizer import SpectralAdamW, CosineSchedule
from training.conformal import NonStationaryConformal

from modules.multitask_head import MultiTaskHead
from monitoring.drift       import DriftDetector
from vulgaris.utils         import set_seed, get_rng
from vulgaris.pretrained    import from_pretrained, save_pretrained

__all__ = [
    "__version__",
    "ModelConfig", "ASEConfig", "SSSRConfig", "CRGConfig", "HMBConfig",
    "SHCALConfig", "DAHConfig", "ESEConfig", "HTDConfig", "SafetyConfig",
    "TrainingConfig", "MultiTaskConfig",
    "Vulgaris", "VulgarisState", "OutputHead",
    "Tensor", "Parameter", "Module",
    "zeros", "ones", "randn", "rand", "cat", "stack",
    "TrainingPipeline", "VulgarisLoss", "SpectralAdamW", "CosineSchedule",
    "NonStationaryConformal", "MultiTaskHead", "DriftDetector",
    "set_seed", "get_rng",
    "from_pretrained", "save_pretrained",
]
