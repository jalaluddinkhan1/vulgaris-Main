"""Compatibility shim — real config lives in vulgaris.config."""
from vulgaris.config import (  # noqa: F401
    ASEConfig, SSSRConfig, CRGConfig, HMBConfig, SHCALConfig,
    DAHConfig, ESEConfig, HTDConfig, SafetyConfig, TrainingConfig,
    MultiTaskConfig, ModelConfig,
)
