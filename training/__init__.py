"""
VULGARIS training utilities.

Includes loss function, SpectralAdamW optimizer, cosine schedule,
non-stationary conformal prediction, and the full training pipeline.
"""

from __future__ import annotations

from .loss       import VulgarisLoss
from .optimizer  import SpectralAdamW, CosineSchedule
from .conformal  import NonStationaryConformal
from .pipeline   import TrainingPipeline
from .self_supervised import SelfSupervisedTrainer

__all__ = [
    "VulgarisLoss",
    "SpectralAdamW",
    "CosineSchedule",
    "NonStationaryConformal",
    "TrainingPipeline",
    "SelfSupervisedTrainer",
]
