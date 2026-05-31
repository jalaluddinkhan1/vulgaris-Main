"""
VULGARIS training utilities.

Includes loss, optimizers, schedules, conformal prediction, distillation,
active learning, and distributed training primitives.
"""

from __future__ import annotations

from .loss              import VulgarisLoss
from .optimizer         import SpectralAdamW, CosineSchedule, MuonOptimizer
from .conformal         import NonStationaryConformal
from .pipeline          import TrainingPipeline, TimeSeriesAugment, CurriculumSchedule
from .self_supervised   import SelfSupervisedTrainer
from .distillation      import DistillationLoss, DistillationTrainer
from .active_learning   import ActiveLearner
from .distributed       import (
    init_process_group, destroy_process_group,
    is_initialized, get_rank, get_world_size,
    is_main_process, barrier,
    broadcast_parameters, allreduce_gradients, allreduce_scalar,
)
from .distributed_sampler import DistributedSampler

__all__ = [
    "VulgarisLoss",
    "SpectralAdamW",
    "CosineSchedule",
    "MuonOptimizer",
    "NonStationaryConformal",
    "TrainingPipeline",
    "TimeSeriesAugment",
    "CurriculumSchedule",
    "SelfSupervisedTrainer",
    "DistillationLoss",
    "DistillationTrainer",
    "ActiveLearner",
    "init_process_group",
    "destroy_process_group",
    "is_initialized",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "broadcast_parameters",
    "allreduce_gradients",
    "allreduce_scalar",
    "DistributedSampler",
]
