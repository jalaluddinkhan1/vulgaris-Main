"""
VULGARIS benchmarking suite.

Baseline model comparisons and standard dataset loaders
(ETT, NAB, EdgeTelemetry — with synthetic fallback).
"""

from __future__ import annotations

from .baselines     import (
    LastValue,
    MovingAverage,
    ExponentialSmoothing,
    ARIMA_lite,
    LSTMLite,
    run_baseline_comparison,
)
from .real_datasets import (
    ETTDataset,
    NABDataset,
    EdgeTelemetryDataset,
    load_benchmark_suite,
)

__all__ = [
    "LastValue", "MovingAverage", "ExponentialSmoothing",
    "ARIMA_lite", "LSTMLite", "run_baseline_comparison",
    "ETTDataset", "NABDataset", "EdgeTelemetryDataset",
    "load_benchmark_suite",
]
