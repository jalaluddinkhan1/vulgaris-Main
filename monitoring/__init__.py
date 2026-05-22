"""
VULGARIS monitoring utilities.

Distribution drift detection using KS, MMD, and Wasserstein statistics.
"""

from __future__ import annotations

from .drift import DriftDetector

__all__ = ["DriftDetector"]
