"""Reproducibility utilities for VULGARIS."""
from __future__ import annotations

import numpy as np
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Sets numpy's global random seed. Since VULGARIS uses only numpy
    for computation, this covers all random operations in the framework.

    Args:
        seed: Integer random seed. Default 42.

    Example::

        import vulgaris
        vulgaris.set_seed(0)
        model = vulgaris.Vulgaris(config)  # weights initialized deterministically
    """
    np.random.seed(seed)


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create an isolated numpy random Generator instance.

    Prefer this over set_seed() when you need reproducibility in a
    specific module without affecting global state.

    Args:
        seed: Optional integer seed. If None, uses system entropy.

    Returns:
        numpy.random.Generator instance.
    """
    return np.random.default_rng(seed)
