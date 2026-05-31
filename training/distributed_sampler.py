"""
Distributed data sampler for VULGARIS multi-process training.

Splits a dataset of N samples into `world_size` disjoint shards so that
every rank processes a different subset each epoch.  All ranks always
receive the same number of batches (the last shard is padded by repeating
from the beginning so no rank is idle while others are still running).

Usage
-----
    from training.distributed_sampler import DistributedSampler

    sampler = DistributedSampler(
        dataset_size=len(my_dataset),
        shuffle=True,
        seed=42,
    )

    for epoch in range(n_epochs):
        indices = sampler.get_indices(epoch)      # list of ints for this rank
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            x, y = my_dataset[idx]
            pipeline.train_step(x, y)
from __future__ import annotations

"""


import numpy as np
from typing import List, Optional


class DistributedSampler:
    """
    Deterministic, reproducible data sharding across `num_replicas` workers.

    Parameters
    ----------
    dataset_size  : total number of samples in the dataset.
    num_replicas  : number of workers (defaults to world_size).
    rank          : rank of this worker (defaults to current rank).
    shuffle       : shuffle sample order each epoch (default True).
    seed          : base random seed.  epoch is XOR'd in so each epoch
                    gets a different but reproducible permutation.
    drop_last     : if True, drop samples that don't divide evenly instead
                    of padding.  All ranks then see exactly
                    dataset_size // num_replicas samples.
    """

    def __init__(
        self,
        dataset_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        from .distributed import get_rank, get_world_size, is_initialized

        if num_replicas is None:
            num_replicas = get_world_size() if is_initialized() else 1
        if rank is None:
            rank = get_rank() if is_initialized() else 0

        if rank >= num_replicas:
            raise ValueError(f"rank={rank} >= num_replicas={num_replicas}")

        self.dataset_size  = dataset_size
        self.num_replicas  = num_replicas
        self.rank          = rank
        self.shuffle       = shuffle
        self.seed          = seed
        self.drop_last     = drop_last

        if drop_last:
            self.num_samples = dataset_size // num_replicas
        else:
            # Ceiling division — pad last shard to equalise batch counts
            self.num_samples = (dataset_size + num_replicas - 1) // num_replicas

        self.total_size = self.num_samples * num_replicas

    # ------------------------------------------------------------------

    def get_indices(self, epoch: int = 0) -> List[int]:
        """
        Return the list of dataset indices for this rank at the given epoch.

        The seed is derived as `self.seed ^ (epoch * 2654435761)` so that
        every epoch produces a different permutation while remaining fully
        reproducible across restarts.
        """
        epoch_seed = self.seed ^ (epoch * 0x9E3779B9)   # Knuth multiplicative hash
        rng = np.random.default_rng(epoch_seed)

        if self.shuffle:
            indices: List[int] = rng.permutation(self.dataset_size).tolist()
        else:
            indices = list(range(self.dataset_size))

        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            # Pad by repeating from the front
            pad = self.total_size - len(indices)
            if pad > 0:
                indices = indices + indices[:pad]

        assert len(indices) == self.total_size

        # Slice this rank's contiguous chunk
        start = self.rank * self.num_samples
        return indices[start : start + self.num_samples]

    def __len__(self) -> int:
        return self.num_samples

    # ------------------------------------------------------------------

    @staticmethod
    def build_batches(
        indices: List[int],
        batch_size: int,
        drop_last: bool = False,
    ) -> List[List[int]]:
        """
        Partition `indices` into sub-lists of length `batch_size`.

        Parameters
        ----------
        drop_last : if True, discard the final batch if smaller than batch_size.
        """
        batches = []
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            if drop_last and len(batch) < batch_size:
                break
            batches.append(batch)
        return batches
