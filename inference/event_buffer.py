"""
Lock-free shared-memory event ring buffer for high-throughput log ingestion.

Architecture
------------
Producer (log tail / network socket) writes encoded events into a
`multiprocessing.shared_memory` block.  Consumer (inference engine)
drains batches of N events without copying.

Layout of the shared block (all little-endian int64 / float64):
  [0]      write_cursor  (int64)  — producer increments atomically
  [1]      read_cursor   (int64)  — consumer increments after each batch
  [2:]     ring data     (float64 pairs: [template_id, severity] × CAPACITY)

The ring never wraps mid-batch: if capacity - write_pos < batch, the
producer stalls until the consumer catches up (back-pressure).

Usage
-----
    # Producer process / thread
    buf = EventBuffer(capacity=8192, batch_size=1000)
    buf.put(encoded_event)          # encoded_event: (2,) float64

    # Consumer process / thread (inference side)
    batch = buf.get_batch()         # returns (N, 2) array or None
    if batch is not None:
        model.infer(batch)
"""

from __future__ import annotations

import time
import ctypes
import numpy as np
from multiprocessing import shared_memory
from typing import Optional


_HEADER_INTS = 2           # write_cursor, read_cursor
_CHANNELS    = 2           # template_id, severity
_SPIN_SLEEP  = 1e-4        # seconds between spin-wait retries


class EventBuffer:
    """
    Shared-memory ring buffer for (template_id, severity) event pairs.

    Parameters
    ----------
    capacity : int
        Maximum events stored simultaneously.  Must be a power of 2.
    batch_size : int
        Consumer drains this many events per `get_batch()` call.
    name : str | None
        Shared-memory block name.  If None, a new block is created and
        its name is stored in `self.shm_name`.  Pass the name to attach
        from a second process.
    create : bool
        True  → allocate new shared block (producer side).
        False → attach to existing block (consumer side).
    """

    def __init__(
        self,
        capacity: int = 8192,
        batch_size: int = 1000,
        name: Optional[str] = None,
        create: bool = True,
    ):
        if capacity & (capacity - 1):
            raise ValueError("capacity must be a power of 2")

        self.capacity   = capacity
        self.batch_size = batch_size
        self._mask      = capacity - 1

        # Byte layout: 2 int64 header + capacity * 2 float64 event slots
        header_bytes = _HEADER_INTS * 8
        data_bytes   = capacity * _CHANNELS * 8
        total_bytes  = header_bytes + data_bytes

        if create:
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=total_bytes
            )
        else:
            if name is None:
                raise ValueError("name required when create=False")
            self._shm = shared_memory.SharedMemory(name=name, create=False)

        self.shm_name = self._shm.name

        # Header view (int64)
        self._header = np.ndarray(
            (_HEADER_INTS,), dtype=np.int64, buffer=self._shm.buf, offset=0
        )

        # Data view (float64): shape (capacity, 2)
        self._data = np.ndarray(
            (capacity, _CHANNELS),
            dtype=np.float64,
            buffer=self._shm.buf,
            offset=header_bytes,
        )

        if create:
            self._header[:] = 0

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------

    def put(self, event: np.ndarray, timeout: float = 5.0) -> bool:
        """
        Write a single (2,) event.  Blocks with back-pressure if full.
        Returns False on timeout.
        """
        deadline = time.monotonic() + timeout
        while True:
            wc = int(self._header[0])
            rc = int(self._header[1])
            if wc - rc < self.capacity:
                slot = wc & self._mask
                self._data[slot] = event
                # Ensure data visible before bumping cursor (store-release)
                ctypes.atomic_memcpy_fence() if hasattr(ctypes, "atomic_memcpy_fence") else None
                self._header[0] = wc + 1
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(_SPIN_SLEEP)

    def put_batch(self, events: np.ndarray, timeout: float = 5.0) -> bool:
        """
        Write a (N, 2) array of events.  Writes are serialised per-event.
        """
        for ev in events:
            if not self.put(ev, timeout=timeout):
                return False
        return True

    # ------------------------------------------------------------------
    # Consumer side
    # ------------------------------------------------------------------

    def get_batch(self, timeout: float = 0.0) -> Optional[np.ndarray]:
        """
        Return the next `batch_size` events as (N, 2) float64 array.
        Returns None if fewer events are available and timeout expires.

        Set timeout > 0 to block until a full batch arrives.
        """
        deadline = time.monotonic() + timeout
        while True:
            wc = int(self._header[0])
            rc = int(self._header[1])
            available = wc - rc
            if available >= self.batch_size:
                n = self.batch_size
            elif available > 0 and timeout == 0.0:
                n = available      # partial drain (non-blocking mode)
            else:
                if time.monotonic() > deadline:
                    return None
                time.sleep(_SPIN_SLEEP)
                continue

            # Gather events — handle ring wrap
            out = np.empty((n, _CHANNELS), dtype=np.float64)
            for i in range(n):
                slot = (rc + i) & self._mask
                out[i] = self._data[slot]

            self._header[1] = rc + n
            return out

    # ------------------------------------------------------------------
    # Stats / housekeeping
    # ------------------------------------------------------------------

    def occupancy(self) -> int:
        """Events currently in the buffer."""
        return int(self._header[0]) - int(self._header[1])

    def close(self):
        """Detach from shared memory (do not unlink)."""
        self._shm.close()

    def unlink(self):
        """Destroy the shared-memory block (call once, from creator)."""
        self._shm.close()
        self._shm.unlink()

    def __del__(self):
        try:
            self._shm.close()
        except Exception:
            pass
