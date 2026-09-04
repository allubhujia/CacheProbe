"""First-in-first-out eviction.

The weakest baseline, and useful precisely because it is: FIFO ignores both
access frequency and recency, so the gap between FIFO and LRU measures what
recency information alone is worth on a given trace. If a smarter policy fails
to beat FIFO by much, the trace has little exploitable structure and no
eviction policy will help.

Accesses are deliberately ignored - an entry's position is fixed at insertion.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from .base import EvictionPolicy

if TYPE_CHECKING:
    from ..cache import CacheEntry


class FIFOPolicy(EvictionPolicy):
    """Evicts whichever entry was admitted earliest."""

    def __init__(self) -> None:
        self._order: deque[int] = deque()
        self._live: set[int] = set()

    def __len__(self) -> int:
        return len(self._live)

    def on_insert(self, entry: "CacheEntry") -> None:
        self._order.append(entry.key)
        self._live.add(entry.key)

    def on_remove(self, entry: "CacheEntry") -> None:
        self._live.discard(entry.key)
        # The key stays in the deque and is skipped lazily by choose_victim,
        # so removal stays O(1) instead of scanning for the key's position.

    def choose_victim(self) -> int:
        while self._order:
            key = self._order[0]
            if key in self._live:
                return key
            self._order.popleft()  # stale: already removed
        raise IndexError("no entries to evict")
