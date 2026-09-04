"""Least-frequently-used eviction, over an indexed min-heap.

Hit counts change while entries are resident, which is precisely the mutable-
score case ``IndexedMinHeap`` exists for: an access is a decrease/increase-key,
not a reinsertion.

Ties on frequency fall back to insertion order via the heap's internal tie
break, so a long-resident entry and a freshly admitted one with the same count
resolve deterministically rather than by dict ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..heap import IndexedMinHeap
from .base import EvictionPolicy

if TYPE_CHECKING:
    from ..cache import CacheEntry


class LFUPolicy(EvictionPolicy):
    """Evicts the entry with the fewest accesses."""

    def __init__(self) -> None:
        self._heap: IndexedMinHeap[int] = IndexedMinHeap()

    def __len__(self) -> int:
        return len(self._heap)

    def on_insert(self, entry: "CacheEntry") -> None:
        self._heap.push(entry.key, float(entry.hits))

    def on_hit(self, entry: "CacheEntry") -> None:
        self._heap.update(entry.key, float(entry.hits))

    def on_remove(self, entry: "CacheEntry") -> None:
        if entry.key in self._heap:
            self._heap.remove(entry.key)

    def choose_victim(self) -> int:
        key, _ = self._heap.peek()
        return key
