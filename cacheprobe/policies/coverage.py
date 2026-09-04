"""Coverage-aware eviction - Contribution A.

Every classical policy assumes an entry serves exactly one key, which makes
recency and frequency adequate proxies for value. A semantic cache breaks that
assumption: an entry serves a *region* of embedding space, and neighbouring
entries' regions overlap. Evicting an entry ringed by close neighbours costs
almost nothing, because those neighbours still answer the queries it was
serving. Evicting an isolated entry removes coverage of its region outright.

    value(e) = w1 * normalised_frequency(e)
             + w2 * normalised_recency(e)
             - w3 * redundancy(e)

with redundancy maintained incrementally by ``cache.py`` (section 5.3) as the
sum of max(0, sim - tau) over resident neighbours. The victim is argmin value,
read from the min-heap root.

Normalisation and heap staleness. Frequency and recency are normalised against
running maxima, so a change to either denominator technically perturbs every
entry's score at once - re-scoring all n entries on each event would defeat the
point of the heap. Scores are therefore refreshed only for entries the cache
actually touches (insert, hit, or a neighbour's redundancy shift), leaving
untouched entries mildly stale. ``exact=True`` re-scores everything at
selection time instead, which is O(n) and exists so tests can measure how far
the cheap path drifts from the true argmin rather than assuming it doesn't.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..heap import IndexedMinHeap
from .base import EvictionPolicy

if TYPE_CHECKING:
    from ..cache import CacheEntry


class CoveragePolicy(EvictionPolicy):
    """Evicts the entry whose loss costs the least coverage.

    ``w3 = 0`` reduces this to the frequency+recency ablation of section 10.3 -
    the comparison that decides whether the redundancy term earns its place.
    """

    def __init__(
        self,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 1.0,
        recency_half_life: float = 500.0,
        exact: bool = False,
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.recency_half_life = recency_half_life
        self.exact = exact

        self._heap: IndexedMinHeap[int] = IndexedMinHeap()
        self._entries: dict[int, "CacheEntry"] = {}

        self._tick = 0  # monotonic logical clock; avoids wall-clock jitter
        self._last_tick: dict[int, int] = {}
        self._max_hits = 1
        self._max_redundancy = 1e-9

    def __len__(self) -> int:
        return len(self._entries)

    # ---------------------------------------------------------------- scoring

    def _score(self, entry: "CacheEntry") -> float:
        frequency = entry.hits / self._max_hits

        # Recency decays with how many events ago this entry was last touched,
        # so it needs no wall clock and cannot drift while the cache is idle.
        age = self._tick - self._last_tick.get(entry.key, 0)
        recency = math.exp(-age / self.recency_half_life)

        redundancy = entry.redundancy / self._max_redundancy

        return self.w1 * frequency + self.w2 * recency - self.w3 * redundancy

    def _refresh(self, entry: "CacheEntry") -> None:
        self._heap.push_or_update(entry.key, self._score(entry))

    # ------------------------------------------------------------------ hooks

    def on_insert(self, entry: "CacheEntry") -> None:
        self._tick += 1
        self._entries[entry.key] = entry
        self._last_tick[entry.key] = self._tick
        self._max_hits = max(self._max_hits, entry.hits or 1)
        self._max_redundancy = max(self._max_redundancy, entry.redundancy)
        self._refresh(entry)

    def on_hit(self, entry: "CacheEntry") -> None:
        self._tick += 1
        self._last_tick[entry.key] = self._tick
        self._max_hits = max(self._max_hits, entry.hits)
        self._refresh(entry)

    def on_score_change(self, entry: "CacheEntry") -> None:
        """A neighbour was inserted or evicted, changing this entry's redundancy."""
        if entry.key not in self._entries:
            return
        self._max_redundancy = max(self._max_redundancy, entry.redundancy)
        self._refresh(entry)

    def on_remove(self, entry: "CacheEntry") -> None:
        self._entries.pop(entry.key, None)
        self._last_tick.pop(entry.key, None)
        if entry.key in self._heap:
            self._heap.remove(entry.key)

    # --------------------------------------------------------------- eviction

    def choose_victim(self) -> int:
        if not self._entries:
            raise IndexError("no entries to evict")
        if self.exact:
            return min(self._entries.values(), key=self._score).key
        key, _ = self._heap.peek()
        return key

    # ------------------------------------------------------------ diagnostics

    def score_of(self, key: int) -> float:
        return self._score(self._entries[key])

    def rank_error(self) -> int:
        """Positions between the heap's victim and the true argmin.

        Zero means the incremental scores agree with a full re-scoring. Used by
        the tests to quantify the staleness the cheap path accepts.
        """
        if not self._entries:
            return 0
        ordered = sorted(self._entries.values(), key=self._score)
        heap_victim, _ = self._heap.peek()
        for position, entry in enumerate(ordered):
            if entry.key == heap_victim:
                return position
        return len(ordered)
