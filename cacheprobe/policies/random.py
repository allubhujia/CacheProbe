"""Random eviction - the control the other policies must beat.

Any policy that fails to beat random selection is contributing nothing, so
this is the floor every result is read against.

Keys are held in a list plus a key-to-index map, which makes removal O(1) by
swapping the doomed key with the last element and popping - a plain
``list.remove`` would be O(n) and would show up in the eviction-overhead
measurements as if it were an inherent cost of eviction.

The generator is seeded (section 12.3 requires fixed seeds for the random
baseline) so a run is reproducible.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .base import EvictionPolicy

if TYPE_CHECKING:
    from ..cache import CacheEntry


class RandomPolicy(EvictionPolicy):
    """Evicts a uniformly chosen resident entry."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._keys: list[int] = []
        self._index: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._keys)

    def on_insert(self, entry: "CacheEntry") -> None:
        self._index[entry.key] = len(self._keys)
        self._keys.append(entry.key)

    def on_remove(self, entry: "CacheEntry") -> None:
        i = self._index.pop(entry.key, None)
        if i is None:
            return
        last = self._keys.pop()
        if last != entry.key:
            self._keys[i] = last
            self._index[last] = i

    def choose_victim(self) -> int:
        if not self._keys:
            raise IndexError("no entries to evict")
        return self._keys[self._rng.randrange(len(self._keys))]
