"""The eviction-policy interface every baseline and the proposal share.

Section 12.2 is explicit about why this interface exists: with every policy
running through identical cache machinery, a measured difference between them
is attributable to the policy rather than to how well each one happened to be
implemented. Nothing here may touch the LSH index, the oracle, or the entry
store - a policy only observes events and names a victim.

The cache calls the hooks; the policy answers ``choose_victim``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoids a runtime import cycle with cache.py
    from ..cache import CacheEntry


class EvictionPolicy(ABC):
    """Observes cache events and names the entry to drop when full."""

    def on_insert(self, entry: "CacheEntry") -> None:
        """A new entry was admitted."""

    def on_hit(self, entry: "CacheEntry") -> None:
        """An existing entry served a query."""

    def on_remove(self, entry: "CacheEntry") -> None:
        """An entry left the cache, whether evicted or dropped explicitly."""

    def on_score_change(self, entry: "CacheEntry") -> None:
        """An entry's value changed for a reason other than its own access.

        Only coverage-aware eviction reacts to this: inserting or evicting a
        neighbour changes an entry's redundancy without that entry being
        touched. Recency- and frequency-based policies have no such coupling
        between entries and correctly ignore it.
        """

    @abstractmethod
    def choose_victim(self) -> int:
        """Return the key of the entry to evict. Must not mutate the cache."""

    def __len__(self) -> int:
        raise NotImplementedError
