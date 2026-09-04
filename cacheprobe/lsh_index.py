"""Random-hyperplane LSH over embeddings, with chained buckets and rehashing.

This is the sublinear neighbour lookup everything else depends on (section
7.1). A brute-force scan is O(n * d) per query; at n = 10,000 and d = 384 that
erodes the entire latency advantage the cache exists to provide. LSH trades
exactness for a probe that costs roughly O(L * k * d + candidates).

Hash function. For k random hyperplanes with normals drawn from a standard
normal distribution, h_i(v) = 1 if dot(normal_i, v) >= 0 else 0. Two vectors at
angle theta collide on one bit with probability 1 - theta/pi, so near
neighbours collide on the full k-bit signature with high probability while far
ones mostly don't.

Amplification. L independent tables, each with its own k hyperplanes. A probe
takes the union of all L bucket hits, which raises recall at the cost of L
times the work. ``theoretical_recall`` gives the closed-form curve to check
measured recall against.

Rehashing. If average bucket occupancy exceeds ``max_load_factor``, k is
incremented (finer signatures -> smaller buckets) and every table is rebuilt
from the vectors already on hand. This is what keeps probe cost from
degrading as the index grows.
"""

from __future__ import annotations

import math
from typing import Generic, Hashable, TypeVar

import numpy as np

K = TypeVar("K", bound=Hashable)


def theoretical_recall(theta: float, k: int, L: int) -> float:
    """Probability a true neighbour at angle ``theta`` (radians) is found.

    recall = 1 - (1 - (1 - theta/pi)^k)^L
    """
    p_bit = 1.0 - theta / math.pi
    p_table = p_bit ** k
    return 1.0 - (1.0 - p_table) ** L


class _ChainNode(Generic[K]):
    """Singly linked-list node; buckets are chains of these, not Python lists."""

    __slots__ = ("key", "next")

    def __init__(self, key: K, next: "_ChainNode[K] | None" = None) -> None:
        self.key = key
        self.next = next


class _Chain(Generic[K]):
    """Separate-chaining bucket: a singly linked list with O(1) prepend."""

    __slots__ = ("head", "size")

    def __init__(self) -> None:
        self.head: _ChainNode[K] | None = None
        self.size = 0

    def append(self, key: K) -> None:
        self.head = _ChainNode(key, self.head)
        self.size += 1

    def remove(self, key: K) -> bool:
        prev, node = None, self.head
        while node is not None:
            if node.key == key:
                if prev is None:
                    self.head = node.next
                else:
                    prev.next = node.next
                self.size -= 1
                return True
            prev, node = node, node.next
        return False

    def __iter__(self):
        node = self.head
        while node is not None:
            yield node.key
            node = node.next

    def __len__(self) -> int:
        return self.size


class LSHIndex(Generic[K]):
    """Approximate cosine-similarity nearest-neighbour index.

    Vectors passed in are assumed L2-normalised (``embedder.py`` normalises by
    default), so cosine similarity is a plain dot product and the hyperplane
    hash's angle-based collision probability applies directly.
    """

    def __init__(
        self,
        dim: int,
        k: int = 8,
        L: int = 4,
        seed: int = 0,
        max_load_factor: float = 4.0,
        max_k: int = 32,
    ) -> None:
        self.dim = dim
        self.k = k
        self.L = L
        self.max_load_factor = max_load_factor
        self.max_k = max_k

        self._rng = np.random.default_rng(seed)
        self._planes: np.ndarray = self._draw_planes(k)  # (L, k, dim)
        self._tables: list[dict[int, _Chain[K]]] = [dict() for _ in range(L)]

        self._vectors: dict[K, np.ndarray] = {}
        self._signatures: dict[K, list[int]] = {}  # per-key sig in each table

    # ------------------------------------------------------------- internals

    def _draw_planes(self, k: int) -> np.ndarray:
        return self._rng.standard_normal((self.L, k, self.dim)).astype(np.float32)

    def _signatures_for(self, vector: np.ndarray) -> list[int]:
        """Compute this vector's k-bit signature in each of the L tables.

        One matmul per table rather than a per-hyperplane Python loop, per the
        vectorisation note in section 12.3 - a scalar loop over k and L would
        be slow enough to invalidate timing results.
        """
        sigs = []
        for t in range(self.L):
            bits = self._planes[t] @ vector >= 0  # (k,) bool
            sig = 0
            for b in bits:  # k is small (tens of bits); packing itself is cheap
                sig = (sig << 1) | int(b)
            sigs.append(sig)
        return sigs

    def _bucket_occupancy(self) -> float:
        total_entries = sum(len(chain) for table in self._tables for chain in table.values())
        total_buckets = sum(len(table) for table in self._tables) or 1
        return total_entries / total_buckets

    def _maybe_rehash(self) -> None:
        if self.k >= self.max_k:
            return
        if self._bucket_occupancy() <= self.max_load_factor:
            return

        new_k = min(self.k + 1, self.max_k)
        self._planes = self._draw_planes(new_k)
        self.k = new_k
        self._tables = [dict() for _ in range(self.L)]
        self._signatures = {}

        for key, vector in self._vectors.items():
            self._place(key, vector)

    def _place(self, key: K, vector: np.ndarray) -> None:
        sigs = self._signatures_for(vector)
        self._signatures[key] = sigs
        for t, sig in enumerate(sigs):
            self._tables[t].setdefault(sig, _Chain()).append(key)

    # ------------------------------------------------------------------ API

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, key: K) -> bool:
        return key in self._vectors

    def insert(self, key: K, vector: np.ndarray) -> None:
        if key in self._vectors:
            raise KeyError(f"{key!r} already indexed; call remove() first")
        vector = np.asarray(vector, dtype=np.float32)
        self._vectors[key] = vector
        self._place(key, vector)
        self._maybe_rehash()

    def remove(self, key: K) -> None:
        del self._vectors[key]
        sigs = self._signatures.pop(key)
        for t, sig in enumerate(sigs):
            chain = self._tables[t].get(sig)
            if chain is not None:
                chain.remove(key)
                if len(chain) == 0:
                    del self._tables[t][sig]

    def probe(self, vector: np.ndarray) -> set[K]:
        """Union of bucket contents across all L tables for ``vector``."""
        vector = np.asarray(vector, dtype=np.float32)
        candidates: set[K] = set()
        for t in range(self.L):
            bits = self._planes[t] @ vector >= 0
            sig = 0
            for b in bits:
                sig = (sig << 1) | int(b)
            chain = self._tables[t].get(sig)
            if chain is not None:
                candidates.update(chain)
        return candidates
