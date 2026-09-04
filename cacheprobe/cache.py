"""The semantic cache itself: lookup, admission, eviction, redundancy upkeep.

Implements the query flow of section 5.1. An incoming query is answered from a
stored entry when its embedding lies within ``tau`` of that entry's, which is
what lets a paraphrase hit an entry no exact-match cache could serve.

The embedding function is injected rather than imported. Two reasons: the
experiments precompute a trace's vectors once and replay them across every
policy (section 12.3), so re-embedding inside the cache would be wasted work;
and it keeps this module testable without loading a transformer.

Redundancy upkeep (section 5.3) runs here rather than inside the coverage
policy, because it is the cache that knows which entries neighbour which. Each
entry stores its own neighbour list with precomputed deltas, so an eviction
decrements its neighbours without recomputing a single similarity. Both insert
and evict touch O(b) entries - the average bucket occupancy - rather than O(n).
Policies that ignore redundancy simply never read the field.

A coupling worth knowing about before reading any Contribution A result:
neighbours are discovered by probing the LSH index, so **redundancy accuracy is
bounded by LSH recall**. At k=12, L=8 the recall for a pair at cosine 0.8 is
only about 0.43 by the closed form in ``lsh_index.theoretical_recall``, which
means over half the redundancy signal is silently dropped and the coverage
policy degrades toward plain frequency+recency. Lower k and higher L recover it
(k=6, L=24 gives >0.99 at the same angle) at the cost of larger candidate sets.
This makes the LSH parameters a confound for Contribution A rather than a pure
latency knob, and any eviction comparison should report the recall it ran at.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .lsh_index import LSHIndex
from .oracle.base import Oracle, OracleResponse
from .policies.base import EvictionPolicy


@dataclass
class CacheEntry:
    """One stored query/response pair and the bookkeeping eviction needs."""

    key: int
    query: str
    vector: np.ndarray
    response: OracleResponse

    #: Ground-truth cluster the stored query came from. Used only to score
    #: false hits offline; the cache never consults it when matching.
    cluster_id: int | None = None

    hits: int = 0
    inserted_at: float = 0.0
    last_access: float = 0.0

    #: Sum of max(0, sim - tau) over resident neighbours. High redundancy means
    #: this entry's region of embedding space is already well covered by
    #: others, so losing it costs little.
    redundancy: float = 0.0

    #: neighbour key -> the delta this pair contributes to both redundancies.
    #: Stored so eviction can decrement without recomputing similarity.
    neighbours: dict[int, float] = field(default_factory=dict)


@dataclass
class LookupResult:
    """What a single query did, for the caller and for the metrics."""

    query: str
    response: OracleResponse
    hit: bool
    similarity: float | None = None
    entry_key: int | None = None
    #: True when a hit served an entry from a different ground-truth cluster.
    false_hit: bool = False
    candidates_examined: int = 0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    exact_hits: int = 0
    false_hits: int = 0
    evictions: int = 0
    insertions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def false_hit_rate(self) -> float:
        """Share of *hits* that served the wrong cluster (section 10.4)."""
        return self.false_hits / self.hits if self.hits else 0.0


class SemanticCache:
    """Bounded cache keyed by embedding proximity.

    ``tau`` is the similarity threshold a candidate must reach to be served.
    Section 6.3 is worth remembering when tuning it: lowering ``tau`` raises the
    hit rate and simultaneously widens the radius an attacker's paraphrase can
    match within, so it trades off against privacy as well as against false
    hits.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        oracle: Oracle,
        policy: EvictionPolicy,
        capacity: int,
        tau: float = 0.85,
        dim: int = 384,
        lsh_k: int = 12,
        lsh_L: int = 8,
        seed: int = 0,
        redundancy_margin: float = 0.10,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")

        self.embed_fn = embed_fn
        self.oracle = oracle
        self.policy = policy
        self.capacity = capacity
        self.tau = tau

        # Redundancy has to be counted from a *lower* threshold than the one
        # used for matching. Section 5.2 defines delta as max(0, sim - tau),
        # but an entry closer than tau to a resident entry would have hit and
        # never been admitted - so under that definition no two resident
        # entries can ever be within tau and redundancy is identically zero.
        #
        # Counting from tau - margin instead captures what the contribution
        # actually means: neighbours near enough that they nearly cover this
        # entry's region, so losing it costs little. Setting margin to 0
        # recovers the document's literal formula, which is useful as an
        # ablation showing why the margin is needed.
        self.redundancy_margin = redundancy_margin
        self.redundancy_tau = max(0.0, tau - redundancy_margin)

        self._entries: dict[int, CacheEntry] = {}
        self._exact: dict[str, int] = {}
        self._index: LSHIndex[int] = LSHIndex(dim=dim, k=lsh_k, L=lsh_L, seed=seed)
        self._next_key = 0

        self.stats = CacheStats()

    # ------------------------------------------------------------- accessors

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, query: str) -> bool:
        return query in self._exact

    def entries(self) -> list[CacheEntry]:
        return list(self._entries.values())

    def get_entry(self, key: int) -> CacheEntry:
        return self._entries[key]

    # ------------------------------------------------------------- retrieval

    def _best_match(self, vector: np.ndarray) -> tuple[CacheEntry | None, float, int]:
        """Exact-rank the LSH candidate set; return (entry, similarity, n_seen).

        Vectors are unit-norm, so cosine similarity is a dot product.
        """
        candidates = self._index.probe(vector)
        best: CacheEntry | None = None
        best_sim = -1.0
        for key in candidates:
            entry = self._entries.get(key)
            if entry is None:
                continue
            sim = float(np.dot(vector, entry.vector))
            if sim > best_sim:
                best, best_sim = entry, sim
        return best, best_sim, len(candidates)

    def get(self, query: str, cluster_id: int | None = None) -> LookupResult:
        """Serve ``query``, from cache if possible and from the oracle if not.

        ``cluster_id`` is the trace's ground-truth label for this query. It is
        recorded and used to score false hits, never to decide the match.
        """
        now = time.time()

        # Fast path: byte-identical repeat.
        key = self._exact.get(query)
        if key is not None:
            entry = self._entries[key]
            self._register_hit(entry, now)
            self.stats.exact_hits += 1
            return LookupResult(
                query=query,
                response=entry.response,
                hit=True,
                similarity=1.0,
                entry_key=entry.key,
                false_hit=self._is_false_hit(entry, cluster_id),
            )

        vector = self._as_vector(query)
        entry, sim, examined = self._best_match(vector)

        if entry is not None and sim >= self.tau:
            self._register_hit(entry, now)
            false_hit = self._is_false_hit(entry, cluster_id)
            if false_hit:
                self.stats.false_hits += 1
            return LookupResult(
                query=query,
                response=entry.response,
                hit=True,
                similarity=sim,
                entry_key=entry.key,
                false_hit=false_hit,
                candidates_examined=examined,
            )

        # Miss: pay the oracle, then admit the result.
        self.stats.misses += 1
        response = self.oracle.answer(query)
        self._admit(query, vector, response, cluster_id, now)
        return LookupResult(
            query=query,
            response=response,
            hit=False,
            similarity=sim if entry is not None else None,
            candidates_examined=examined,
        )

    def _as_vector(self, query: str) -> np.ndarray:
        vector = np.asarray(self.embed_fn(query), dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"embed_fn must return a 1-D vector, got shape {vector.shape}")
        return vector

    def _register_hit(self, entry: CacheEntry, now: float) -> None:
        entry.hits += 1
        entry.last_access = now
        self.stats.hits += 1
        self.policy.on_hit(entry)

    @staticmethod
    def _is_false_hit(entry: CacheEntry, cluster_id: int | None) -> bool:
        if cluster_id is None or entry.cluster_id is None:
            return False
        return entry.cluster_id != cluster_id

    # ------------------------------------------------------------- admission

    def _admit(
        self,
        query: str,
        vector: np.ndarray,
        response: OracleResponse,
        cluster_id: int | None,
        now: float,
    ) -> CacheEntry:
        if len(self._entries) >= self.capacity:
            self.evict()

        entry = CacheEntry(
            key=self._next_key,
            query=query,
            vector=vector,
            response=response,
            cluster_id=cluster_id,
            inserted_at=now,
            last_access=now,
        )
        self._next_key += 1

        self._entries[entry.key] = entry
        self._exact[query] = entry.key
        self._index.insert(entry.key, vector)
        self._link_neighbours(entry)

        self.policy.on_insert(entry)
        self.stats.insertions += 1
        return entry

    def _link_neighbours(self, entry: CacheEntry) -> None:
        """Add this entry's redundancy contribution to and from its neighbours.

        Only pairs closer than ``redundancy_tau`` contribute, so an entry with
        no close neighbours carries zero redundancy and is strongly protected
        under a coverage-aware policy.
        """
        for key in self._index.probe(entry.vector):
            if key == entry.key:
                continue
            other = self._entries.get(key)
            if other is None:
                continue
            delta = float(np.dot(entry.vector, other.vector)) - self.redundancy_tau
            if delta <= 0:
                continue
            entry.neighbours[key] = delta
            other.neighbours[entry.key] = delta
            entry.redundancy += delta
            other.redundancy += delta
            self.policy.on_score_change(other)

    def _unlink_neighbours(self, entry: CacheEntry) -> None:
        """Reverse the contributions recorded at insertion, without recomputing."""
        for key, delta in entry.neighbours.items():
            other = self._entries.get(key)
            if other is None:
                continue
            other.redundancy -= delta
            other.neighbours.pop(entry.key, None)
            self.policy.on_score_change(other)
        entry.neighbours.clear()

    # -------------------------------------------------------------- eviction

    def evict(self) -> CacheEntry:
        """Drop the policy's chosen victim and return it."""
        victim_key = self.policy.choose_victim()
        entry = self.remove(victim_key)
        self.stats.evictions += 1
        return entry

    def remove(self, key: int) -> CacheEntry:
        """Remove one entry and undo all of its bookkeeping."""
        entry = self._entries[key]
        self._unlink_neighbours(entry)
        self._index.remove(key)
        self._exact.pop(entry.query, None)
        del self._entries[key]
        self.policy.on_remove(entry)
        return entry
