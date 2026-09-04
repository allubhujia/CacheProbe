"""Sweep harness: run one configuration, or a grid of them, and collect metrics.

Every policy runs through identical cache machinery (section 12.2), so the
harness's job is to vary exactly one thing at a time and record the result.

Two rules the harness enforces, because getting either wrong invalidates the
comparison:

  Embeddings are computed once and shared. They are identical across policies,
  and re-embedding per configuration would dominate runtime while adding
  nothing (section 12.3).

  The oracle never sleeps during eviction sweeps. Hit-rate results do not
  depend on wall-clock latency, and a 0.8 s sleep per miss would turn a sweep
  into an overnight job. Timing experiments re-enable it deliberately.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .cache import SemanticCache
from .oracle.mock import MockOracle
from .policies.base import EvictionPolicy
from .policies.coverage import CoveragePolicy
from .policies.fifo import FIFOPolicy
from .policies.lfu import LFUPolicy
from .policies.lru import LRUPolicy
from .policies.random import RandomPolicy
from .traces import TraceItem

PolicyFactory = Callable[[], EvictionPolicy]

#: The baselines of section 10.3. "coverage-ablation" is the important one -
#: frequency plus recency with the redundancy term switched off. If it matches
#: the full policy, the redundancy term contributes nothing and that must be
#: reported rather than buried.
POLICIES: dict[str, PolicyFactory] = {
    "lru": LRUPolicy,
    "lfu": LFUPolicy,
    "fifo": FIFOPolicy,
    "random": RandomPolicy,
    "coverage-ablation": lambda: CoveragePolicy(w3=0.0),
    "coverage": CoveragePolicy,
}


@dataclass
class RunResult:
    """Metrics from one cache configuration over one trace."""

    policy: str
    capacity: int
    tau: float
    trace_length: int

    hits: int = 0
    misses: int = 0
    false_hits: int = 0
    evictions: int = 0

    hit_rate: float = 0.0
    false_hit_rate: float = 0.0
    coverage: float = 0.0

    mean_evict_us: float = 0.0
    mean_lookup_us: float = 0.0
    oracle_calls: int = 0
    cost: float = 0.0

    extra: dict = field(default_factory=dict)


def precompute_vectors(
    trace: Sequence[TraceItem],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> dict[str, np.ndarray]:
    """Embed every distinct query in the trace once."""
    unique = list(dict.fromkeys(item.query for item in trace))
    vectors = np.asarray(embed_fn(unique), dtype=np.float32)
    return {query: vectors[i] for i, query in enumerate(unique)}


def run_trace(
    trace: Sequence[TraceItem],
    vectors: dict[str, np.ndarray],
    policy_name: str,
    capacity: int,
    tau: float = 0.85,
    dim: int = 384,
    lsh_k: int = 12,
    lsh_L: int = 8,
    seed: int = 0,
    cost_per_call: float = 0.0,
    sleep_on_miss: bool = False,
) -> RunResult:
    """Replay ``trace`` through one cache configuration."""
    if policy_name not in POLICIES:
        raise KeyError(f"unknown policy {policy_name!r}; have {sorted(POLICIES)}")

    oracle = MockOracle(cost_per_call=cost_per_call, sleep=sleep_on_miss)
    policy = POLICIES[policy_name]()
    cache = SemanticCache(
        embed_fn=lambda q: vectors[q],
        oracle=oracle,
        policy=policy,
        capacity=capacity,
        tau=tau,
        dim=dim,
        lsh_k=lsh_k,
        lsh_L=lsh_L,
        seed=seed,
    )

    lookup_times: list[float] = []
    evictions_before = 0
    evict_times: list[float] = []

    for item in trace:
        started = time.perf_counter()
        cache.get(item.query, cluster_id=item.cluster_id)
        elapsed = time.perf_counter() - started

        # An iteration that evicted attributes its whole cost to eviction; one
        # that did not is a pure lookup. Crude, but it isolates the eviction
        # overhead section 10.4 asks to be measured in microseconds.
        if cache.stats.evictions > evictions_before:
            evict_times.append(elapsed)
            evictions_before = cache.stats.evictions
        else:
            lookup_times.append(elapsed)

    stats = cache.stats
    return RunResult(
        policy=policy_name,
        capacity=capacity,
        tau=tau,
        trace_length=len(trace),
        hits=stats.hits,
        misses=stats.misses,
        false_hits=stats.false_hits,
        evictions=stats.evictions,
        hit_rate=stats.hit_rate,
        false_hit_rate=stats.false_hit_rate,
        coverage=coverage_of(cache, vectors, trace),
        mean_evict_us=1e6 * (sum(evict_times) / len(evict_times)) if evict_times else 0.0,
        mean_lookup_us=1e6 * (sum(lookup_times) / len(lookup_times)) if lookup_times else 0.0,
        oracle_calls=oracle.stats.calls,
        cost=oracle.stats.total_cost,
    )


def coverage_of(
    cache: SemanticCache,
    vectors: dict[str, np.ndarray],
    trace: Sequence[TraceItem],
) -> float:
    """Share of the trace's distinct queries within ``tau`` of some resident entry.

    The coverage metric of section 10.4, measured against the final cache state
    rather than averaged over time - a snapshot of how much of the query
    distribution the cache can currently serve.
    """
    entries = cache.entries()
    if not entries:
        return 0.0

    matrix = np.stack([e.vector for e in entries])
    distinct = list(dict.fromkeys(item.query for item in trace))
    covered = sum(
        1 for query in distinct if float(np.max(matrix @ vectors[query])) >= cache.tau
    )
    return covered / len(distinct)


def sweep(
    trace: Sequence[TraceItem],
    vectors: dict[str, np.ndarray],
    policies: Iterable[str] = tuple(POLICIES),
    capacities: Iterable[int] = (100, 250, 500, 1000, 2000),
    taus: Iterable[float] = (0.85,),
    **kwargs: object,
) -> list[RunResult]:
    """Cross policies with capacities and thresholds. Figures 1 and 3 come from this."""
    results = []
    for tau in taus:
        for capacity in capacities:
            for policy in policies:
                results.append(
                    run_trace(
                        trace,
                        vectors,
                        policy_name=policy,
                        capacity=capacity,
                        tau=tau,
                        **kwargs,  # type: ignore[arg-type]
                    )
                )
    return results


def save_results(results: Sequence[RunResult], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    return out


def load_results(path: Path | str) -> list[RunResult]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RunResult(**row) for row in payload]


def summarise(results: Sequence[RunResult]) -> str:
    lines = [f"{'policy':<20}{'cap':>6}{'tau':>6}{'hit%':>8}{'false%':>8}{'evict_us':>10}"]
    for r in sorted(results, key=lambda x: (x.tau, x.capacity, x.policy)):
        lines.append(
            f"{r.policy:<20}{r.capacity:>6}{r.tau:>6.2f}"
            f"{100 * r.hit_rate:>8.2f}{100 * r.false_hit_rate:>8.2f}{r.mean_evict_us:>10.1f}"
        )
    return "\n".join(lines)
