"""The expensive backend the cache sits in front of.

Section 2 frames the problem around "an expensive oracle mapping a query to a
response at fixed cost C and latency L". Everything the cache is trying to
avoid happens behind this interface, so it is deliberately narrow: a query goes
in, an answer comes out, and the oracle keeps count of what that cost.

Two implementations exist and they must never be confused:

  MockOracle (mock.py) - a fixed sleep, no network, no spend. Every reported
      experimental number comes from this one, because a real API's latency
      variance would land directly on top of the hit/miss timing signal that
      Contribution B is trying to measure.

  RAGOracle (rag.py) - retrieval over the medical corpus plus a real LLM call.
      Demo only.

Keeping them behind one interface means ``cache.py`` never learns which is
plugged in, and an experiment cannot accidentally bill a real API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class OracleResponse:
    """One answer from the backend.

    ``sources`` is empty for the mock and carries retrieved-chunk attribution
    for the RAG oracle, which the demo's chat pane displays so answers are
    traceable to a document rather than taken on trust.
    """

    text: str
    sources: tuple[str, ...] = ()


@dataclass
class OracleStats:
    """Running totals, for the cost-saved metric in section 10.4."""

    calls: int = 0
    total_latency_s: float = 0.0
    total_cost: float = 0.0

    def record(self, latency_s: float, cost: float) -> None:
        self.calls += 1
        self.total_latency_s += latency_s
        self.total_cost += cost

    def reset(self) -> None:
        self.calls = 0
        self.total_latency_s = 0.0
        self.total_cost = 0.0


class Oracle(ABC):
    """Interface for anything the cache can fall back to on a miss.

    ``cost_per_call`` is left at zero by default rather than carrying a
    hard-coded price, because published API pricing changes and a stale
    constant buried in the source would quietly corrupt the cost-saved figures.
    Set it explicitly from current pricing when running the cost experiments.
    """

    def __init__(self, cost_per_call: float = 0.0) -> None:
        self.cost_per_call = cost_per_call
        self.stats = OracleStats()

    @abstractmethod
    def answer(self, query: str) -> OracleResponse:
        """Produce an answer, paying this backend's full latency and cost."""

    def __call__(self, query: str) -> OracleResponse:
        return self.answer(query)
