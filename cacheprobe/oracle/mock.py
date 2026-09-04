"""Mocked LLM backend: a fixed sleep, no network, no spend.

Every experimental number in the project comes from this oracle. Section 12.3
requires it: a real API call carries latency variance of its own, and that
variance would land directly on top of the hit/miss timing signal Contribution
B exists to measure. Mocking removes cost, removes variance, and makes runs
reproducible.

Default latency. Section 6.2 puts a real cache miss at 500-2000 ms against a
3-10 ms hit, and sections 6.5 and 11.2 both reason about the resulting gap as
"~800 ms separation". ``DEFAULT_MISS_LATENCY_S`` is set to match that figure so
the measured separation lines up with the analysis the document already
contains, rather than introducing a third number.
"""

from __future__ import annotations

import time

from .base import Oracle, OracleResponse

DEFAULT_MISS_LATENCY_S = 0.8


class MockOracle(Oracle):
    """Answers after a fixed delay, deterministically.

    The response text is derived from the query so that a wrong-entry cache hit
    is visible in the output during debugging, but it carries no real content -
    correctness of *answers* is not what the experiments measure, only which
    entry was served.
    """

    def __init__(
        self,
        latency_s: float = DEFAULT_MISS_LATENCY_S,
        cost_per_call: float = 0.0,
        sleep: bool = True,
    ) -> None:
        super().__init__(cost_per_call=cost_per_call)
        self.latency_s = latency_s
        self.sleep = sleep

    def answer(self, query: str) -> OracleResponse:
        started = time.perf_counter()
        if self.sleep:
            time.sleep(self.latency_s)
        elapsed = time.perf_counter() - started
        self.stats.record(latency_s=elapsed, cost=self.cost_per_call)
        return OracleResponse(text=f"[mock answer] {query}")
