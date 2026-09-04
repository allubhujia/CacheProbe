"""The adversary's measurement instrument: submit a query, time the reply.

This is the whole capability the threat model grants (section 6.1). No
credentials, no logs, no cache internals - just the public interface and a
clock.

Timing is measured at the client, over a real HTTP path, because section 12.3
is right that internal instrumentation would miss exactly the implementation
leaks that matter: queueing contention, connection reuse, garbage collection.
An in-process transport is provided too, but only for testing the analysis
code; any reported attack number should come from the HTTP path.

Classification is calibrated, not hard-coded. The adversary does not know the
server's latencies in advance, so it probes queries it knows to be absent
(random nonsense strings are reliably uncached), observes that distribution,
and puts the decision boundary below it. This is what makes the attack
practical against a deployment whose timings were never published.
"""

from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass
class Probe:
    """One timed request."""

    query: str
    latency_s: float
    at: float
    repetitions: int = 1
    samples: tuple[float, ...] = ()


class Transport(Protocol):
    """Anything that turns a query into a response, taking real time to do it."""

    def __call__(self, query: str) -> object: ...


class HTTPTransport:
    """Posts to the served endpoint and blocks until the response completes.

    The response body is read before the clock stops. A transport that returned
    on headers alone would under-measure a padded response, since padding
    delays the body.
    """

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.url = url
        self.timeout = timeout
        self._session = None

    def __call__(self, query: str) -> object:
        if self._session is None:
            import requests

            self._session = requests.Session()
        response = self._session.post(self.url, json={"query": query}, timeout=self.timeout)
        return response.content  # force the body to be read


@dataclass
class Prober:
    """Times queries and decides which were served from cache.

    ``repetitions`` is the counter to D2: averaging r independent measurements
    shrinks the jitter's standard error by sqrt(r) while leaving the hit/miss
    separation untouched. Set it to 1 against undefended or padded targets,
    and raise it when jitter is suspected.
    """

    transport: Transport
    repetitions: int = 1
    aggregate: Callable[[list[float]], float] = statistics.median
    threshold_s: float | None = None
    probes: list[Probe] = field(default_factory=list)

    #: Latencies observed for queries known to be absent, from ``calibrate``.
    control_latencies: list[float] = field(default_factory=list)

    def probe(self, query: str) -> Probe:
        samples = []
        for _ in range(max(1, self.repetitions)):
            started = time.perf_counter()
            self.transport(query)
            samples.append(time.perf_counter() - started)

        result = Probe(
            query=query,
            latency_s=self.aggregate(samples),
            at=time.time(),
            repetitions=len(samples),
            samples=tuple(samples),
        )
        self.probes.append(result)
        return result

    # ------------------------------------------------------------ calibration

    def calibrate(self, n_controls: int = 10, margin: float = 0.5) -> float:
        """Set the decision boundary using queries guaranteed to be absent.

        Random unique strings cannot be semantically close to anything a victim
        asked, so their latencies sample the miss distribution. The boundary is
        placed a fraction ``margin`` of the way down from that distribution
        toward zero, which is robust as long as hits are far faster than misses
        - and section 6.2 puts the gap at two orders of magnitude.
        """
        self.control_latencies = []
        for _ in range(n_controls):
            nonsense = f"zzq-{uuid.uuid4().hex}-unlikely-query"
            self.control_latencies.append(self.probe(nonsense).latency_s)

        miss_floor = min(self.control_latencies)
        self.threshold_s = miss_floor * (1.0 - margin)
        return self.threshold_s

    def calibrate_with_known(self, absent: list[str], present: list[str]) -> float:
        """Boundary from labelled examples, when the adversary has both.

        Used for the upper-bound evaluation: it tells us how well *any*
        threshold could do, separating the attack's difficulty from the
        calibration procedure's.
        """
        absent_latencies = [self.probe(q).latency_s for q in absent]
        present_latencies = [self.probe(q).latency_s for q in present]
        if not absent_latencies or not present_latencies:
            raise ValueError("need both absent and present examples")
        self.threshold_s = (max(present_latencies) + min(absent_latencies)) / 2.0
        return self.threshold_s

    # ---------------------------------------------------------- classification

    def looks_cached(self, probe: Probe) -> bool:
        if self.threshold_s is None:
            raise RuntimeError("call calibrate() before classifying")
        return probe.latency_s < self.threshold_s

    def separation(self) -> float:
        """Observed gap between the control floor and the fastest probe.

        Reported alongside attack accuracy: a large separation means the
        channel is wide open, and near-zero means a defence is working.
        """
        if not self.probes or not self.control_latencies:
            return 0.0
        return min(self.control_latencies) - min(p.latency_s for p in self.probes)
