"""Procedure 3 - temporal monitoring.

A single enumeration pass is a snapshot. Repeating it turns the leak into a
timeline: a concept flipping miss -> hit is the moment somebody first asked
about it, and hit -> miss is the moment it was evicted. That is strictly more
informative than any one snapshot, because it dates the activity rather than
merely listing it.

It also connects the two contributions. Eviction policy decides how long an
entry survives, so it directly sets how long a topic stays observable. Coverage
-aware eviction protects isolated entries - and isolated entries are
disproportionately the rare, unusual, sensitive ones. A better cache policy can
therefore mean a longer window in which a sensitive query is detectable.
``residency_report`` measures that window per concept, which is the quantity
section 11.3 asks for and which neither contribution produces alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .prober import Prober


@dataclass
class Observation:
    at: float
    cached: bool
    latency_s: float


@dataclass
class Transition:
    concept: str
    at: float
    #: True for miss -> hit (a topic appeared), False for hit -> miss (evicted).
    appeared: bool


@dataclass
class MonitorReport:
    timeline: dict[str, list[Observation]] = field(default_factory=dict)
    transitions: list[Transition] = field(default_factory=list)

    def appearances(self, concept: str) -> list[float]:
        return [t.at for t in self.transitions if t.concept == concept and t.appeared]

    def evictions(self, concept: str) -> list[float]:
        return [t.at for t in self.transitions if t.concept == concept and not t.appeared]

    def residency_report(self) -> dict[str, float]:
        """Mean observed time between a concept appearing and being evicted.

        Concepts still resident at the end of monitoring contribute the time
        from their last appearance to the final observation, so a policy that
        simply never evicts is not rewarded with a missing measurement.
        """
        residencies: dict[str, float] = {}
        for concept, observations in self.timeline.items():
            if not observations:
                continue
            spans: list[float] = []
            entered: float | None = None
            for observation in observations:
                if observation.cached and entered is None:
                    entered = observation.at
                elif not observation.cached and entered is not None:
                    spans.append(observation.at - entered)
                    entered = None
            if entered is not None:
                spans.append(observations[-1].at - entered)
            if spans:
                residencies[concept] = sum(spans) / len(spans)
        return residencies

    def summary(self) -> str:
        appeared = sum(1 for t in self.transitions if t.appeared)
        evicted = len(self.transitions) - appeared
        return (
            f"concepts={len(self.timeline)} rounds="
            f"{max((len(v) for v in self.timeline.values()), default=0)} "
            f"appearances={appeared} evictions={evicted}"
        )


class TemporalMonitor:
    """Probes a fixed concept set repeatedly and records state changes."""

    def __init__(self, prober: Prober) -> None:
        self.prober = prober

    def run(
        self,
        concepts: dict[str, str],
        rounds: int,
        interval_s: float = 0.0,
    ) -> MonitorReport:
        """Probe every concept once per round, ``rounds`` times.

        ``concepts`` maps concept name to the adversary's phrasing, as in
        Procedure 1 - and with the same rule: never a phrasing the victim used.
        """
        if self.prober.threshold_s is None:
            self.prober.calibrate()

        report = MonitorReport(timeline={name: [] for name in concepts})
        previous: dict[str, bool] = {}

        for round_index in range(rounds):
            if round_index and interval_s:
                time.sleep(interval_s)

            for name, phrasing in concepts.items():
                probe = self.prober.probe(phrasing)
                cached = self.prober.looks_cached(probe)
                report.timeline[name].append(
                    Observation(at=probe.at, cached=cached, latency_s=probe.latency_s)
                )

                was = previous.get(name)
                if was is not None and was != cached:
                    report.transitions.append(
                        Transition(concept=name, at=probe.at, appeared=cached)
                    )
                previous[name] = cached

        return report
