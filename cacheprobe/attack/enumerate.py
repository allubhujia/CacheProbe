"""Procedure 1 - membership enumeration.

Walk a list of concepts, submit one canonical phrasing of each, and record
which came back fast. The output is the subset of the concept list present in
the cache, which is to say: the topics other users have been asking about.

This is the procedure that makes the whole attack practical. Against an
exact-match cache the adversary would have to reproduce a victim's sentence
byte for byte, and free-text key space makes that hopeless. Against a semantic
cache any phrasing within ``tau`` matches, so the search space collapses from
strings to *concepts* - and concept lists are short and public. A few hundred
medical conditions is a complete enumeration of a health assistant's sensitive
surface.

The probes must never reuse a victim's phrasing (section 10.2, step 5). If they
do, the experiment measures exact matching and says nothing about the semantic
claim, so ``EnumerationAttack`` takes the adversary's phrasings as a separate
input and never looks at the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .prober import Probe, Prober


@dataclass
class ConceptResult:
    concept: str
    probe_query: str
    latency_s: float
    predicted_cached: bool
    actually_cached: bool | None = None

    @property
    def correct(self) -> bool | None:
        if self.actually_cached is None:
            return None
        return self.predicted_cached == self.actually_cached


@dataclass
class EnumerationReport:
    """Attack accuracy against ground truth, plus what it cost to get it."""

    results: list[ConceptResult] = field(default_factory=list)
    requests_made: int = 0
    wall_clock_s: float = 0.0

    @property
    def predicted_present(self) -> list[str]:
        return [r.concept for r in self.results if r.predicted_cached]

    def _counts(self) -> tuple[int, int, int, int]:
        tp = fp = fn = tn = 0
        for r in self.results:
            if r.actually_cached is None:
                continue
            if r.predicted_cached and r.actually_cached:
                tp += 1
            elif r.predicted_cached and not r.actually_cached:
                fp += 1
            elif not r.predicted_cached and r.actually_cached:
                fn += 1
            else:
                tn += 1
        return tp, fp, fn, tn

    @property
    def precision(self) -> float:
        tp, fp, _, _ = self._counts()
        return tp / (tp + fp) if (tp + fp) else 0.0

    @property
    def recall(self) -> float:
        tp, _, fn, _ = self._counts()
        return tp / (tp + fn) if (tp + fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        tp, fp, fn, tn = self._counts()
        total = tp + fp + fn + tn
        return (tp + tn) / total if total else 0.0

    def summary(self) -> str:
        return (
            f"precision={self.precision:.3f} recall={self.recall:.3f} "
            f"f1={self.f1:.3f} accuracy={self.accuracy:.3f} "
            f"requests={self.requests_made} time={self.wall_clock_s:.1f}s"
        )


class EnumerationAttack:
    """Probes a concept list and reports which concepts are cached."""

    def __init__(self, prober: Prober) -> None:
        self.prober = prober

    def run(
        self,
        concepts: dict[str, str],
        ground_truth: set[str] | None = None,
        calibrate: bool = True,
    ) -> EnumerationReport:
        """Probe every concept once.

        ``concepts`` maps a concept name to the adversary's canonical phrasing
        for it - one phrasing per concept, none of them taken from the victim
        trace. ``ground_truth`` is the set of concept names actually resident,
        supplied by the harness for scoring only; the attack never reads it.
        """
        import time as _time

        started = _time.perf_counter()
        before = len(self.prober.probes)

        if calibrate and self.prober.threshold_s is None:
            self.prober.calibrate()

        report = EnumerationReport()
        for concept, phrasing in concepts.items():
            probe: Probe = self.prober.probe(phrasing)
            report.results.append(
                ConceptResult(
                    concept=concept,
                    probe_query=phrasing,
                    latency_s=probe.latency_s,
                    predicted_cached=self.prober.looks_cached(probe),
                    actually_cached=(concept in ground_truth) if ground_truth is not None else None,
                )
            )

        report.requests_made = len(self.prober.probes) - before
        report.wall_clock_s = _time.perf_counter() - started
        return report
