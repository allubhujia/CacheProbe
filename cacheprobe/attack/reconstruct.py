"""Procedure 2 - content reconstruction by boundary search.

Once enumeration confirms a concept is cached, membership is no longer the
only thing leaking. The match region is a ball of radius ``tau`` around the
stored vector, so probing near its boundary localises that vector: every hit
says "the target is within tau of this probe", every miss says "it is not".
Intersecting the hit balls and subtracting the miss balls narrows the feasible
region, and the adversary recovers an approximation of what the victim actually
asked.

This is binary search in a metric space, which is why section 14 lists it under
searching rather than under the attack: each probe ideally halves the remaining
feasible set, and the interesting quantity is reconstruction quality as a
function of probe budget.

Variant generation is template-based on purpose. Qualifiers are added,
removed, and substituted from fixed lists - no model is involved, so the
reconstruction result measures the geometry of the leak rather than the
fluency of a paraphraser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .prober import Prober

#: Qualifier fragments spliced into probes. Short, generic, and domain-shaped:
#: the point is to move the embedding a controlled amount, not to sound natural.
DEFAULT_QUALIFIERS = (
    "in elderly patients",
    "in children",
    "long term",
    "permanent",
    "how long",
    "side effects of",
    "treatment for",
    "symptoms of",
    "is it serious",
    "at home",
)


@dataclass
class ReconstructionStep:
    query: str
    hit: bool
    #: Similarity to the true stored query, filled in by the harness for
    #: scoring. The adversary never observes this.
    true_similarity: float | None = None


@dataclass
class ReconstructionReport:
    seed_query: str
    steps: list[ReconstructionStep] = field(default_factory=list)
    best_query: str | None = None
    best_similarity: float | None = None

    @property
    def probe_budget_used(self) -> int:
        return len(self.steps)

    @property
    def hits(self) -> int:
        return sum(1 for s in self.steps if s.hit)

    def summary(self) -> str:
        sim = f"{self.best_similarity:.4f}" if self.best_similarity is not None else "n/a"
        return (
            f"probes={self.probe_budget_used} hits={self.hits} "
            f"best_similarity={sim} best={self.best_query!r}"
        )


def generate_variants(query: str, qualifiers: tuple[str, ...] = DEFAULT_QUALIFIERS) -> list[str]:
    """Template variants of ``query``: add, remove, or substitute a qualifier.

    Deterministic and finite, so a reconstruction run is reproducible and its
    probe budget is meaningful.
    """
    variants: list[str] = []
    lowered = query.lower()

    for qualifier in qualifiers:
        if qualifier in lowered:
            variants.append(lowered.replace(qualifier, "").strip())  # removal
        else:
            variants.append(f"{query} {qualifier}")  # addition
            variants.append(f"{qualifier} {query}")

    for present in qualifiers:
        if present not in lowered:
            continue
        for replacement in qualifiers:
            if replacement != present:
                variants.append(lowered.replace(present, replacement).strip())

    seen: set[str] = set()
    unique = []
    for v in variants:
        v = " ".join(v.split())
        if v and v != query and v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


class BoundarySearchAttack:
    """Refines a confirmed-cached concept toward the victim's actual phrasing.

    The greedy rule: keep probing variants of whichever query is currently the
    best-known hit. A hit means the probe is inside the ball, so it becomes the
    new centre and search continues from there; a miss prunes that direction.
    This is a hill climb over the hit region rather than an exact bisection -
    exact bisection would need distances the adversary cannot observe, only
    hit/miss.
    """

    def __init__(self, prober: Prober) -> None:
        self.prober = prober

    def run(
        self,
        seed_query: str,
        budget: int = 100,
        qualifiers: tuple[str, ...] = DEFAULT_QUALIFIERS,
        score_fn: Callable[[str], float] | None = None,
    ) -> ReconstructionReport:
        """Probe up to ``budget`` variants, expanding from each hit.

        ``score_fn`` is the harness's oracle for reconstruction quality -
        typically cosine similarity between a probe's embedding and the true
        stored vector. It is recorded for reporting only and never steers the
        search, since the adversary has no access to it.
        """
        if self.prober.threshold_s is None:
            self.prober.calibrate()

        report = ReconstructionReport(seed_query=seed_query)
        frontier = [seed_query]
        visited: set[str] = set()

        while frontier and report.probe_budget_used < budget:
            current = frontier.pop(0)
            if current in visited:
                continue
            visited.add(current)

            probe = self.prober.probe(current)
            hit = self.prober.looks_cached(probe)
            similarity = score_fn(current) if score_fn is not None else None
            report.steps.append(
                ReconstructionStep(query=current, hit=hit, true_similarity=similarity)
            )

            if hit:
                if similarity is not None and (
                    report.best_similarity is None or similarity > report.best_similarity
                ):
                    report.best_similarity = similarity
                    report.best_query = current
                elif report.best_query is None:
                    report.best_query = current

                # Inside the ball: expand from here.
                for variant in generate_variants(current, qualifiers):
                    if variant not in visited:
                        frontier.append(variant)

        return report


def reconstruction_distance(recovered: np.ndarray, true_vector: np.ndarray) -> float:
    """Embedding-space distance between the recovery and the stored query.

    Reported against probe budget (section 10.4). Cosine distance rather than
    Euclidean, to stay on the same scale as ``tau``.
    """
    a = np.asarray(recovered, dtype=np.float32)
    b = np.asarray(true_vector, dtype=np.float32)
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
