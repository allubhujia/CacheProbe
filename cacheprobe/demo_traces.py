"""Medical topic groups, split into victim and adversary phrasings.

Pipeline 2, demo only.

The split is the point of this module, not an incidental detail. Section 10.2
step 5 calls it essential: if the adversary probes with a string a victim
actually submitted, the demo proves only that an exact-match cache works and
says nothing whatever about the semantic claim. Contribution B rests on *a
paraphrase being enough*, so the attacker must be forced to use words the
victim never used.

``split_topics`` therefore partitions each topic's phrasings into two disjoint
sets before anything is served, and the victim trace and adversary probe list
are drawn from different halves by construction. There is no code path that
lets them mix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DEFAULT_TOPICS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "medical" / "topics.json"
)


@dataclass(frozen=True)
class Topic:
    """One medical concept and the ways of asking about it, already split."""

    topic_id: int
    name: str
    victim_phrasings: tuple[str, ...]
    adversary_phrasings: tuple[str, ...]
    sensitive: bool = True

    def __post_init__(self) -> None:
        overlap = set(self.victim_phrasings) & set(self.adversary_phrasings)
        if overlap:
            raise ValueError(
                f"topic {self.name!r} leaks phrasings across the split: {sorted(overlap)}. "
                f"An adversary reusing a victim string invalidates the experiment."
            )


def load_topics(path: Path | str | None = None) -> list[dict]:
    """Read raw topic definitions.

    Expected shape: a list of ``{"name": str, "phrasings": [str, ...],
    "sensitive": bool}``. Kept separate from ``split_topics`` so the split is
    always an explicit, seeded step rather than something a data file decides.
    """
    topics_path = Path(path) if path is not None else _DEFAULT_TOPICS_PATH
    if not topics_path.exists():
        raise FileNotFoundError(
            f"topic definitions not found at {topics_path}. Expected a JSON list of "
            f'{{"name", "phrasings", "sensitive"}} objects.'
        )
    return json.loads(topics_path.read_text(encoding="utf-8"))


def split_topics(
    raw_topics: list[dict],
    adversary_fraction: float = 0.5,
    seed: int = 0,
    min_each: int = 1,
) -> list[Topic]:
    """Partition each topic's phrasings into disjoint victim and adversary sets.

    Topics with too few phrasings to give both sides at least ``min_each`` are
    dropped rather than shared, because sharing even one phrasing would let the
    adversary land an exact-match hit and quietly inflate the attack's apparent
    success.
    """
    rng = np.random.default_rng(seed)
    topics: list[Topic] = []

    for raw in raw_topics:
        phrasings = list(dict.fromkeys(raw.get("phrasings", [])))
        if len(phrasings) < 2 * min_each:
            continue

        order = rng.permutation(len(phrasings))
        n_adversary = max(min_each, int(round(len(phrasings) * adversary_fraction)))
        n_adversary = min(n_adversary, len(phrasings) - min_each)

        adversary = tuple(phrasings[i] for i in order[:n_adversary])
        victim = tuple(phrasings[i] for i in order[n_adversary:])

        topics.append(
            Topic(
                topic_id=len(topics),
                name=raw["name"],
                victim_phrasings=victim,
                adversary_phrasings=adversary,
                sensitive=bool(raw.get("sensitive", True)),
            )
        )

    return topics


def victim_trace(
    topics: list[Topic],
    length: int,
    zipf_exponent: float = 1.0,
    seed: int = 0,
) -> list[tuple[str, int]]:
    """The stream the demo's "users" submit. Returns (query, topic_id) pairs.

    Same Zipf-over-shuffled-ranks construction as ``traces.py``, for the same
    reason: topic popularity must not be tied to how many phrasings a topic
    happens to have.
    """
    if not topics:
        raise ValueError("no topics to sample from")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(topics))
    ranks = np.arange(1, len(topics) + 1, dtype=np.float64)
    weights = ranks**-zipf_exponent
    weights /= weights.sum()

    trace = []
    for rank in rng.choice(len(topics), size=length, p=weights):
        topic = topics[order[rank]]
        phrasing = topic.victim_phrasings[rng.integers(len(topic.victim_phrasings))]
        trace.append((phrasing, topic.topic_id))
    return trace


def adversary_probes(topics: list[Topic], seed: int = 0) -> dict[str, str]:
    """One canonical probe phrasing per topic, drawn only from the held-out half.

    This is what feeds Procedures 1 and 3 - ``{topic name: probe query}``.
    """
    rng = np.random.default_rng(seed)
    return {
        topic.name: topic.adversary_phrasings[rng.integers(len(topic.adversary_phrasings))]
        for topic in topics
        if topic.adversary_phrasings
    }


def assert_disjoint(topics: list[Topic]) -> None:
    """Fail loudly if any phrasing appears on both sides, anywhere.

    Cheap to run before an attack experiment, and worth running: a single
    shared string turns a semantic-matching result into an exact-matching one
    without changing anything visible in the output.
    """
    victim_strings = {p for t in topics for p in t.victim_phrasings}
    adversary_strings = {p for t in topics for p in t.adversary_phrasings}
    shared = victim_strings & adversary_strings
    if shared:
        raise AssertionError(
            f"{len(shared)} phrasing(s) appear in both victim and adversary sets: "
            f"{sorted(shared)[:5]}"
        )
