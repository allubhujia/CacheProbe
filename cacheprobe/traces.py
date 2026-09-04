"""Quora Question Pairs -> paraphrase clusters -> Zipf-skewed query trace.

This is pipeline 1, the experimental one. Every reported hit-rate and
false-hit-rate number comes from traces built here, because Quora's
human-labelled ``is_duplicate`` flags are the only ground truth available for
which queries *should* hit the cache.

Trace construction follows section 10.2 of the project document:

  1. group questions into clusters via the labelled duplicate pairs
  2. sample a cluster per step from a Zipf distribution (popularity skew)
  3. sample a paraphrase uniformly within the chosen cluster

Steps 4 (mid-trace Zipf rotation, for distribution shift) and 5 (holding out a
disjoint paraphrase set for the adversary) are not implemented yet; they are
needed by the drift and attack experiments respectively.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DEFAULT_QUORA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "quora" / "train.csv"
)

_REQUIRED_COLUMNS = {"qid1", "qid2", "question1", "question2", "is_duplicate"}


class UnionFind:
    """Disjoint-set forest with path compression and union by rank."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def add(self, x: int) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: int) -> int:
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


@dataclass(frozen=True)
class Cluster:
    """A set of questions the dataset labels as asking the same thing."""

    cluster_id: int
    questions: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.questions)


@dataclass(frozen=True)
class TraceItem:
    """One query in the stream, tagged with the cluster it came from.

    ``cluster_id`` is the ground truth a hit is scored against: a cache hit
    whose stored entry came from a different cluster is a false hit.
    """

    query: str
    cluster_id: int


def load_quora_clusters(
    csv_path: Path | str | None = None,
    min_size: int = 2,
) -> list[Cluster]:
    """Build paraphrase clusters from the Quora Question Pairs CSV.

    Expects the standard released schema, with columns ``qid1``, ``qid2``,
    ``question1``, ``question2`` and ``is_duplicate``. Only rows flagged as
    duplicates contribute edges; questions that never appear in a duplicate
    pair end up as singletons and are dropped by ``min_size``, since a cluster
    of one offers no paraphrase to sample.
    """
    path = Path(csv_path) if csv_path is not None else _DEFAULT_QUORA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Quora Question Pairs CSV not found at {path}. Download it and "
            f"place train.csv there, or pass csv_path explicitly."
        )

    uf = UnionFind()
    text_of: dict[int, str] = {}

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing expected column(s): {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            try:
                qid1, qid2 = int(row["qid1"]), int(row["qid2"])
            except (TypeError, ValueError):
                continue  # malformed row; the released CSV has a few

            text_of.setdefault(qid1, row["question1"] or "")
            text_of.setdefault(qid2, row["question2"] or "")

            if row["is_duplicate"] == "1":
                uf.union(qid1, qid2)

    grouped: dict[int, list[str]] = {}
    for qid, text in text_of.items():
        if text:
            grouped.setdefault(uf.find(qid), []).append(text)

    clusters = []
    for questions in grouped.values():
        unique = tuple(dict.fromkeys(questions))  # dedupe, keep order
        if len(unique) >= min_size:
            clusters.append(Cluster(cluster_id=len(clusters), questions=unique))
    return clusters


def zipf_weights(n: int, exponent: float) -> np.ndarray:
    """Normalised Zipf probabilities over ``n`` ranks, p(rank) ~ rank^-exponent."""
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = ranks ** -exponent
    return weights / weights.sum()


def generate_trace(
    clusters: list[Cluster],
    length: int,
    zipf_exponent: float = 1.0,
    seed: int = 0,
) -> list[TraceItem]:
    """Sample a query stream of ``length`` items from ``clusters``.

    Popularity rank is assigned by shuffling the clusters rather than by
    cluster size, so that how often a topic is asked about stays independent of
    how many paraphrases the dataset happens to contain for it. Otherwise
    popularity and paraphrase count are confounded and the eviction comparison
    cannot separate them.
    """
    if not clusters:
        raise ValueError("no clusters to sample from")

    rng = np.random.default_rng(seed)

    order = rng.permutation(len(clusters))
    probabilities = zipf_weights(len(clusters), zipf_exponent)

    picks = rng.choice(len(clusters), size=length, p=probabilities)

    trace = []
    for rank in picks:
        cluster = clusters[order[rank]]
        question = cluster.questions[rng.integers(len(cluster.questions))]
        trace.append(TraceItem(query=question, cluster_id=cluster.cluster_id))
    return trace
