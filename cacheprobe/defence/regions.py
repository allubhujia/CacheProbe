"""Offline partition of embedding space, plus sensitivity labels per region.

D4 pads by region, and the security argument depends entirely on where those
region boundaries come from. Section 6.5 states the constraint: the partition
must be fixed **independently of cache contents**, computed offline from the
query distribution, never from which entries happen to be resident. A boundary
that moves as entries arrive and leave is itself a signal, and since
sensitivity is derived from content, such a signal leaks content.

So the flow is: fit the partition once, offline, from a corpus of
representative queries; label each region; then freeze both. At serving time a
region lookup is a nearest-centroid comparison and nothing about it depends on
the cache.

Labelling has three options in section 6.6, of which two are implemented here:

  keyword/lexicon  - match a region's member queries against curated term
      lists. Simple, auditable, brittle. No model involved.
  operator policy  - the deployment owner supplies labels directly. Most
      realistic in production, least interesting academically.
  zero-shot        - an LLM or NLI model labels each region centroid. This is
      the one path that requires a model, and it is left unimplemented on
      purpose (see ``label_zero_shot``).

Misclassification is asymmetric and the default reflects it: a benign region
wrongly marked sensitive costs latency, while a sensitive region wrongly marked
benign costs privacy. ``KeywordLabeller`` therefore marks a region sensitive on
a single matching term rather than requiring a majority.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def kmeans(
    vectors: np.ndarray,
    n_regions: int,
    iterations: int = 50,
    seed: int = 0,
    tol: float = 1e-6,
) -> np.ndarray:
    """Lloyd's algorithm with k-means++ seeding; returns ``(n_regions, dim)``.

    Written out rather than imported so the partition stays a plain algorithm
    with no model dependency. On unit-normalised vectors, Euclidean k-means and
    spherical k-means agree closely enough for partitioning purposes, since
    squared distance is then ``2 - 2*cos``.
    """
    rng = np.random.default_rng(seed)
    vectors = np.asarray(vectors, dtype=np.float32)
    n = len(vectors)
    if n == 0:
        raise ValueError("no vectors to partition")
    n_regions = min(n_regions, n)

    # k-means++ seeding: each new centre is drawn with probability
    # proportional to its squared distance from the nearest chosen centre.
    centres = [vectors[rng.integers(n)]]
    for _ in range(1, n_regions):
        d2 = np.min(
            ((vectors[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(axis=2),
            axis=1,
        )
        total = d2.sum()
        if total <= 0:
            centres.append(vectors[rng.integers(n)])
            continue
        centres.append(vectors[rng.choice(n, p=d2 / total)])
    centroids = np.array(centres, dtype=np.float32)

    for _ in range(iterations):
        assignments = np.argmin(
            ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1
        )
        moved = 0.0
        for r in range(len(centroids)):
            members = vectors[assignments == r]
            if len(members) == 0:
                continue  # keep the stale centroid; an empty region is harmless
            new_centroid = members.mean(axis=0)
            moved = max(moved, float(np.linalg.norm(new_centroid - centroids[r])))
            centroids[r] = new_centroid
        if moved < tol:
            break

    return centroids


@dataclass(frozen=True)
class Region:
    region_id: int
    centroid: np.ndarray
    sensitive: bool
    #: Padding deadline for this region, in seconds from request arrival.
    pad_s: float = 0.0


class KeywordLabeller:
    """Marks a region sensitive if any member query contains a listed term.

    Deliberately trigger-happy, per the asymmetry in section 6.6: the cost of a
    false positive is latency, the cost of a false negative is disclosure.
    """

    def __init__(self, terms: list[str]) -> None:
        self.terms = [t.lower() for t in terms]

    def is_sensitive(self, queries: list[str]) -> bool:
        joined = " ".join(queries).lower()
        return any(term in joined for term in self.terms)


class RegionPartition:
    """A frozen partition of embedding space with a sensitivity label per region.

    Built once by ``fit`` and then read-only. ``region_of`` is the O(1)-ish
    serving-path lookup - a single matrix product against the centroids -
    matching the "region lookup for padding: O(1)" row in section 8.
    """

    def __init__(self, regions: list[Region]) -> None:
        self.regions = regions
        self._centroids = np.stack([r.centroid for r in regions])
        self._sensitive = np.array([r.sensitive for r in regions])

    # ------------------------------------------------------------------- fit

    @classmethod
    def fit(
        cls,
        vectors: np.ndarray,
        queries: list[str],
        labeller: KeywordLabeller,
        n_regions: int = 16,
        pad_s: float = 2.0,
        seed: int = 0,
    ) -> "RegionPartition":
        """Cluster a representative query sample and label the resulting regions.

        ``vectors`` and ``queries`` must be a *sample of the query
        distribution*, not the cache's current contents - that distinction is
        the whole security argument.
        """
        if len(vectors) != len(queries):
            raise ValueError("vectors and queries must be the same length")

        centroids = kmeans(vectors, n_regions=n_regions, seed=seed)
        assignments = cls._assign(centroids, np.asarray(vectors, dtype=np.float32))

        regions = []
        for r in range(len(centroids)):
            members = [queries[i] for i in range(len(queries)) if assignments[i] == r]
            sensitive = labeller.is_sensitive(members) if members else False
            regions.append(
                Region(
                    region_id=r,
                    centroid=centroids[r],
                    sensitive=sensitive,
                    pad_s=pad_s if sensitive else 0.0,
                )
            )
        return cls(regions)

    @staticmethod
    def _assign(centroids: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        return np.argmin(
            ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1
        )

    @classmethod
    def from_operator_policy(
        cls,
        centroids: np.ndarray,
        sensitive_ids: set[int],
        pad_s: float = 2.0,
    ) -> "RegionPartition":
        """Option 3 of section 6.6: the deployment owner supplies the labels."""
        return cls(
            [
                Region(
                    region_id=r,
                    centroid=centroids[r],
                    sensitive=(r in sensitive_ids),
                    pad_s=pad_s if r in sensitive_ids else 0.0,
                )
                for r in range(len(centroids))
            ]
        )

    # --------------------------------------------------------------- serving

    def __len__(self) -> int:
        return len(self.regions)

    def region_of(self, vector: np.ndarray) -> int:
        """Nearest centroid to ``vector``. Never consults cache state."""
        v = np.asarray(vector, dtype=np.float32)
        return int(np.argmin(((self._centroids - v) ** 2).sum(axis=1)))

    def is_sensitive(self, region_id: int) -> bool:
        return bool(self._sensitive[region_id])

    def pad_for(self, region_id: int) -> float:
        return self.regions[region_id].pad_s

    def sensitive_fraction(self) -> float:
        """Share of regions marked sensitive; drives D4's cost curve."""
        return float(self._sensitive.mean()) if len(self._sensitive) else 0.0


def label_zero_shot(*_args: object, **_kwargs: object) -> None:
    """Option 2 of section 6.6 - requires a model, so not implemented here.

    A zero-shot classifier (an LLM or an NLI model) would label each region's
    centroid text without training data, at one inference per region. That cost
    is offline and therefore acceptable, but it is the only labelling path that
    introduces a model dependency, so it is left as an explicit decision rather
    than smuggled in behind this module's otherwise model-free interface.
    """
    raise NotImplementedError(
        "zero-shot region labelling needs a model; use KeywordLabeller or "
        "RegionPartition.from_operator_policy, or implement this deliberately"
    )
