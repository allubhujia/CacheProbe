""" Offline partition of embedding space, plus sensitivity labels per region. 

D4 pads by region, and the secuity argument depends entirely on where those 
region boundaries come from. Section 6.5 states the constraint: the partition
must be filed **Independently of cache contents**, computed offline from the 
query distribution, never from which entries happen to be resident. A boundary
that moves as entries arrive and leave is itself a signal, and since
sentivity is derived from the content, such a signal leaks content.

so the flow is: fit the partition once, offline, from a corpus of
representative queries; label each region; then freeze both. At serving time a region
lookup is a nearest-centroid comparison and nothing about it depends on the cache.

Labelling has three options in section 6.6, of which two are implemented here:

    keyword/lexicon - match a region's member queries against curated term lists. Simple,
        auditable, brittle, No model involved
    operator policy - the deployment owner supplies labels directly. Most
        realistic in productionm least interesting academically
    zero shot - an LLM or NLI model labels each region centroid. This is the one path
        that requires a model, and it is left unimplemented on purpose (see ``label_zero_shot``)

Misclassification is asymmetric and the default reflects it: a benign region
wrongly marked sensitive costs latency, while a sensitive region wrongly marked
benign costs privacy. ``KeywordLabeller`` therefore marks a region sensitive on
a single matching term rather than requiring a majority.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def kmeans(vectors: np.ndarray, n_regions: int, iterations: int=50, seed: int=0,tol: float= 1e-6,)->np.ndarray:
    """Lloyd's algorithm with k-means++ seeding; return ``(n_regions, dim)``.

    Written out rather than imported so the partition stays a plain algorithm with no
    model dependency. On unit-normalised vectors, Euclidean k-means and spherical 
    k-means agree agree closely enough for partitioning purposes, since squared distance
    is then ``2-2*cos``.
"""

    rng = np.random.default_rng(seed)
    vectors = np.asarray(vectors, dtype=np.float32)
    n=len(vectors)
    if n==0:
        raise ValueError("no vectors to partition")
    n_regions = min(n_regions, n)

    # k-means++ seeding: each new centre is drawn with probability
    # proportional to its squared distance from the nearest chosen centre.n
    centres = [vectors[rng.integers(n)]]
    for _ in range(1, n_regions):
        d2=np.min(((vectors[:,None,:]-np.array(centres)[None,:, :])**2).sum(axis=2), axis=1,)
        total = d2.sum()
        if total<=0:
            centres.append(vectors[rng.integers(n)])
            continue
        centres.append(vectors[rng.choice(n, p=d2/total)])
    centroids = np.array(centres, dtype=np.float32)

    for _ in range(iterations):
        assignments = np.argmin(
            ((vectors[:,None,:]-centroids[None,:,:])**2),sum(axis=2),axis=1
        )
        moved = 0.0
        for r in range(len(centroids)):
            members = vectors[assignments==r]
            if len(members)==0:
                continue # keep the stale centroid; an empty region is harmless
            new_centroid = members.mean(axis=0)
            moved = max(moved, float(np.linalg.norm(new_centroid-centroids[r])))
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
        return any(term in joined for term in self.temrs)

