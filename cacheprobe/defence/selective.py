"""D4 - selective region-based padding. The proposed defence.

Leakage only matters where the leaked fact is sensitive. "What are your
visiting hours" reveals nothing worth hiding; a query about a treatment does.
So protection is applied per region of embedding space rather than globally:
sensitive regions serve at a fixed deadline, benign regions serve at full
speed, and most traffic never pays.

The critical design constraint (section 6.5). Padding must never be applied
per entry. If entry e is padded and its neighbour e' is not, an adversary
learns that e was classified sensitive - and since classification derives from
content, the padding decision itself leaks content. Padding is therefore
uniform across a whole region whose boundary comes from a fixed offline
partition (``regions.py``), not from which entries happen to be resident.

Under that construction a probe into a protected region observes the same
latency whether or not the query is cached, so the requirement

    Pr[latency | region, member] == Pr[latency | region, not member]

holds by construction: this shaper is never told whether the request hit, and
its output depends only on the arrival time and the region. Verifying it
empirically - showing the adversary's accuracy inside protected regions falls
to chance - is the central security experiment.

The cost scales with the fraction of traffic landing in protected regions. If
nearly all traffic is sensitive, D4 degenerates to D1 and offers nothing; its
value is the benign share, so the crossover point is what to report.
"""

from __future__ import annotations

from .base import LatencyShaper
from .regions import RegionPartition


class SelectivePadding(LatencyShaper):
    """Pads uniformly within protected regions; leaves the rest untouched.

    ``region`` is resolved by the caller through ``RegionPartition.region_of``
    before the response is ready, so the lookup cannot depend on the outcome of
    the cache lookup.
    """

    name = "D4-selective"

    def __init__(
        self,
        partition: RegionPartition,
        default_pad_s: float | None = None,
    ) -> None:
        self.partition = partition
        self.default_pad_s = default_pad_s

    def release_at(self, arrival: float, region: int | None = None) -> float:
        if region is None:
            # No region resolved: fail closed. Guessing "benign" here would
            # turn a lookup failure into a silent hole in the defence.
            pad = self.default_pad_s if self.default_pad_s is not None else 0.0
            return arrival + pad

        if not self.partition.is_sensitive(region):
            return arrival

        pad = self.partition.pad_for(region)
        if self.default_pad_s is not None:
            pad = max(pad, self.default_pad_s)
        return arrival + pad

    def protected_fraction(self, regions_seen: list[int]) -> float:
        """Share of observed traffic that fell in a padded region.

        This is the cost side of D4's headline figure - the mean added latency
        it imposes is this fraction times the region deadline.
        """
        if not regions_seen:
            return 0.0
        protected = sum(1 for r in regions_seen if self.partition.is_sensitive(r))
        return protected / len(regions_seen)
