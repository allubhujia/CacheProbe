"""D3 - quantised latency buckets. Effective, expensive.

Every response is rounded up to the next rung of a fixed ladder. A 3 ms hit and
a 150 ms fast miss both land on the 200 ms rung and become indistinguishable;
residual leakage survives only where a hit and a miss straddle a rung boundary.

The cost falls entirely on hits: a 3 ms response now takes 200 ms, so the
cache's latency benefit is bought back at the price of the bucket floor. The
ladder is the tunable knob, and sweeping it traces a security/latency curve
that D4 is expected to dominate.

The ladder must be fixed in advance and identical for all traffic. Choosing
rungs from observed latencies would make the ladder itself depend on cache
contents, which is the same self-defeating mistake section 6.5 identifies for
per-entry padding.
"""

from __future__ import annotations

import bisect
import time

from .base import LatencyShaper

DEFAULT_LADDER = (0.2, 0.9, 2.0)


class QuantisedBuckets(LatencyShaper):
    """Rounds elapsed time up to the next fixed rung.

    Anything exceeding the top rung is released as soon as it is ready, which
    is a real if narrow leak: an unusually slow miss announces itself by
    overrunning the ladder. Setting the top rung above the worst-case miss
    latency closes it and makes this defence degenerate to D1.
    """

    name = "D3-quantised"

    def __init__(self, ladder: tuple[float, ...] = DEFAULT_LADDER) -> None:
        if not ladder or list(ladder) != sorted(ladder):
            raise ValueError("ladder must be non-empty and ascending")
        self.ladder = tuple(ladder)

    def bucket_for(self, elapsed_s: float) -> float:
        """The rung an observed elapsed time rounds up to."""
        i = bisect.bisect_left(self.ladder, elapsed_s)
        if i == len(self.ladder):
            return elapsed_s  # overruns the ladder; released immediately
        return self.ladder[i]

    def release_at(self, arrival: float, region: int | None = None) -> float:
        elapsed = time.perf_counter() - arrival
        return arrival + self.bucket_for(elapsed)
