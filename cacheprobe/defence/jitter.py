"""D2 - randomised jitter. The intuitive defence, and it fails.

A random delay is added to every response. It looks like it should work: any
single measurement is now noisy, so a single probe no longer cleanly reveals a
hit.

It does not survive repetition. The adversary submits the same probe r times
and averages. The standard error of that mean shrinks as sigma/sqrt(r), so the
noise the defence adds is beaten by patience alone, while the hit/miss
separation stays exactly where it was. With a ~800 ms gap and ~200 ms of
jitter, a handful of repetitions suffice.

D2 is included because its failure is instructive and, more usefully,
*predictable*: ``repetitions_to_distinguish`` gives the closed form, and
section 10.6's fourth figure plots it against the measured repetition count.
Matching the two is the result.
"""

from __future__ import annotations

import math
import random
import time

from .base import LatencyShaper


def repetitions_to_distinguish(
    separation_s: float,
    jitter_std_s: float,
    z: float = 1.96,
) -> int:
    """Repetitions needed before averaged probes separate hits from misses.

    Two sample means, each of r draws with standard deviation sigma, are
    distinguishable at confidence ``z`` once the separation exceeds the
    combined standard error:

        separation >= z * sigma * sqrt(2/r)   =>   r >= 2 * (z*sigma/separation)^2

    The quadratic in sigma is the defence's problem: doubling the jitter buys
    only a fourfold increase in the adversary's effort, and every millisecond
    of it is paid by every legitimate request forever.
    """
    if separation_s <= 0:
        return math.inf  # type: ignore[return-value]
    if jitter_std_s <= 0:
        return 1
    return max(1, math.ceil(2.0 * (z * jitter_std_s / separation_s) ** 2))


class RandomJitter(LatencyShaper):
    """Adds a random delay drawn fresh for each response.

    Note what this does *not* do: the deadline is computed from the current
    moment rather than from arrival, so the natural hit/miss difference passes
    straight through with noise layered on top. That is faithful to the defence
    as usually proposed, and is precisely why it fails.
    """

    name = "D2-jitter"

    def __init__(
        self,
        mean_s: float = 0.2,
        std_s: float = 0.05,
        distribution: str = "normal",
        seed: int = 0,
    ) -> None:
        self.mean_s = mean_s
        self.std_s = std_s
        self.distribution = distribution
        self._rng = random.Random(seed)

    def _sample(self) -> float:
        if self.distribution == "uniform":
            span = self.std_s * math.sqrt(12.0) / 2.0
            return max(0.0, self._rng.uniform(self.mean_s - span, self.mean_s + span))
        if self.distribution == "exponential":
            return self._rng.expovariate(1.0 / self.mean_s) if self.mean_s > 0 else 0.0
        return max(0.0, self._rng.gauss(self.mean_s, self.std_s))

    def release_at(self, arrival: float, region: int | None = None) -> float:
        return time.perf_counter() + self._sample()
