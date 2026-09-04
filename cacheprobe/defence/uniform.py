"""D1 - uniform padding. Correct, and useless.

Every response, hit or miss, is held until a fixed deadline at or beyond the
worst-case miss latency. The hit/miss channel closes completely: observed
latency is a constant, so it carries no information about cache membership.

It also destroys the entire reason the cache exists. A 3 ms hit now costs
``T_pad``, so the cache saves money and saves no time at all.

D1 is implemented because it is the security upper bound. Every other defence
is judged by how much of D1's protection it keeps while giving back latency.
"""

from __future__ import annotations

from .base import LatencyShaper


class UniformPadding(LatencyShaper):
    """Releases every response at ``arrival + pad_s``.

    ``pad_s`` must be at least the worst-case miss latency; if a miss ever
    overruns it, that response is released late and its overrun is exactly the
    signal the padding was meant to hide.
    """

    name = "D1-uniform"

    def __init__(self, pad_s: float = 2.0) -> None:
        self.pad_s = pad_s

    def release_at(self, arrival: float, region: int | None = None) -> float:
        return arrival + self.pad_s
