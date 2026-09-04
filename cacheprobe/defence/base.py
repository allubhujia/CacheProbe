"""The latency-shaping interface every defence implements.

``shape_latency`` is the hook section 5.1 leaves in the query flow, and a
no-op in the baseline system.

One deliberate omission drives the whole design: **a shaper is never told
whether the request was a cache hit.** It receives the arrival timestamp and
the query's region, and nothing else. A defence that cannot observe the secret
cannot leak it through a branch, so the property is structural rather than
something each implementation has to remember not to violate.

Section 7.2 supplies the other rule. A deadline must be computed as
``arrival + T``, an absolute offset from *request receipt* - never as
``now + T`` after processing, because a fast hit would then still be released
earlier than a slow miss and the padding would shape nothing.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod


class LatencyShaper(ABC):
    """Decides when a response is allowed to leave the server."""

    name = "base"

    @abstractmethod
    def release_at(self, arrival: float, region: int | None = None) -> float:
        """Absolute timestamp at which this response may be released.

        ``arrival`` is when the request was received, captured before any
        cache work began. Returning ``arrival`` itself means "release now".
        """

    def wait(self, arrival: float, region: int | None = None) -> float:
        """Block until the deadline. Returns the added delay in seconds."""
        deadline = self.release_at(arrival, region)
        delay = deadline - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
            return delay
        return 0.0


class NoDefence(LatencyShaper):
    """D0 - responses leave as soon as they are ready.

    Establishes the attack ceiling: with a ~800 ms gap between hit and miss,
    single-sample classification should be near perfect (section 11.2). A weak
    attack result against this shaper indicates an experimental artifact, not a
    secure system.
    """

    name = "D0-none"

    def release_at(self, arrival: float, region: int | None = None) -> float:
        return arrival
