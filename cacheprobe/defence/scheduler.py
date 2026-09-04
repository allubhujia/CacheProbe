"""Deadline scheduler for padded response release.

A min-heap keyed on release timestamp (section 7.2): push on arrival, pop when
the earliest deadline expires. This is the second use of the heap structure
from ``heap.py``, with a wall-clock deadline standing in for an eviction score.

Section 7.2 warns that queueing can reintroduce the very signal padding
removes: if padded responses contend with unpadded ones for the same workers,
contention becomes a timing channel of its own. ``DeadlineScheduler`` therefore
runs its own release thread and touches nothing else.

The overhead requirement from section 8 is that scheduling and release cost
strictly less than the padding interval, or deadlines cannot be met.
``overhead_samples`` records the observed lateness of every release so that
claim is measured rather than asserted.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class _Scheduled:
    release_at: float
    sequence: int
    payload: Any = field(compare=False)
    event: threading.Event = field(compare=False, default_factory=threading.Event)


class DeadlineScheduler:
    """Holds items until their release timestamp, then hands them back.

    Usable two ways: ``wait_for`` blocks the calling thread until a deadline
    (what an HTTP handler wants), while ``submit``/``collect`` provide the
    non-blocking queue used when many padded responses are in flight at once.
    """

    def __init__(self) -> None:
        self._heap: list[_Scheduled] = []
        self._lock = threading.Lock()
        self._wakeup = threading.Condition(self._lock)
        self._sequence = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self.overhead_samples: list[float] = []

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._wakeup:
            self._running = False
            self._wakeup.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> "DeadlineScheduler":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ----------------------------------------------------------------- queue

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    def submit(self, payload: Any, release_at: float) -> _Scheduled:
        """Queue an item for release at an absolute ``perf_counter`` timestamp."""
        with self._wakeup:
            self._sequence += 1
            item = _Scheduled(release_at=release_at, sequence=self._sequence, payload=payload)
            heapq.heappush(self._heap, item)
            self._wakeup.notify_all()
        return item

    def collect(self, item: _Scheduled, timeout: float | None = None) -> Any:
        """Block until ``item`` has been released, then return its payload."""
        item.event.wait(timeout=timeout)
        return item.payload

    def _run(self) -> None:
        while True:
            with self._wakeup:
                if not self._running:
                    return
                if not self._heap:
                    self._wakeup.wait(timeout=0.05)
                    continue
                now = time.perf_counter()
                nearest = self._heap[0]
                if nearest.release_at > now:
                    self._wakeup.wait(timeout=nearest.release_at - now)
                    continue
                item = heapq.heappop(self._heap)
            self.overhead_samples.append(time.perf_counter() - item.release_at)
            item.event.set()

    # -------------------------------------------------------------- blocking

    def wait_for(self, release_at: float) -> float:
        """Sleep until ``release_at``; returns observed lateness in seconds.

        Lateness is how far past the deadline the release actually happened -
        the scheduler overhead section 8 asks to be reported as a fraction of
        the padding interval.
        """
        remaining = release_at - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        lateness = time.perf_counter() - release_at
        self.overhead_samples.append(lateness)
        return lateness

    def mean_overhead(self) -> float:
        if not self.overhead_samples:
            return 0.0
        return sum(self.overhead_samples) / len(self.overhead_samples)
