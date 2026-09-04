"""Min-heaps keyed by a mutable score, for eviction and padding scheduling.

A plain binary heap breaks once scores are allowed to change after insertion
(section 5.4): a stale root can sit un-popped forever if nothing tells the heap
its neighbours' redundancy changed. Two designs address this, and both are
implemented so they can be compared:

  IndexedMinHeap - a hash map from id to array position, kept in sync on every
      swap, so ``update`` can sift an entry up or down in O(log n). Exact at
      every point in time; more bookkeeping per operation.

  LazyMinHeap - push a new node on every score change instead of relocating
      the old one. Pop discards nodes whose recorded score no longer matches
      the id's current score. Simpler and cheaper per update; the heap grows
      with every update until stale nodes are popped off, so it trades memory
      for simplicity.

Both are also used, unmodified, as the deadline scheduler for padded response
release (section 7.2) - the "score" there is just a release timestamp instead
of an eviction value.
"""

from __future__ import annotations

from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)


class IndexedMinHeap(Generic[K]):
    """Binary min-heap over (score, key) with O(log n) decrease/increase-key.

    Ties in score are broken by insertion order (a monotonically increasing
    counter), so the heap never has to compare two keys directly - keys need
    only be hashable, not orderable.
    """

    __slots__ = ("_heap", "_pos", "_counter")

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, K]] = []  # (score, tie_break, key)
        self._pos: dict[K, int] = {}
        self._counter = 0

    def __len__(self) -> int:
        return len(self._heap)

    def __contains__(self, key: K) -> bool:
        return key in self._pos

    def score_of(self, key: K) -> float:
        return self._heap[self._pos[key]][0]

    # ------------------------------------------------------------- internals

    def _swap(self, i: int, j: int) -> None:
        heap = self._heap
        heap[i], heap[j] = heap[j], heap[i]
        self._pos[heap[i][2]] = i
        self._pos[heap[j][2]] = j

    def _sift_up(self, i: int) -> None:
        heap = self._heap
        while i > 0:
            parent = (i - 1) // 2
            if heap[i] < heap[parent]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        heap = self._heap
        n = len(heap)
        while True:
            left, right = 2 * i + 1, 2 * i + 2
            smallest = i
            if left < n and heap[left] < heap[smallest]:
                smallest = left
            if right < n and heap[right] < heap[smallest]:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    # ------------------------------------------------------------------ API

    def push(self, key: K, score: float) -> None:
        """Insert a new key. Raises if the key is already present."""
        if key in self._pos:
            raise KeyError(f"{key!r} already in heap; use update()")
        self._heap.append((score, self._counter, key))
        self._counter += 1
        self._pos[key] = len(self._heap) - 1
        self._sift_up(len(self._heap) - 1)

    def update(self, key: K, score: float) -> None:
        """Change an existing key's score, moving it in O(log n)."""
        i = self._pos[key]
        old_score = self._heap[i][0]
        self._heap[i] = (score, self._heap[i][1], key)
        if score < old_score:
            self._sift_up(i)
        elif score > old_score:
            self._sift_down(i)

    def push_or_update(self, key: K, score: float) -> None:
        if key in self._pos:
            self.update(key, score)
        else:
            self.push(key, score)

    def peek(self) -> tuple[K, float]:
        """Return (key, score) of the minimum without removing it."""
        score, _, key = self._heap[0]
        return key, score

    def pop(self) -> tuple[K, float]:
        """Remove and return (key, score) of the minimum."""
        heap = self._heap
        last = heap.pop()
        if not heap:
            del self._pos[last[2]]
            return last[2], last[0]

        root = heap[0]
        del self._pos[root[2]]
        heap[0] = last
        self._pos[last[2]] = 0
        self._sift_down(0)
        return root[2], root[0]

    def remove(self, key: K) -> None:
        """Remove an arbitrary key (not just the minimum)."""
        i = self._pos[key]
        last = self._heap.pop()
        del self._pos[key]
        if i == len(self._heap):
            return  # removed key was the last element; nothing to re-seat
        self._heap[i] = last
        self._pos[last[2]] = i
        self._sift_up(i)
        self._sift_down(i)


class LazyMinHeap(Generic[K]):
    """Min-heap that never relocates entries; staleness is filtered on pop.

    ``current`` holds each key's live score. Every ``push`` adds a fresh heap
    node without touching any earlier node for that key, so an update is O(log
    n) with a small constant but leaves obsolete nodes behind. ``pop`` throws
    away nodes whose recorded score no longer matches ``current`` until it
    finds one that's still valid.
    """

    __slots__ = ("_heap", "_current", "_counter", "_stale_count")

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, K]] = []
        self._current: dict[K, float] = {}
        self._counter = 0
        self._stale_count = 0

    def __len__(self) -> int:
        """Number of *live* keys, not raw heap nodes (see ``raw_size``)."""
        return len(self._current)

    def raw_size(self) -> int:
        return len(self._heap)

    def __contains__(self, key: K) -> bool:
        return key in self._current

    def score_of(self, key: K) -> float:
        return self._current[key]

    def _push_raw(self, key: K, score: float) -> None:
        heap = self._heap
        heap.append((score, self._counter, key))
        self._counter += 1
        i = len(heap) - 1
        while i > 0:
            parent = (i - 1) // 2
            if heap[i] < heap[parent]:
                heap[i], heap[parent] = heap[parent], heap[i]
                i = parent
            else:
                break

    def push(self, key: K, score: float) -> None:
        if key in self._current:
            raise KeyError(f"{key!r} already in heap; use update()")
        self._current[key] = score
        self._push_raw(key, score)

    def update(self, key: K, score: float) -> None:
        """Record a new score and push a fresh node; the old node goes stale."""
        self._current[key]  # KeyError if absent, matching IndexedMinHeap
        self._current[key] = score
        self._push_raw(key, score)
        self._stale_count += 1

    def push_or_update(self, key: K, score: float) -> None:
        if key in self._current:
            self.update(key, score)
        else:
            self.push(key, score)

    def remove(self, key: K) -> None:
        """Mark a key gone; its heap nodes are discarded lazily on pop."""
        del self._current[key]

    def _pop_raw(self) -> tuple[float, int, K]:
        heap = self._heap
        last = heap.pop()
        if not heap:
            return last
        root = heap[0]
        heap[0] = last
        i = 0
        n = len(heap)
        while True:
            left, right = 2 * i + 1, 2 * i + 2
            smallest = i
            if left < n and heap[left] < heap[smallest]:
                smallest = left
            if right < n and heap[right] < heap[smallest]:
                smallest = right
            if smallest == i:
                break
            heap[i], heap[smallest] = heap[smallest], heap[i]
            i = smallest
        return root

    def pop(self) -> tuple[K, float]:
        """Discard stale/removed nodes off the top, then pop the true minimum."""
        while self._heap:
            score, _, key = self._pop_raw()
            live_score = self._current.get(key)
            if live_score is not None and live_score == score:
                del self._current[key]
                return key, score
            # else: key was removed, or superseded by a later update - stale
        raise IndexError("pop from empty heap")

    def peek(self) -> tuple[K, float]:
        """Return the true minimum without removing it (may discard staleness)."""
        key, score = self.pop()
        self._current[key] = score
        self._push_raw(key, score)
        return key, score

    def compact(self) -> None:
        """Rebuild the heap from only live entries, reclaiming stale memory."""
        self._heap = []
        self._counter = 0
        self._stale_count = 0
        for key, score in self._current.items():
            self._push_raw(key, score)
