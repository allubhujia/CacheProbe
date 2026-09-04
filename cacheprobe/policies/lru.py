"""Least-recently-used eviction, on an explicit doubly linked list.

Section 7 lists the doubly linked list as the LRU baseline's structure, so it
is written out rather than delegated to ``OrderedDict``: a node per entry, most
recent at the head, and a key-to-node map so an access can splice a node out
of the middle and move it to the front in O(1).

LRU is the policy Contribution A argues against. It assumes each entry serves
one key, so "recently used" is a sufficient proxy for value - which is exactly
the assumption a semantic cache breaks, since a recently used entry sitting in
a dense cluster may be almost entirely redundant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EvictionPolicy

if TYPE_CHECKING:
    from ..cache import CacheEntry


class _Node:
    __slots__ = ("key", "prev", "next")

    def __init__(self, key: int) -> None:
        self.key = key
        self.prev: _Node | None = None
        self.next: _Node | None = None


class LRUPolicy(EvictionPolicy):
    """Evicts the entry least recently accessed.

    Sentinel head and tail nodes remove every empty-list and end-of-list
    special case from the splice operations.
    """

    def __init__(self) -> None:
        self._head = _Node(-1)  # sentinel: most-recent side
        self._tail = _Node(-1)  # sentinel: least-recent side
        self._head.next = self._tail
        self._tail.prev = self._head
        self._nodes: dict[int, _Node] = {}

    def __len__(self) -> int:
        return len(self._nodes)

    def _unlink(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _push_front(self, node: _Node) -> None:
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _touch(self, key: int) -> None:
        node = self._nodes.get(key)
        if node is None:
            return
        self._unlink(node)
        self._push_front(node)

    def on_insert(self, entry: "CacheEntry") -> None:
        node = _Node(entry.key)
        self._nodes[entry.key] = node
        self._push_front(node)

    def on_hit(self, entry: "CacheEntry") -> None:
        self._touch(entry.key)

    def on_remove(self, entry: "CacheEntry") -> None:
        node = self._nodes.pop(entry.key, None)
        if node is not None:
            self._unlink(node)

    def choose_victim(self) -> int:
        victim = self._tail.prev
        if victim is self._head:
            raise IndexError("no entries to evict")
        return victim.key
