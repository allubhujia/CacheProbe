"""Attack-independent leakage measurement over the tau-threshold query graph.

Attack accuracy is the headline number, but it measures one attack. A defence
tuned against this particular prober might do nothing against a better one. The
graph model in section 6.7 gives a bound that holds regardless of implementation:

  nodes  - queries in the concept space
  edges  - (u, v) whenever sim(u, v) >= tau, meaning a probe at u would reveal
           v's presence

Probing a node therefore reveals the membership status of its entire closed
neighbourhood, and the adversary's total knowledge from a budget of B probes is
a maximum-coverage problem over this graph. Connected components bound what any
adversary can ever learn: within a component, a single probe is one step from
reaching more, and no probe reaches across components at all.

Maximum coverage is NP-hard, but the objective is submodular, so the greedy
algorithm is within 1 - 1/e of optimal (Nemhauser, Wolsey & Fisher). That
factor is what makes ``greedy_coverage`` a defensible bound rather than a
heuristic guess.

Graph construction is offline analysis, never on the serving path: O(n^2 * d)
brute force, or O(n * b * d) by reusing the LSH index.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LeakageGraph:
    """Undirected tau-threshold graph over a concept space, as an adjacency list."""

    nodes: list[str]
    adjacency: dict[int, set[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for i in range(len(self.nodes)):
            self.adjacency.setdefault(i, set())

    # ---------------------------------------------------------- construction

    @classmethod
    def from_vectors(
        cls,
        nodes: list[str],
        vectors: np.ndarray,
        tau: float,
        lsh_index: object | None = None,
    ) -> "LeakageGraph":
        """Build the graph, brute force or via an LSH index.

        Passing an ``LSHIndex`` keyed by node position drops construction from
        O(n^2 * d) to roughly O(n * b * d), at the cost of missing the edges LSH
        itself misses - so the brute-force path is the one to use when the
        bound needs to be exact.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        graph = cls(nodes=list(nodes))

        if lsh_index is None:
            similarity = vectors @ vectors.T
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if similarity[i, j] >= tau:
                        graph.add_edge(i, j)
            return graph

        for i in range(len(nodes)):
            for j in lsh_index.probe(vectors[i]):  # type: ignore[attr-defined]
                if i == j:
                    continue
                if float(np.dot(vectors[i], vectors[j])) >= tau:
                    graph.add_edge(i, int(j))
        return graph

    def add_edge(self, u: int, v: int) -> None:
        self.adjacency.setdefault(u, set()).add(v)
        self.adjacency.setdefault(v, set()).add(u)

    # -------------------------------------------------------------- topology

    def __len__(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return sum(len(neighbours) for neighbours in self.adjacency.values()) // 2

    def closed_neighbourhood(self, node: int) -> set[int]:
        """What one probe at ``node`` reveals: itself plus everything within tau."""
        return {node} | self.adjacency.get(node, set())

    def connected_components(self) -> list[set[int]]:
        """Components under the tau threshold - the ceiling on any adversary.

        Breadth-first traversal, iterative so a large component cannot blow the
        recursion limit.
        """
        seen: set[int] = set()
        components: list[set[int]] = []

        for start in range(len(self.nodes)):
            if start in seen:
                continue
            component = set()
            queue = deque([start])
            seen.add(start)
            while queue:
                node = queue.popleft()
                component.add(node)
                for neighbour in self.adjacency.get(node, ()):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        queue.append(neighbour)
            components.append(component)

        return components

    def reachable_from(self, probes: list[int]) -> set[int]:
        """Union of the closed neighbourhoods of a probe set."""
        revealed: set[int] = set()
        for probe in probes:
            revealed |= self.closed_neighbourhood(probe)
        return revealed

    # -------------------------------------------------------------- coverage

    def greedy_coverage(self, budget: int) -> tuple[list[int], set[int]]:
        """Greedy maximum coverage: pick the probes revealing the most.

        Returns the chosen probes and everything they reveal. Guaranteed within
        1 - 1/e of the optimal set of the same size, which is what lets the
        result stand as a bound on adversary knowledge rather than one
        strategy's performance.
        """
        chosen: list[int] = []
        covered: set[int] = set()

        for _ in range(min(budget, len(self.nodes))):
            best_node, best_gain = None, 0
            for candidate in range(len(self.nodes)):
                if candidate in chosen:
                    continue
                gain = len(self.closed_neighbourhood(candidate) - covered)
                if gain > best_gain:
                    best_node, best_gain = candidate, gain
            if best_node is None:
                break  # nothing left to reveal
            chosen.append(best_node)
            covered |= self.closed_neighbourhood(best_node)

        return chosen, covered

    def leakage_curve(self, max_budget: int) -> list[tuple[int, float]]:
        """Fraction of the concept space revealed, per probe budget.

        The shape is the finding: a steep early rise means a handful of probes
        map most of the space, which is what a dense tau-threshold graph
        implies and what lowering tau makes worse.
        """
        chosen: list[int] = []
        covered: set[int] = set()
        curve: list[tuple[int, float]] = []

        for budget in range(1, min(max_budget, len(self.nodes)) + 1):
            best_node, best_gain = None, 0
            for candidate in range(len(self.nodes)):
                if candidate in chosen:
                    continue
                gain = len(self.closed_neighbourhood(candidate) - covered)
                if gain > best_gain:
                    best_node, best_gain = candidate, gain
            if best_node is None:
                curve.append((budget, len(covered) / len(self.nodes)))
                continue
            chosen.append(best_node)
            covered |= self.closed_neighbourhood(best_node)
            curve.append((budget, len(covered) / len(self.nodes)))

        return curve

    # --------------------------------------------------------------- summary

    def stats(self) -> dict[str, float]:
        components = self.connected_components()
        sizes = [len(c) for c in components]
        degrees = [len(self.adjacency.get(i, ())) for i in range(len(self.nodes))]
        return {
            "nodes": len(self.nodes),
            "edges": self.edge_count(),
            "components": len(components),
            "largest_component": max(sizes) if sizes else 0,
            "largest_component_fraction": (max(sizes) / len(self.nodes)) if self.nodes else 0.0,
            "mean_degree": (sum(degrees) / len(degrees)) if degrees else 0.0,
            "max_degree": max(degrees) if degrees else 0,
        }


def coverage_bound(graph: LeakageGraph, budget: int) -> float:
    """Optimistic ceiling on what ``budget`` probes could reveal.

    Sorting by degree and summing the largest closed neighbourhoods ignores
    overlap, so it over-counts - deliberately. Compared against the greedy
    curve, the gap between them shows how much redundancy the graph's structure
    forces on the adversary.
    """
    sizes = sorted(
        (len(graph.closed_neighbourhood(i)) for i in range(len(graph.nodes))),
        reverse=True,
    )
    return min(1.0, sum(sizes[:budget]) / len(graph.nodes)) if graph.nodes else 0.0


def expected_probes_for_coverage(graph: LeakageGraph, target_fraction: float) -> int:
    """Greedy probes needed to reveal ``target_fraction`` of the space.

    The "probes to success" metric of section 10.4, computed structurally
    rather than by running an attack.
    """
    if not graph.nodes:
        return 0
    target = math.ceil(target_fraction * len(graph.nodes))
    chosen, covered = [], set()

    while len(covered) < target and len(chosen) < len(graph.nodes):
        best_node, best_gain = None, 0
        for candidate in range(len(graph.nodes)):
            if candidate in chosen:
                continue
            gain = len(graph.closed_neighbourhood(candidate) - covered)
            if gain > best_gain:
                best_node, best_gain = candidate, gain
        if best_node is None:
            break
        chosen.append(best_node)
        covered |= graph.closed_neighbourhood(best_node)

    return len(chosen)
