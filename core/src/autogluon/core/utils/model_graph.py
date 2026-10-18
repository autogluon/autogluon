from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Any


class _NodeView:
    """Read access to a graph's nodes and their attributes: `name in nodes`, `nodes[name]`, iteration."""

    def __init__(self, attributes: dict[str, dict[str, Any]]):
        self._attributes = attributes

    def __contains__(self, name: object) -> bool:
        return name in self._attributes

    def __getitem__(self, name: str) -> dict[str, Any]:
        return self._attributes[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._attributes)

    def __len__(self) -> int:
        return len(self._attributes)


class ModelGraph:
    """Directed acyclic graph of model names with per-node attributes.

    The trainer's record of which models feed which: an edge ``base -> stacker`` means the
    stacker consumes the base model's predictions. Nodes carry a dict of attributes
    (``path``, ``type``, ``level``, ``fit_time``, ...). Insertion order of nodes is kept.
    """

    def __init__(self):
        self._attributes: dict[str, dict[str, Any]] = {}
        self._successors: dict[str, set[str]] = {}
        self._predecessors: dict[str, set[str]] = {}

    # --- nodes and edges ---

    def __contains__(self, name: object) -> bool:
        return name in self._attributes

    def __iter__(self) -> Iterator[str]:
        return iter(self._attributes)

    def __len__(self) -> int:
        return len(self._attributes)

    @property
    def nodes(self) -> _NodeView:
        return _NodeView(self._attributes)

    def add_node(self, name: str, **attributes: Any) -> None:
        """Add ``name`` with ``attributes``, or update the attributes of an existing node."""
        if name not in self._attributes:
            self._attributes[name] = {}
            self._successors[name] = set()
            self._predecessors[name] = set()
        self._attributes[name].update(attributes)

    def add_edge(self, source: str, target: str) -> None:
        """Add ``source -> target``, adding either node if it is missing."""
        self.add_node(source)
        self.add_node(target)
        self._successors[source].add(target)
        self._predecessors[target].add(source)

    def remove_node(self, name: str) -> None:
        for successor in self._successors.pop(name):
            self._predecessors[successor].discard(name)
        for predecessor in self._predecessors.pop(name):
            self._successors[predecessor].discard(name)
        del self._attributes[name]

    def remove_edges_from(self, edges: Iterable[tuple[str, str]]) -> None:
        for source, target in edges:
            self._successors[source].discard(target)
            self._predecessors[target].discard(source)

    def in_edges(self, name: str) -> list[tuple[str, str]]:
        return [(predecessor, name) for predecessor in self._predecessors[name]]

    def edges(self) -> list[tuple[str, str]]:
        return [(source, target) for source, targets in self._successors.items() for target in targets]

    def predecessors(self, name: str) -> Iterator[str]:
        return iter(self._predecessors[name])

    def successors(self, name: str) -> Iterator[str]:
        return iter(self._successors[name])

    def copy(self) -> ModelGraph:
        """An independent graph with the same nodes, edges and attribute dicts (attribute values are shared)."""
        graph = ModelGraph()
        for name, attributes in self._attributes.items():
            graph.add_node(name, **attributes)
        for source, target in self.edges():
            graph.add_edge(source, target)
        return graph

    def subgraph(self, names: Iterable[str]) -> ModelGraph:
        """An independent graph of ``names`` and the edges among them."""
        names = set(names)
        graph = ModelGraph()
        for name in self._attributes:
            if name in names:
                graph.add_node(name, **self._attributes[name])
        for source, target in self.edges():
            if source in names and target in names:
                graph.add_edge(source, target)
        return graph

    def get_node_attributes(self, attribute: str) -> dict[str, Any]:
        """``name -> value`` for every node that has ``attribute``."""
        return {
            name: attributes[attribute] for name, attributes in self._attributes.items() if attribute in attributes
        }

    # --- traversal ---

    def _reachable(self, name: str, neighbors: dict[str, set[str]]) -> list[str]:
        """Nodes reachable from ``name`` along ``neighbors`` in breadth-first order, ``name`` first."""
        seen = {name}
        order = [name]
        queue = deque([name])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(neighbors[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    order.append(neighbor)
                    queue.append(neighbor)
        return order

    def ancestors_with_self(self, name: str) -> list[str]:
        """``name`` and every node it depends on, transitively, in breadth-first order."""
        return self._reachable(name, self._predecessors)

    def ancestors(self, name: str) -> set[str]:
        return set(self.ancestors_with_self(name)) - {name}

    def descendants(self, name: str) -> set[str]:
        return set(self._reachable(name, self._successors)) - {name}

    def has_path(self, source: str, target: str) -> bool:
        return target in self._reachable(source, self._successors)

    def bfs_layers(self, sources: Iterable[str]) -> list[list[str]]:
        """Nodes grouped by their distance from the nearest of ``sources`` along successor edges, nearest first."""
        current = list(dict.fromkeys(sources))
        seen = set(current)
        layers: list[list[str]] = []
        while current:
            layers.append(current)
            following: list[str] = []
            for name in current:
                for successor in sorted(self._successors[name]):
                    if successor not in seen:
                        seen.add(successor)
                        following.append(successor)
            current = following
        return layers

    def shortest_path_lengths(self) -> dict[str, dict[str, int]]:
        """``source -> {reachable node -> number of edges}`` for every node, following successor edges."""
        lengths: dict[str, dict[str, int]] = {}
        for source in self._attributes:
            distance = {source: 0}
            queue = deque([source])
            while queue:
                current = queue.popleft()
                for successor in self._successors[current]:
                    if successor not in distance:
                        distance[successor] = distance[current] + 1
                        queue.append(successor)
            lengths[source] = distance
        return lengths

    def lexicographical_topological_sort(self) -> list[str]:
        """Nodes ordered so that every edge points forward; ties broken by name."""
        in_degree = {name: len(predecessors) for name, predecessors in self._predecessors.items()}
        ready = [name for name, degree in in_degree.items() if degree == 0]
        heapq.heapify(ready)
        order: list[str] = []
        while ready:
            name = heapq.heappop(ready)
            order.append(name)
            for successor in self._successors[name]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    heapq.heappush(ready, successor)
        if len(order) != len(self._attributes):
            raise ValueError("The model graph contains a cycle.")
        return order
