"""
TCA Graph — Topological data structure.

A node's meaning is ENTIRELY its edge set. No embeddings. No coordinates.
7 edge types mirror the chestohedron gates. Topology IS the representation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EdgeType(Enum):
    MIRRORS = "MIRRORS"       # A parallels B (analogy)
    INHERITS = "INHERITS"     # A depends on B
    BOUNDS = "BOUNDS"         # A constrains B (power)
    EXPRESSES = "EXPRESSES"   # A produces B
    VERIFIES = "VERIFIES"     # A proves B
    REMOVES = "REMOVES"       # A contradicts B
    SEEKS = "SEEKS"           # A wants B (unproven)


@dataclass
class EdgeRelation:
    """A typed, weighted relationship to another node."""
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    grounded: bool = False

    def __post_init__(self) -> None:
        self.weight = max(0.0, min(10.0, self.weight))


@dataclass
class TopologicalNode:
    """A node whose meaning is entirely its edge set."""
    id: str
    label: str = ""
    edges: dict[str, list[EdgeRelation]] = field(default_factory=dict)
    activation: float = 0.0
    decay_rate: float = 0.1
    last_activated: float = field(default_factory=time.time)

    def add_edge(self, target_id: str, edge_type: EdgeType,
                 weight: float = 1.0, grounded: bool = False) -> EdgeRelation:
        rel = EdgeRelation(
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            grounded=grounded,
        )
        if target_id not in self.edges:
            self.edges[target_id] = []
        self.edges[target_id].append(rel)
        return rel

    def get_edges_by_type(self, edge_type: EdgeType) -> list[EdgeRelation]:
        result = []
        for rels in self.edges.values():
            for rel in rels:
                if rel.edge_type == edge_type:
                    result.append(rel)
        return result

    def get_neighbors(self) -> set[str]:
        return set(self.edges.keys())

    def edge_count(self) -> int:
        return sum(len(rels) for rels in self.edges.values())

    def edge_type_distribution(self) -> dict[EdgeType, int]:
        dist: dict[EdgeType, int] = {et: 0 for et in EdgeType}
        for rels in self.edges.values():
            for rel in rels:
                dist[rel.edge_type] += 1
        return dist


class TopologicalGraph:
    """Directed graph with 7 typed edges. The core TCA data structure."""

    def __init__(self) -> None:
        self._nodes: dict[str, TopologicalNode] = {}

    @property
    def nodes(self) -> dict[str, TopologicalNode]:
        return self._nodes

    def add_node(self, label: str = "",
                 node_id: str | None = None) -> TopologicalNode:
        nid = node_id or str(uuid.uuid4())
        node = TopologicalNode(id=nid, label=label)
        self._nodes[nid] = node
        return node

    def add_edge(self, source_id: str, target_id: str,
                 edge_type: EdgeType, weight: float = 1.0,
                 grounded: bool = False) -> Optional[EdgeRelation]:
        source = self._nodes.get(source_id)
        if source is None:
            return None
        return source.add_edge(target_id, edge_type, weight, grounded)

    def get_node(self, node_id: str) -> Optional[TopologicalNode]:
        return self._nodes.get(node_id)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def total_edge_count(self) -> int:
        return sum(n.edge_count() for n in self._nodes.values())
