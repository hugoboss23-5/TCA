"""
TCA Layer 2: Topological Node

The most important file in TCA. A TopologicalNode's meaning is
ENTIRELY its edge set. No vector embeddings. No coordinates.
No latent space. Topology IS the representation.

Edge types mirror the 7 chestohedron gates:
  MIRRORS  — is analogous to
  INHERITS — derives from
  BOUNDS   — constrains
  EXPRESSES — manifests as
  VERIFIES — provides evidence for
  REMOVES  — contradicts/negates
  SEEKS    — open question toward
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    """The 7 topological relationship types (chestohedron gates)."""
    MIRRORS = "MIRRORS"
    INHERITS = "INHERITS"
    BOUNDS = "BOUNDS"
    EXPRESSES = "EXPRESSES"
    VERIFIES = "VERIFIES"
    REMOVES = "REMOVES"
    SEEKS = "SEEKS"


# Map router gate names to edge types for gate-weighted activation.
GATE_TO_EDGE: dict[str, EdgeType] = {
    "MIRROR": EdgeType.MIRRORS,
    "INHERIT": EdgeType.INHERITS,
    "BOUND": EdgeType.BOUNDS,
    "EXPRESS": EdgeType.EXPRESSES,
    "VERIFY": EdgeType.VERIFIES,
    "REMOVE": EdgeType.REMOVES,
    "DARASH": EdgeType.SEEKS,
}


@dataclass
class EdgeRelation:
    """A typed, weighted relationship to another node."""
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    grounded: bool = False  # Has this edge been validated by L3?

    def __post_init__(self) -> None:
        self.weight = max(0.0, min(10.0, self.weight))


@dataclass
class TopologicalNode:
    """A node whose meaning is entirely its edge set.

    No coordinates. No embeddings. The node IS its connections.

    Attributes:
        id: Unique identifier.
        label: Human-readable label (for debugging/display only,
               NOT used for meaning computation).
        edges: Dict mapping target_id -> list of EdgeRelations.
               A node can have multiple typed edges to the same target.
        activation: Current activation level (for spreading activation).
        decay_rate: How fast activation decays per tick (0-1).
        last_activated: Timestamp of last activation.
    """
    id: str
    label: str = ""
    edges: dict[str, list[EdgeRelation]] = field(default_factory=dict)
    activation: float = 0.0
    decay_rate: float = 0.1
    last_activated: float = field(default_factory=time.time)

    def add_edge(self, target_id: str, edge_type: EdgeType,
                 weight: float = 1.0, grounded: bool = False) -> EdgeRelation:
        """Add a typed edge to another node."""
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
        """Get all edges of a specific type."""
        result = []
        for rels in self.edges.values():
            for rel in rels:
                if rel.edge_type == edge_type:
                    result.append(rel)
        return result

    def get_neighbors(self) -> set[str]:
        """Get all connected node IDs."""
        return set(self.edges.keys())

    def edge_count(self) -> int:
        """Total number of edges (including multiple edges to same target)."""
        return sum(len(rels) for rels in self.edges.values())

    def activate(self, level: float) -> None:
        """Set activation level and update timestamp."""
        self.activation = level
        self.last_activated = time.time()

    def decay(self) -> None:
        """Apply one tick of decay to activation."""
        self.activation *= (1.0 - self.decay_rate)

    def edge_type_distribution(self) -> dict[EdgeType, int]:
        """Count edges by type — the topological fingerprint of this node."""
        dist: dict[EdgeType, int] = {et: 0 for et in EdgeType}
        for rels in self.edges.values():
            for rel in rels:
                dist[rel.edge_type] += 1
        return dist
