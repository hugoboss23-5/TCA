"""
Adapter for KD (Knowledge Dynamics) system.

KD is an MCP server with these core capabilities:
  - vault_deposit: store facts/beliefs with confidence, domain, tags
  - vault_query: keyword search ranked by relevance x confidence
  - graph_query: spreading activation search (depth, decay params)
  - graph_build: build edges via trigram similarity + domain + tags
  - belief_review: FSRS-based spaced repetition
  - record_outcome: close prediction loops (confirmed/refuted)

Since KD is an MCP server (not a Python library), this adapter provides:
  1. Data classes matching KD's schema for use in TCA layers
  2. A local cache mode for testing without MCP connectivity
  3. Conversion functions: KD format <-> TCA TopologicalNode format

In production, the MCP bridge would call the actual KD tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KDNode:
    """A knowledge item as KD represents it."""
    id: str
    content: str
    vault: str  # "facts" or "beliefs"
    confidence: float = 0.5
    domain: str = ""
    tags: list[str] = field(default_factory=list)
    ontology_type: str = ""


@dataclass
class KDEdge:
    """An edge in KD's knowledge graph."""
    source_id: str
    target_id: str
    similarity: float = 0.0
    edge_type: str = "similarity"  # KD uses trigram similarity


class KDLocalCache:
    """Local cache that mirrors KD's data model for testing.

    In production, methods here would call MCP tools instead.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, KDNode] = {}
        self._edges: list[KDEdge] = []

    def deposit(self, content: str, vault: str = "facts",
                confidence: float = 0.5, domain: str = "",
                tags: list[str] | None = None,
                node_id: str | None = None) -> str:
        """Deposit a fact or belief. Returns node ID."""
        import uuid
        nid = node_id or str(uuid.uuid4())
        self._nodes[nid] = KDNode(
            id=nid,
            content=content,
            vault=vault,
            confidence=confidence,
            domain=domain,
            tags=tags or [],
        )
        return nid

    def query(self, keywords: str, limit: int = 20) -> list[KDNode]:
        """Simple keyword search over cached nodes."""
        kw_lower = keywords.lower().split()
        scored: list[tuple[float, KDNode]] = []
        for node in self._nodes.values():
            content_lower = node.content.lower()
            hits = sum(1 for kw in kw_lower if kw in content_lower)
            if hits > 0:
                scored.append((hits * node.confidence, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:limit]]

    def get_node(self, node_id: str) -> Optional[KDNode]:
        """Retrieve a node by ID."""
        return self._nodes.get(node_id)

    def get_edges(self, node_id: str) -> list[KDEdge]:
        """Get all edges connected to a node."""
        return [e for e in self._edges
                if e.source_id == node_id or e.target_id == node_id]

    def add_edge(self, source_id: str, target_id: str,
                 similarity: float = 0.5) -> None:
        """Add an edge between two nodes."""
        self._edges.append(KDEdge(
            source_id=source_id,
            target_id=target_id,
            similarity=similarity,
        ))

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)
