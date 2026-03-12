"""
TCA Layer 2: Graph Operations

Core operations on the topological knowledge graph.
All queries work through edge traversal — no similarity search,
no embeddings.
"""

from __future__ import annotations

import uuid
from typing import Optional

from tca.L2_graph.topo_node import TopologicalNode, EdgeType, EdgeRelation
from tca.L2_graph.activation import spread
from tca.adapters.kd_adapter import KDNode, KDLocalCache


class TopologicalGraph:
    """The topological knowledge graph.

    Stores nodes whose meaning is entirely their edge sets.
    All queries operate through edge traversal and spreading activation.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, TopologicalNode] = {}

    @property
    def nodes(self) -> dict[str, TopologicalNode]:
        return self._nodes

    def add_node(self, edges: dict[str, list[tuple[EdgeType, float]]] | None = None,
                 label: str = "",
                 node_id: str | None = None) -> TopologicalNode:
        """Create a node defined by its relationships.

        Args:
            edges: Dict of {target_id: [(edge_type, weight), ...]}.
            label: Human-readable label (for debugging only).
            node_id: Optional fixed ID. Generated if not provided.

        Returns:
            The created TopologicalNode.
        """
        nid = node_id or str(uuid.uuid4())
        node = TopologicalNode(id=nid, label=label)

        if edges:
            for target_id, rels in edges.items():
                for edge_type, weight in rels:
                    node.add_edge(target_id, edge_type, weight)

        self._nodes[nid] = node
        return node

    def add_edge(self, source_id: str, target_id: str,
                 edge_type: EdgeType, weight: float = 1.0,
                 grounded: bool = False) -> Optional[EdgeRelation]:
        """Connect two nodes with a typed edge.

        Returns the EdgeRelation, or None if source doesn't exist.
        """
        source = self._nodes.get(source_id)
        if source is None:
            return None
        return source.add_edge(target_id, edge_type, weight, grounded)

    def get_node(self, node_id: str) -> Optional[TopologicalNode]:
        """Retrieve a node by ID."""
        return self._nodes.get(node_id)

    def query_by_activation(self, entry_nodes: list[str],
                            gate_weights: dict[str, float],
                            depth: int = 3) -> list[tuple[str, float]]:
        """Query the graph via spreading activation.

        Args:
            entry_nodes: Starting node IDs.
            gate_weights: L1 router gate weights.
            depth: How many hops to propagate.

        Returns:
            List of (node_id, activation) sorted by activation.
        """
        return spread(entry_nodes, gate_weights, self._nodes, depth=depth)

    def subgraph(self, node_ids: list[str]) -> dict[str, TopologicalNode]:
        """Extract the connected subgraph for a set of node IDs.

        Returns only the specified nodes (not their neighbors unless
        those neighbors are also in node_ids).
        """
        return {nid: self._nodes[nid]
                for nid in node_ids
                if nid in self._nodes}

    def read_activated_subgraph(self, threshold: float = 0.01
                                ) -> list[tuple[str, float]]:
        """Read all nodes with activation above threshold.

        Returns sorted list of (node_id, activation).
        """
        result = [(nid, node.activation)
                  for nid, node in self._nodes.items()
                  if node.activation > threshold]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def find_entry(self, query: str) -> list[str]:
        """Find entry nodes for a query by matching labels.

        Simple keyword matching on node labels. In production, this
        would use KD adapter for richer semantic matching.
        """
        keywords = query.lower().split()
        scored: list[tuple[float, str]] = []
        for nid, node in self._nodes.items():
            label_lower = node.label.lower()
            hits = sum(1 for kw in keywords if kw in label_lower)
            if hits > 0:
                scored.append((hits, nid))
        scored.sort(reverse=True)
        return [nid for _, nid in scored[:5]]

    def merge_from_kd(self, kd_nodes: list[KDNode],
                      kd_cache: KDLocalCache) -> list[str]:
        """Import KD query results into topological format.

        Creates TopologicalNodes from KD nodes, and maps KD edges
        to MIRRORS edge type (since KD edges are similarity-based).

        Args:
            kd_nodes: List of KDNode from KD adapter query.
            kd_cache: The KD cache to look up edges.

        Returns:
            List of created node IDs.
        """
        created_ids: list[str] = []

        for kd_node in kd_nodes:
            topo_node = self.add_node(
                label=kd_node.content[:80],
                node_id=kd_node.id,
            )
            created_ids.append(topo_node.id)

        # Now add edges between nodes that exist in both KD and our graph.
        for kd_node in kd_nodes:
            kd_edges = kd_cache.get_edges(kd_node.id)
            for kd_edge in kd_edges:
                other_id = (kd_edge.target_id
                            if kd_edge.source_id == kd_node.id
                            else kd_edge.source_id)
                if other_id in self._nodes:
                    self.add_edge(
                        kd_node.id, other_id,
                        EdgeType.MIRRORS,  # KD similarity -> MIRRORS
                        weight=kd_edge.similarity,
                    )

        return created_ids

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def total_edge_count(self) -> int:
        return sum(n.edge_count() for n in self._nodes.values())
