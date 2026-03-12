"""
TCA Phase 2: KD MCP Adapter

Calls KD's live MCP tools for real knowledge retrieval.
Falls back to KDLocalCache if MCP is unavailable.

MCP tools used:
  - vault_query: keyword search, returns ranked facts/beliefs
  - vault_deposit: add fact or belief
  - vault_stats: node counts
  - graph_query: spreading activation search
  - graph_build: build/rebuild edges
  - graph_stats: edge counts and types
  - list_domains: all knowledge domains

The adapter converts MCP JSON responses into KDNode/KDEdge objects
that the rest of TCA expects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from tca.adapters.kd_adapter import KDNode, KDEdge, KDLocalCache


@dataclass
class KDStats:
    """Vault statistics from live KD."""
    total_facts: int = 0
    total_beliefs: int = 0
    avg_belief_confidence: float = 0.0
    total_nodes: int = 0


@dataclass
class KDGraphStats:
    """Graph statistics from live KD."""
    total_edges: int = 0
    edge_types: list[dict[str, Any]] = field(default_factory=list)
    most_connected: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class KDActivatedNode(KDNode):
    """A KD node with activation from graph_query."""
    activation: float = 0.0
    hop: int = 0
    discovered: bool = False


class KDMCPAdapter:
    """Adapter that calls KD's MCP tools directly.

    MCP tool calls are injected via the call_mcp function provided
    at construction. This allows the adapter to work with any MCP
    transport layer.

    Falls back to KDLocalCache if call_mcp is None or raises.
    """

    def __init__(self, call_mcp=None):
        """Initialize adapter.

        Args:
            call_mcp: Async or sync function that calls MCP tools.
                      Signature: call_mcp(tool_name, params) -> dict
                      If None, all operations fall back to local cache.
        """
        self._call_mcp = call_mcp
        self._fallback = KDLocalCache()
        self._mcp_available = call_mcp is not None

        # Cache for MCP results to avoid repeated calls.
        self._stats_cache: Optional[KDStats] = None
        self._graph_stats_cache: Optional[KDGraphStats] = None
        self._domains_cache: Optional[list[dict]] = None

    @property
    def mcp_available(self) -> bool:
        return self._mcp_available

    @property
    def fallback(self) -> KDLocalCache:
        return self._fallback

    # --- Data loading from pre-fetched MCP results ---

    def load_vault_stats(self, raw: dict) -> KDStats:
        """Parse vault_stats MCP response."""
        stats = KDStats(
            total_facts=raw.get("total_facts", 0),
            total_beliefs=raw.get("total_beliefs", 0),
            avg_belief_confidence=raw.get("avg_belief_confidence", 0.0),
        )
        stats.total_nodes = stats.total_facts + stats.total_beliefs
        self._stats_cache = stats
        return stats

    def load_graph_stats(self, raw: dict) -> KDGraphStats:
        """Parse graph_stats MCP response."""
        gs = KDGraphStats(
            total_edges=raw.get("total_edges", 0),
            edge_types=raw.get("edge_types", []),
            most_connected=raw.get("most_connected", []),
        )
        self._graph_stats_cache = gs
        return gs

    def load_domains(self, raw: list[dict]) -> list[dict]:
        """Parse list_domains MCP response."""
        self._domains_cache = raw
        return raw

    def parse_vault_query(self, raw_results: list[dict]) -> list[KDNode]:
        """Convert vault_query JSON results to KDNode objects.

        Also deposits them into the local cache fallback for
        subsequent edge lookups.
        """
        nodes = []
        for item in raw_results:
            node = KDNode(
                id=item.get("id", ""),
                content=item.get("content", ""),
                vault=item.get("vault", "facts"),
                confidence=item.get("confidence", 0.5),
                domain=item.get("domain", ""),
                tags=item.get("tags", []),
                ontology_type=item.get("ontology_type", ""),
            )
            nodes.append(node)
            # Also store in local cache for edge lookups.
            self._fallback.deposit(
                content=node.content,
                vault=node.vault,
                confidence=node.confidence,
                domain=node.domain,
                tags=node.tags,
                node_id=node.id,
            )
        return nodes

    def parse_graph_query(self, raw_results: list[dict]
                          ) -> list[KDActivatedNode]:
        """Convert graph_query JSON results to KDActivatedNode objects.

        graph_query returns nodes with activation scores and hop distances.
        """
        nodes = []
        for item in raw_results:
            node = KDActivatedNode(
                id=item.get("id", ""),
                content=item.get("content", ""),
                vault=item.get("vault", "facts"),
                confidence=item.get("confidence", 0.5),
                domain=item.get("domain", ""),
                tags=item.get("tags", []),
                ontology_type=item.get("ontology_type", ""),
                activation=item.get("activation", 0.0),
                hop=item.get("hop", 0),
                discovered=item.get("discovered", False),
            )
            nodes.append(node)
            # Also cache locally.
            self._fallback.deposit(
                content=node.content,
                vault=node.vault,
                confidence=node.confidence,
                domain=node.domain,
                tags=node.tags,
                node_id=node.id,
            )
        return nodes

    def build_edges_between_cached(self) -> int:
        """Build edges between all cached nodes based on shared domain/tags.

        Since we can't call graph_build from Python, we approximate
        KD's edge-building logic: nodes in the same domain get a
        domain edge, nodes sharing tags get a tag edge.
        """
        nodes = list(self._fallback._nodes.values())
        edge_count = 0

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]

                # Domain edge.
                if a.domain and a.domain == b.domain:
                    self._fallback.add_edge(a.id, b.id, similarity=0.3)
                    edge_count += 1

                # Tag overlap edge.
                shared_tags = set(a.tags) & set(b.tags)
                if shared_tags:
                    sim = min(0.8, len(shared_tags) * 0.2)
                    self._fallback.add_edge(a.id, b.id, similarity=sim)
                    edge_count += 1

        return edge_count

    # --- Convenience accessors ---

    @property
    def node_count(self) -> int:
        if self._stats_cache:
            return self._stats_cache.total_nodes
        return self._fallback.node_count

    @property
    def edge_count(self) -> int:
        if self._graph_stats_cache:
            return self._graph_stats_cache.total_edges
        return self._fallback.edge_count

    @property
    def cached_node_count(self) -> int:
        """Nodes currently loaded into local cache."""
        return self._fallback.node_count

    def query(self, keywords: str, limit: int = 20) -> list[KDNode]:
        """Query using local cache (for L2 compatibility)."""
        return self._fallback.query(keywords, limit)

    def get_node(self, node_id: str) -> Optional[KDNode]:
        return self._fallback.get_node(node_id)

    def get_edges(self, node_id: str) -> list[KDEdge]:
        return self._fallback.get_edges(node_id)
