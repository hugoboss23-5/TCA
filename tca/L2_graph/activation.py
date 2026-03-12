"""
TCA Layer 2: Spreading Activation

Propagates activation through the topological graph, weighted by
edge weight AND edge type relevance to active gates from L1.

If the VERIFY gate is high, VERIFIES edges propagate more activation
than MIRRORS edges, and so on.
"""

from __future__ import annotations

from tca.L2_graph.topo_node import TopologicalNode, EdgeType, GATE_TO_EDGE


def _edge_type_relevance(edge_type: EdgeType,
                         gate_weights: dict[str, float]) -> float:
    """How relevant is this edge type given the current gate weights?

    Each edge type corresponds to a gate. The gate's weight IS the
    edge type's relevance multiplier.
    """
    for gate_name, et in GATE_TO_EDGE.items():
        if et == edge_type:
            return gate_weights.get(gate_name, 0.0)
    return 0.0


def spread(entry_nodes: list[str],
           gate_weights: dict[str, float],
           graph: dict[str, TopologicalNode],
           depth: int = 3,
           decay_factor: float = 0.5,
           initial_activation: float = 1.0) -> list[tuple[str, float]]:
    """Spreading activation from entry nodes through the graph.

    Args:
        entry_nodes: IDs of nodes to start from.
        gate_weights: Gate activation weights from L1 router.
        graph: Dict mapping node ID -> TopologicalNode.
        depth: Maximum number of hops to propagate.
        decay_factor: Activation multiplier per hop (0-1).
        initial_activation: Starting activation for entry nodes.

    Returns:
        List of (node_id, activation) sorted by activation descending.
        Only includes nodes with activation > 0.
    """
    # Reset all activations.
    for node in graph.values():
        node.activation = 0.0

    # Set entry node activations.
    for nid in entry_nodes:
        if nid in graph:
            graph[nid].activate(initial_activation)

    # BFS-style spreading, depth rounds.
    frontier = set(entry_nodes)

    for d in range(depth):
        next_frontier: set[str] = set()
        for nid in frontier:
            node = graph.get(nid)
            if node is None:
                continue
            source_activation = node.activation

            for target_id, relations in node.edges.items():
                if target_id not in graph:
                    continue

                # Sum contributions from all edges to this target.
                total_contribution = 0.0
                for rel in relations:
                    relevance = _edge_type_relevance(rel.edge_type, gate_weights)
                    contribution = (source_activation
                                    * rel.weight
                                    * relevance
                                    * decay_factor)
                    total_contribution += contribution

                if total_contribution > 0:
                    target_node = graph[target_id]
                    # Accumulate — don't replace.
                    new_level = target_node.activation + total_contribution
                    target_node.activate(new_level)
                    next_frontier.add(target_id)

        frontier = next_frontier

    # Collect results.
    results = [(nid, node.activation)
               for nid, node in graph.items()
               if node.activation > 0]
    results.sort(key=lambda x: x[1], reverse=True)
    return results
