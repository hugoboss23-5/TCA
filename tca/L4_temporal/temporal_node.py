"""
TCA Layer 4: Temporal Node Enhancement

Extends TopologicalNode (L2) with temporal reasoning capabilities:
  - activation_history: track activation over time
  - synchronization: measure correlation with neighbors

Nodes that synchronize = part of the same "thought".
Nodes that desynchronize = competing hypotheses.
"""

from __future__ import annotations

from tca.L2_graph.topo_node import TopologicalNode


def record_activation(node: TopologicalNode, max_history: int = 50) -> None:
    """Record current activation into the node's history.

    Uses a list stored in a '_activation_history' attribute added
    dynamically. This avoids modifying the L2 TopologicalNode class
    while still extending its behavior.
    """
    if not hasattr(node, "_activation_history"):
        node._activation_history = []
    node._activation_history.append(node.activation)
    if len(node._activation_history) > max_history:
        node._activation_history = node._activation_history[-max_history:]


def get_activation_history(node: TopologicalNode) -> list[float]:
    """Get the activation history for a node."""
    return getattr(node, "_activation_history", [])


def compute_synchronization(node_a: TopologicalNode,
                            node_b: TopologicalNode) -> float:
    """Compute synchronization between two nodes.

    Synchronization = correlation of their activation histories.
    Uses a simple correlation measure: how often they move in
    the same direction (both increase or both decrease).

    Returns:
        Float in [-1, 1]. 1 = perfectly synchronized,
        -1 = perfectly anti-synchronized, 0 = uncorrelated.
    """
    hist_a = get_activation_history(node_a)
    hist_b = get_activation_history(node_b)

    if len(hist_a) < 2 or len(hist_b) < 2:
        return 0.0

    # Align to same length (use most recent).
    min_len = min(len(hist_a), len(hist_b))
    ha = hist_a[-min_len:]
    hb = hist_b[-min_len:]

    # Compute direction agreement.
    agreements = 0
    total = 0
    for i in range(1, min_len):
        da = ha[i] - ha[i - 1]
        db = hb[i] - hb[i - 1]
        if abs(da) > 1e-9 or abs(db) > 1e-9:
            total += 1
            if (da > 0 and db > 0) or (da < 0 and db < 0):
                agreements += 1
            elif (da > 0 and db < 0) or (da < 0 and db > 0):
                agreements -= 1
            # If one is zero and other isn't, count as 0 (neutral).

    if total == 0:
        return 0.0

    return agreements / total


def compute_graph_synchronization(
    graph: dict[str, TopologicalNode]
) -> dict[str, float]:
    """Compute average synchronization for each node with its neighbors.

    Returns dict of node_id -> avg_sync_with_neighbors.
    """
    sync_scores: dict[str, float] = {}

    for nid, node in graph.items():
        neighbors = node.get_neighbors()
        if not neighbors:
            sync_scores[nid] = 0.0
            continue

        total_sync = 0.0
        count = 0
        for nb_id in neighbors:
            nb_node = graph.get(nb_id)
            if nb_node is not None:
                total_sync += compute_synchronization(node, nb_node)
                count += 1

        sync_scores[nid] = total_sync / count if count > 0 else 0.0

    return sync_scores
