"""
TCA Layer 5: Confidence from Topology

Confidence is NOT a learned scalar. It is COMPUTED from topology:

  confidence = (
      path_diversity * 0.3 +         # multiple supporting paths
      (1 - cycle_ratio) * 0.2 +      # low circular reasoning
      grounding_ratio * 0.3 +         # proportion of edges validated by L3
      convergence_speed * 0.2         # how quickly L4 stabilized
  )

This replaces black-box "how confident are you?" with a transparent,
auditable metric.
"""

from __future__ import annotations

from dataclasses import dataclass

from tca.L2_graph.topo_node import TopologicalNode
from tca.L5_metacognition.health import (
    compute_health, path_diversity, cycle_detection,
)


@dataclass
class ConfidenceReport:
    """Detailed confidence breakdown."""
    confidence: float
    path_diversity_score: float
    cycle_penalty: float
    grounding_ratio: float
    convergence_speed: float

    # Weights (transparent, auditable).
    W_PATH = 0.3
    W_CYCLE = 0.2
    W_GROUNDING = 0.3
    W_CONVERGENCE = 0.2


def compute_grounding_ratio(subgraph: dict[str, TopologicalNode]) -> float:
    """Proportion of edges that have been grounded by L3."""
    total_edges = 0
    grounded_edges = 0
    for node in subgraph.values():
        for rels in node.edges.values():
            for rel in rels:
                total_edges += 1
                if rel.grounded:
                    grounded_edges += 1
    return grounded_edges / total_edges if total_edges > 0 else 0.0


def compute_convergence_speed(ticks_used: int, max_ticks: int) -> float:
    """How quickly reasoning converged. 1.0 = instant, 0.0 = timed out."""
    if max_ticks <= 0:
        return 1.0
    return max(0.0, 1.0 - (ticks_used / max_ticks))


def compute_confidence(subgraph: dict[str, TopologicalNode],
                       source: str | None = None,
                       target: str | None = None,
                       ticks_used: int = 1,
                       max_ticks: int = 100) -> ConfidenceReport:
    """Compute confidence from topological metrics.

    Args:
        subgraph: The active reasoning subgraph.
        source: Source node ID (for path diversity).
        target: Target node ID (for path diversity).
        ticks_used: How many ticks L4 used.
        max_ticks: Maximum ticks allowed.

    Returns:
        ConfidenceReport with breakdown.
    """
    # Path diversity.
    if source and target:
        pd = path_diversity(subgraph, source, target)
    else:
        pd = 0.0

    # Cycle ratio.
    cycles = cycle_detection(subgraph)
    total_nodes = len(subgraph)
    if total_nodes > 0:
        # Count unique nodes involved in cycles.
        cycle_nodes: set[str] = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)
        cycle_ratio = len(cycle_nodes) / total_nodes
    else:
        cycle_ratio = 0.0

    # Grounding ratio.
    gr = compute_grounding_ratio(subgraph)

    # Convergence speed.
    cs = compute_convergence_speed(ticks_used, max_ticks)

    # Compute confidence.
    confidence = (
        pd * ConfidenceReport.W_PATH +
        (1.0 - cycle_ratio) * ConfidenceReport.W_CYCLE +
        gr * ConfidenceReport.W_GROUNDING +
        cs * ConfidenceReport.W_CONVERGENCE
    )

    return ConfidenceReport(
        confidence=confidence,
        path_diversity_score=pd,
        cycle_penalty=cycle_ratio,
        grounding_ratio=gr,
        convergence_speed=cs,
    )
