"""
TCA Layer 4: Reasoning Loop

Reasoning unfolds over internal time:
1. Route query through L1 gates
2. Find entry nodes in L2 graph
3. For each tick:
   a. Propagate activation (L2)
   b. Record activation history
   c. Compute synchronization
   d. Check convergence
   e. If diverging, re-route through different gates
4. Read activated subgraph
5. Ground result via L3 prediction-error
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tca.L1_router.router import route as l1_route
from tca.L2_graph.activation import spread
from tca.L2_graph.operations import TopologicalGraph
from tca.L3_grounding.prediction_loop import GroundingLoop
from tca.L4_temporal.clock import InternalClock, ConvergenceState
from tca.L4_temporal.temporal_node import (
    record_activation, compute_graph_synchronization,
)


@dataclass
class SynchronizationReport:
    """Report on synchronization state across the graph."""
    converged: bool = False
    diverging: bool = False
    avg_sync: float = 0.0
    failure_signal: str = ""


@dataclass
class ReasoningResult:
    """Result of a reasoning cycle."""
    query: str
    ticks_used: int
    converged: bool
    activated_nodes: list[tuple[str, float]]  # (node_id, activation)
    gate_weights: dict[str, float]
    rerouted: bool = False
    activation_history: list[float] = field(default_factory=list)


def _compute_sync_report(graph: TopologicalGraph,
                         clock_state: ConvergenceState
                         ) -> SynchronizationReport:
    """Compute synchronization report from graph state."""
    sync_scores = compute_graph_synchronization(graph.nodes)
    avg_sync = (sum(sync_scores.values()) / len(sync_scores)
                if sync_scores else 0.0)

    return SynchronizationReport(
        converged=clock_state.converged,
        diverging=clock_state.diverging,
        avg_sync=avg_sync,
        failure_signal=clock_state.failure_signal,
    )


def reroute(query: str, failure_signal: str,
            current_weights: dict[str, float]) -> dict[str, float]:
    """Re-route query through different gates after divergence.

    Boosts DARASH (exploration) and VERIFY (checking) gates,
    reduces the currently dominant gate.
    """
    new_weights = dict(current_weights)

    # Find the dominant gate and reduce it.
    max_gate = max(new_weights, key=lambda g: new_weights[g])
    new_weights[max_gate] *= 0.5

    # Boost exploration and verification.
    new_weights["DARASH"] = min(0.4, new_weights.get("DARASH", 0.1) * 2)
    new_weights["VERIFY"] = min(0.3, new_weights.get("VERIFY", 0.1) * 1.5)

    # Re-normalize.
    total = sum(new_weights.values())
    return {g: w / total for g, w in new_weights.items()}


def reason(query: str,
           graph: TopologicalGraph,
           grounding_loop: GroundingLoop | None = None,
           max_ticks: int = 100,
           convergence_threshold: float = 0.01) -> ReasoningResult:
    """Execute a full reasoning cycle.

    Args:
        query: The input query to reason about.
        graph: The topological knowledge graph.
        grounding_loop: Optional L3 grounding loop.
        max_ticks: Maximum reasoning ticks.
        convergence_threshold: When to stop (activation delta).

    Returns:
        ReasoningResult with activated nodes, ticks used, etc.
    """
    # L1: Route
    gate_weights = l1_route(query)

    # L2: Find entry nodes
    entry_nodes = graph.find_entry(query)
    if not entry_nodes:
        # No entry nodes found — return empty result.
        return ReasoningResult(
            query=query, ticks_used=0, converged=True,
            activated_nodes=[], gate_weights=gate_weights,
        )

    # L4: Temporal reasoning loop
    clock = InternalClock(
        max_ticks=max_ticks,
        convergence_threshold=convergence_threshold,
    )
    rerouted = False

    while clock.has_budget():
        # L2: Spreading activation
        spread(entry_nodes, gate_weights, graph.nodes, depth=1)

        # Record activation history for synchronization.
        for node in graph.nodes.values():
            record_activation(node)

        # Get current activations.
        current_activations = {
            nid: node.activation
            for nid, node in graph.nodes.items()
        }

        # Check convergence.
        clock_state = clock.advance(current_activations)

        if clock_state.converged:
            break

        if clock_state.diverging and clock.tick > 10:
            # Re-route through different gates.
            gate_weights = reroute(query, clock_state.failure_signal,
                                   gate_weights)
            rerouted = True

    # Read result.
    activated = graph.read_activated_subgraph(threshold=0.001)

    result = ReasoningResult(
        query=query,
        ticks_used=clock.tick,
        converged=clock_state.converged if clock.tick > 0 else True,
        activated_nodes=activated,
        gate_weights=gate_weights,
        rerouted=rerouted,
        activation_history=clock.history,
    )

    return result
