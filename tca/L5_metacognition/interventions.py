"""
TCA Layer 5: Metacognitive Interventions

Based on health metrics, the monitor can intervene in reasoning:
  - deepen: increase max_ticks, let reasoning continue
  - redirect: change gate weights, explore different subgraph
  - prune: deactivate low-activation dead-end nodes
  - flag_uncertainty: mark output as uncertain if confidence low
  - request_grounding: trigger L3 actions if grounding_ratio low
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tca.L2_graph.topo_node import TopologicalNode, EdgeType
from tca.L5_metacognition.health import compute_health, HealthReport
from tca.L5_metacognition.confidence import (
    compute_confidence, ConfidenceReport,
)


@dataclass
class Intervention:
    """A metacognitive intervention."""
    action: str  # deepen, redirect, prune, flag_uncertainty, request_grounding
    reason: str
    details: dict = field(default_factory=dict)


@dataclass
class MonitorResult:
    """Result of metacognitive monitoring."""
    health: HealthReport
    confidence: ConfidenceReport
    interventions: list[Intervention] = field(default_factory=list)


def deepen(current_max_ticks: int, factor: float = 1.5) -> Intervention:
    """Increase max_ticks to allow more reasoning time."""
    new_max = int(current_max_ticks * factor)
    return Intervention(
        action="deepen",
        reason="Reasoning has not converged — extending time budget",
        details={"old_max_ticks": current_max_ticks, "new_max_ticks": new_max},
    )


def redirect(gate_weights: dict[str, float],
             failure_signal: str) -> Intervention:
    """Change gate weights to explore different subgraph."""
    # Reduce dominant gate, boost exploration.
    new_weights = dict(gate_weights)
    max_gate = max(new_weights, key=lambda g: new_weights[g])
    new_weights[max_gate] *= 0.5
    new_weights["DARASH"] = min(0.4, new_weights.get("DARASH", 0.1) * 2)
    total = sum(new_weights.values())
    new_weights = {g: w / total for g, w in new_weights.items()}

    return Intervention(
        action="redirect",
        reason=f"Divergence detected: {failure_signal}",
        details={"new_gate_weights": new_weights},
    )


def prune(subgraph: dict[str, TopologicalNode],
          threshold: float = 0.01) -> Intervention:
    """Deactivate low-activation dead-end nodes."""
    pruned = []
    for nid, node in subgraph.items():
        if node.activation < threshold and node.edge_count() <= 1:
            node.activation = 0.0
            pruned.append(nid)

    return Intervention(
        action="prune",
        reason=f"Pruned {len(pruned)} low-activation dead-end nodes",
        details={"pruned_nodes": pruned},
    )


def flag_uncertainty(confidence: float,
                     threshold: float = 0.3) -> Intervention | None:
    """Flag output as uncertain if confidence below threshold."""
    if confidence < threshold:
        return Intervention(
            action="flag_uncertainty",
            reason=f"Confidence {confidence:.3f} below threshold {threshold}",
            details={"confidence": confidence, "threshold": threshold},
        )
    return None


def request_grounding(grounding_ratio: float,
                      threshold: float = 0.3) -> Intervention | None:
    """Request L3 grounding if grounding ratio is low."""
    if grounding_ratio < threshold:
        return Intervention(
            action="request_grounding",
            reason=f"Grounding ratio {grounding_ratio:.3f} below threshold {threshold}",
            details={"grounding_ratio": grounding_ratio, "threshold": threshold},
        )
    return None


def monitor(subgraph: dict[str, TopologicalNode],
            gate_weights: dict[str, float],
            ticks_used: int = 1,
            max_ticks: int = 100,
            source: str | None = None,
            target: str | None = None,
            confidence_threshold: float = 0.3,
            grounding_threshold: float = 0.3) -> MonitorResult:
    """Run full metacognitive monitoring on a reasoning subgraph.

    The monitor can READ the subgraph (health metrics) and WRITE
    to it (prune, modify activations). This is self-modification.

    Args:
        subgraph: Active reasoning subgraph from L2.
        gate_weights: Current L1 gate weights.
        ticks_used: L4 ticks consumed.
        max_ticks: L4 tick budget.
        source: Source node for path analysis.
        target: Target node for path analysis.
        confidence_threshold: Below this, flag uncertainty.
        grounding_threshold: Below this, request grounding.

    Returns:
        MonitorResult with health, confidence, and interventions.
    """
    health = compute_health(subgraph, source, target)
    conf = compute_confidence(subgraph, source, target, ticks_used, max_ticks)

    result = MonitorResult(health=health, confidence=conf)

    # Check for interventions.
    unc = flag_uncertainty(conf.confidence, confidence_threshold)
    if unc:
        result.interventions.append(unc)

    gnd = request_grounding(conf.grounding_ratio, grounding_threshold)
    if gnd:
        result.interventions.append(gnd)

    if health.has_cycles:
        result.interventions.append(Intervention(
            action="warning",
            reason=f"Circular reasoning detected: {health.cycle_count} cycle(s)",
            details={"cycles": health.cycles},
        ))

    if health.isolated:
        p = prune(subgraph)
        result.interventions.append(p)

    return result
