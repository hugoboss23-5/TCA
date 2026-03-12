"""
TCA Layer 3: Prediction-Error Loop

Grounds symbols in consequences through prediction-error loops:
1. Before action: generate PREDICTION about outcome
2. Action executes
3. After action: observe ACTUAL outcome
4. DELTA = prediction - actual
5. If delta > threshold: update relevant edges in Layer 2 graph
   - Strengthen edges that led to correct predictions
   - Weaken edges that led to wrong predictions
   - Create new edges if outcome reveals unknown relationships
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tca.L2_graph.topo_node import EdgeType
from tca.L2_graph.operations import TopologicalGraph


@dataclass
class PredictionResult:
    """Result of a prediction-error cycle."""
    prediction: float  # predicted value (0.0 = no, 1.0 = yes)
    actual: float      # actual value (0.0 = no, 1.0 = yes)
    delta: float       # prediction - actual
    edges_updated: list[str] = field(default_factory=list)


# How much to adjust edge weights per prediction cycle.
_LEARNING_RATE = 0.1
# Maximum edge weight (prevents unbounded growth).
_MAX_WEIGHT = 5.0
# Minimum edge weight (stays positive).
_MIN_WEIGHT = 0.01
# Delta threshold — below this, don't bother updating.
_DELTA_THRESHOLD = 0.05


class GroundingLoop:
    """Manages prediction-error grounding for a topological graph.

    Tracks prediction history to enable learning: after wrong
    predictions, subsequent predictions on the same relationship
    should change (demonstrating learning).
    """

    def __init__(self, graph: TopologicalGraph) -> None:
        self.graph = graph
        # History: maps (source_id, action_type) -> list of deltas.
        self._history: dict[tuple[str, str], list[float]] = {}

    def predict(self, source_id: str, action_type: str) -> float:
        """Generate a prediction based on edge strengths and history.

        Args:
            source_id: Node related to this prediction.
            action_type: Type of action being predicted.

        Returns:
            Predicted outcome (0.0 to 1.0).
        """
        key = (source_id, action_type)
        node = self.graph.get_node(source_id)

        # Base prediction from edge weights.
        base = 0.7  # Default: mildly confident
        if node is not None:
            verifies = node.get_edges_by_type(EdgeType.VERIFIES)
            removes = node.get_edges_by_type(EdgeType.REMOVES)
            if verifies:
                avg_v = sum(e.weight for e in verifies) / len(verifies)
                base = min(0.95, base + avg_v * 0.1)
            if removes:
                avg_r = sum(e.weight for e in removes) / len(removes)
                base = max(0.05, base - avg_r * 0.1)

        # Adjust based on history (learning).
        history = self._history.get(key, [])
        if history:
            # Recent errors push prediction away from previous errors.
            recent_delta = history[-1]
            base = max(0.05, min(0.95, base - recent_delta * 0.3))

        return base

    def ground(self, source_id: str, target_id: str,
               action_type: str,
               prediction: float, actual: float) -> PredictionResult:
        """Execute the full prediction-error loop.

        Updates edges between source and target based on prediction error.

        Args:
            source_id: Node that made the prediction.
            target_id: Node that was predicted about.
            action_type: Type of grounding action.
            prediction: What was predicted (0-1).
            actual: What actually happened (0-1).

        Returns:
            PredictionResult with delta and list of updated edges.
        """
        delta = prediction - actual
        result = PredictionResult(
            prediction=prediction,
            actual=actual,
            delta=delta,
        )

        # Record history.
        key = (source_id, action_type)
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(delta)

        # Skip small deltas.
        if abs(delta) < _DELTA_THRESHOLD:
            return result

        node = self.graph.get_node(source_id)
        if node is None:
            return result

        # Update edges.
        if target_id in node.edges:
            for rel in node.edges[target_id]:
                old_weight = rel.weight
                if abs(delta) < 0.2:
                    # Close prediction — strengthen.
                    rel.weight = min(_MAX_WEIGHT,
                                     rel.weight + _LEARNING_RATE)
                    rel.grounded = True
                else:
                    # Bad prediction — weaken.
                    rel.weight = max(_MIN_WEIGHT,
                                     rel.weight - _LEARNING_RATE * abs(delta))
                result.edges_updated.append(
                    f"{source_id}->{target_id} ({rel.edge_type.value}): "
                    f"{old_weight:.3f} -> {rel.weight:.3f}"
                )
        else:
            # No edge exists — create one based on outcome.
            if actual > 0.5:
                edge_type = EdgeType.VERIFIES
            else:
                edge_type = EdgeType.REMOVES
            self.graph.add_edge(source_id, target_id, edge_type,
                                weight=abs(actual) * 0.5)
            result.edges_updated.append(
                f"NEW {source_id}->{target_id} ({edge_type.value}): "
                f"weight={abs(actual) * 0.5:.3f}"
            )

        return result

    def get_history(self, source_id: str, action_type: str
                    ) -> list[float]:
        """Get prediction error history for a source+action pair."""
        return self._history.get((source_id, action_type), [])
