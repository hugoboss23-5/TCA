"""
TCA Layer 3: Resonance Grounding Interface

Refactors grounding from predict→check→update to a resonance primitive:
"Is this node's position in the graph IN PHASE with the signal from this source?"

V0.1 sources (software):
  - FileSystemSource: does the graph's claim about a file match reality?
  - CodeExecutionSource: does the graph's prediction about code output match actual?

Future sources (hardware, Phase 2+):
  - WattyNode: does the graph's state correlate with biometric coherence?
  - Receiver: does the graph's activation pattern resonate with 7.83Hz field data?
  - Saturn: does the graph's prediction match keystroke behavioral patterns?

ALL sources implement one method: measure() → signal_data.
This same function works unchanged when hardware sources plug in.
"""

from __future__ import annotations

import os

from tca.L2_graph.topo_node import TopologicalNode, EdgeType
from tca.L2_graph.operations import TopologicalGraph


class SignalSource:
    """Anything TCA can test its graph against.

    All sources implement measure() → dict with signal data.
    The resonance test is the same regardless of source type:
    is the graph in phase with the signal?
    """

    def measure(self) -> dict:
        raise NotImplementedError


class FileSystemSource(SignalSource):
    """Wraps file existence check as a signal source."""

    def __init__(self, path: str) -> None:
        self.path = path

    def measure(self) -> dict:
        return {"exists": os.path.exists(self.path),
                "source_type": "filesystem"}


class CodeExecutionSource(SignalSource):
    """Wraps code execution check as a signal source."""

    def __init__(self, code_string: str) -> None:
        self.code_string = code_string

    def measure(self) -> dict:
        try:
            exec(self.code_string)  # noqa: S102 — intentional for grounding
            return {"success": True, "error": None, "source_type": "code"}
        except Exception as e:
            return {"success": False, "error": str(e), "source_type": "code"}


def derive_prediction_from_topology(node: TopologicalNode,
                                    graph: TopologicalGraph) -> dict:
    """Derive what the topology predicts from a node's edges.

    VERIFIES edges → predict positive outcome (exists/success).
    REMOVES edges → predict negative outcome (absent/failure).
    Weight by edge weights to get confidence.
    """
    verifies = node.get_edges_by_type(EdgeType.VERIFIES)
    removes = node.get_edges_by_type(EdgeType.REMOVES)

    verify_strength = (sum(e.weight for e in verifies) / len(verifies)
                       if verifies else 0.0)
    remove_strength = (sum(e.weight for e in removes) / len(removes)
                       if removes else 0.0)

    total = verify_strength + remove_strength
    if total < 1e-9:
        # No evidence edges — neutral prediction.
        return {"predicted_positive": True, "confidence": 0.5}

    positive_ratio = verify_strength / total
    return {
        "predicted_positive": positive_ratio > 0.5,
        "confidence": max(positive_ratio, 1.0 - positive_ratio),
    }


def compute_alignment(prediction: dict, signal: dict) -> float:
    """Compute phase alignment between topology prediction and signal.

    Returns 0.0 (completely wrong) to 1.0 (perfect resonance).
    """
    source_type = signal.get("source_type", "")

    if source_type == "filesystem":
        signal_positive = signal.get("exists", False)
    elif source_type == "code":
        signal_positive = signal.get("success", False)
    else:
        # Generic: look for any truthy value in signal.
        signal_positive = any(
            v for k, v in signal.items() if k != "source_type"
        )

    pred_positive = prediction.get("predicted_positive", True)
    pred_confidence = prediction.get("confidence", 0.5)

    if pred_positive == signal_positive:
        # Prediction matches reality — alignment scales with confidence.
        return 0.5 + 0.5 * pred_confidence
    else:
        # Prediction wrong — anti-alignment scales with confidence.
        return 0.5 - 0.5 * pred_confidence


# Edge weight adjustment constants (match L3 prediction_loop conventions).
_STRENGTHEN = 0.1
_MAX_WEIGHT = 5.0
_MIN_WEIGHT = 0.01


def update_graph_from_resonance(node: TopologicalNode,
                                graph: TopologicalGraph,
                                alignment: float) -> None:
    """Update edge weights based on resonance alignment.

    High alignment (> 0.7): strengthen edges, mark grounded.
    Low alignment (< 0.3): weaken edges.
    """
    for rels in node.edges.values():
        for rel in rels:
            if alignment > 0.7:
                rel.weight = min(_MAX_WEIGHT, rel.weight + _STRENGTHEN)
                rel.grounded = True
            elif alignment < 0.3:
                weaken = _STRENGTHEN * (1.0 - alignment)
                rel.weight = max(_MIN_WEIGHT, rel.weight - weaken)


def test_resonance(node: TopologicalNode,
                   signal_source: SignalSource,
                   graph: TopologicalGraph) -> float:
    """THE CORE GROUNDING PRIMITIVE.

    Instead of predict → check → update, we ask:
    "Is this node's position in the graph IN PHASE with the signal?"

    1. Derive prediction from topology (what do the edges claim?)
    2. Measure reality via signal source
    3. Compute phase alignment
    4. Update graph based on alignment

    Returns: phase_alignment_score (0.0-1.0).

    This same function works unchanged when hardware sources plug in.
    """
    prediction = derive_prediction_from_topology(node, graph)
    signal = signal_source.measure()
    alignment = compute_alignment(prediction, signal)
    update_graph_from_resonance(node, graph, alignment)
    return alignment
