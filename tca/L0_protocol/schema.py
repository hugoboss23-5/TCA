"""
TCA Layer 0: Cognitive Boot Protocol

Standardized serialization for a complete TCA state snapshot.
Any system that loads this format gets a fully initialized TCA instance.

The ARCHITECTURE is open. The DATA is private.
- serialize_tca_state: full cognitive state (architecture + data)
- deserialize_tca_state: reconstruct from portable format
- export_boot_protocol: structure only, no content (this is what gets open-sourced)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from tca.L2_graph.topo_node import (
    TopologicalNode, EdgeRelation, EdgeType, GATE_TO_EDGE,
)
from tca.L2_graph.operations import TopologicalGraph
from tca.L4_temporal.temporal_node import get_activation_history

TCA_PROTOCOL_VERSION = "0.1.0"


def serialize_tca_state(
    router_weights: dict[str, float] | None = None,
    graph: TopologicalGraph | None = None,
    grounding_loop=None,
    temporal_result=None,
    metacognition_report=None,
) -> dict:
    """Serialize the COMPLETE cognitive state of a TCA instance.

    Returns a JSON-compatible dict that any system can load.

    Args:
        router_weights: L1 gate weights from last routing.
        graph: L2 topological graph.
        grounding_loop: L3 GroundingLoop instance (optional).
        temporal_result: L4 ReasoningResult (optional).
        metacognition_report: L5 ConfidenceReport or ConfidenceV2Report (optional).
    """
    state: dict = {
        "protocol_version": TCA_PROTOCOL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # L1: Router.
    state["L1_router"] = {
        "gate_weights": dict(router_weights) if router_weights else {},
    }

    # L2: Graph.
    nodes_data: list[dict] = []
    if graph is not None:
        for nid, node in graph.nodes.items():
            edges_data: list[dict] = []
            for target_id, rels in node.edges.items():
                for rel in rels:
                    edges_data.append({
                        "target": rel.target_id,
                        "type": rel.edge_type.value,
                        "weight": rel.weight,
                        "grounded": rel.grounded,
                    })
            activation_history = get_activation_history(node)
            nodes_data.append({
                "id": nid,
                "label": node.label,
                "edges": edges_data,
                "activation": node.activation,
                "decay_rate": node.decay_rate,
                "activation_history": activation_history,
            })
    state["L2_graph"] = {"nodes": nodes_data}

    # L3: Grounding.
    grounding_data: dict = {
        "prediction_history": {},
        "signal_sources_registered": [],
    }
    if grounding_loop is not None:
        for (src_id, action_type), deltas in grounding_loop._history.items():
            key = f"{src_id}:{action_type}"
            grounding_data["prediction_history"][key] = deltas
    state["L3_grounding"] = grounding_data

    # L4: Temporal.
    temporal_data: dict = {
        "current_tick": 0,
        "coherence_history": [],
        "ticks_used": 0,
        "converged": False,
    }
    if temporal_result is not None:
        temporal_data["ticks_used"] = getattr(temporal_result, "ticks_used", 0)
        temporal_data["converged"] = getattr(temporal_result, "converged", False)
        temporal_data["current_tick"] = temporal_data["ticks_used"]
        activation_hist = getattr(temporal_result, "activation_history", [])
        temporal_data["coherence_history"] = activation_hist
    state["L4_temporal"] = temporal_data

    # L5: Metacognition.
    meta_data: dict = {
        "confidence": 0.0,
        "confidence_state": "unknown",
        "health_metrics": {},
    }
    if metacognition_report is not None:
        meta_data["confidence"] = getattr(
            metacognition_report, "confidence", 0.0)
        meta_data["confidence_state"] = getattr(
            metacognition_report, "state", "unknown")
        # Include all numeric fields as health metrics.
        for attr in ("path_diversity_score", "cycle_penalty",
                     "grounding_ratio", "convergence_speed",
                     "static_confidence", "dynamic_coherence"):
            val = getattr(metacognition_report, attr, None)
            if val is not None:
                meta_data["health_metrics"][attr] = val
        meta_data["flags"] = getattr(metacognition_report, "flags", [])
    state["L5_metacognition"] = meta_data

    return state


def deserialize_tca_state(json_data: dict) -> dict:
    """Load a TCA state from the portable format.

    Returns a dict with:
        "graph": TopologicalGraph (fully reconstructed)
        "router_weights": dict[str, float]
        "grounding_history": dict[str, list[float]]
        "temporal": dict with ticks_used, converged, coherence_history
        "metacognition": dict with confidence, state, flags
        "protocol_version": str
    """
    result: dict = {
        "protocol_version": json_data.get("protocol_version",
                                          TCA_PROTOCOL_VERSION),
    }

    # L1: Router.
    l1 = json_data.get("L1_router", {})
    result["router_weights"] = l1.get("gate_weights", {})

    # L2: Graph.
    graph = TopologicalGraph()
    l2 = json_data.get("L2_graph", {})
    nodes_list = l2.get("nodes", [])

    # First pass: create all nodes.
    for node_data in nodes_list:
        nid = node_data["id"]
        graph.add_node(
            label=node_data.get("label", ""),
            node_id=nid,
        )
        node = graph.get_node(nid)
        node.activation = node_data.get("activation", 0.0)
        node.decay_rate = node_data.get("decay_rate", 0.1)
        # Restore activation history.
        hist = node_data.get("activation_history", [])
        if hist:
            node._activation_history = list(hist)

    # Second pass: add edges (targets must exist as node IDs).
    for node_data in nodes_list:
        nid = node_data["id"]
        for edge_data in node_data.get("edges", []):
            try:
                edge_type = EdgeType(edge_data["type"])
            except (ValueError, KeyError):
                continue
            graph.add_edge(
                nid,
                edge_data["target"],
                edge_type,
                weight=edge_data.get("weight", 1.0),
                grounded=edge_data.get("grounded", False),
            )

    result["graph"] = graph

    # L3: Grounding.
    l3 = json_data.get("L3_grounding", {})
    result["grounding_history"] = l3.get("prediction_history", {})

    # L4: Temporal.
    l4 = json_data.get("L4_temporal", {})
    result["temporal"] = {
        "ticks_used": l4.get("ticks_used", 0),
        "converged": l4.get("converged", False),
        "coherence_history": l4.get("coherence_history", []),
    }

    # L5: Metacognition.
    l5 = json_data.get("L5_metacognition", {})
    result["metacognition"] = {
        "confidence": l5.get("confidence", 0.0),
        "state": l5.get("confidence_state", "unknown"),
        "flags": l5.get("flags", []),
        "health_metrics": l5.get("health_metrics", {}),
    }

    return result


def export_boot_protocol(graph: TopologicalGraph) -> dict:
    """Export ONLY the structural elements needed to boot TCA from scratch.

    This is what gets open-sourced. This is what the 3am kid loads.
    Contains:
    - The 7 gate definitions and their relationship types
    - The coherence metric specification
    - The resonance grounding interface spec
    - An empty graph with the structural schema ready to populate

    Does NOT contain:
    - Any specific knowledge graph data
    - Any node content or labels
    - Any personal data from grounding history
    - Any activation state

    The ARCHITECTURE is open, the DATA is private.
    """
    return {
        "protocol_version": TCA_PROTOCOL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "gates": list(GATE_TO_EDGE.keys()),
            "edge_types": [et.value for et in EdgeType],
            "gate_to_edge_mapping": {
                gate: et.value for gate, et in GATE_TO_EDGE.items()
            },
        },
        "specifications": {
            "coherence": {
                "method": "pearson_correlation",
                "direct_edge_discount": 0.3,
                "phase_transition_threshold": 0.2,
                "phase_transition_window": 3,
                "states": ["understanding", "premature",
                           "forming", "confused"],
            },
            "resonance": {
                "interface": "SignalSource.measure() -> dict",
                "alignment_range": [0.0, 1.0],
                "strengthen_threshold": 0.7,
                "weaken_threshold": 0.3,
                "v01_sources": ["FileSystemSource",
                                "CodeExecutionSource"],
            },
            "confidence_v2": {
                "components": ["static_topology", "dynamic_coherence"],
                "premature_override": True,
                "description": ("coherence overrides convergence "
                                "when state is premature"),
            },
        },
        "graph_schema": {
            "node_count": graph.node_count if graph else 0,
            "edge_count": graph.total_edge_count() if graph else 0,
            "node_fields": ["id", "label", "edges", "activation",
                            "decay_rate", "activation_history"],
            "edge_fields": ["target", "type", "weight", "grounded"],
        },
    }
