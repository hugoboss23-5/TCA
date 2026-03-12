"""TCA Calculator — Engine wrapper.

Thin adapter between FastAPI routes and the real TCA engine.
Does NOT reimplement any logic. Calls the real thing.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from tca.L0_protocol.schema import (
    deserialize_tca_state,
    export_boot_protocol,
    serialize_tca_state,
)
from tca.L2_graph.operations import TopologicalGraph
from tca.L2_graph.topo_node import EdgeType, TopologicalNode
from tca.L5_metacognition.confidence import compute_confidence
from tca.L5_metacognition.health import compute_health
from tca.L5_metacognition.solutions import Solution, TopologicalSolver

# ── In-memory graph store ───────────────────────────────────────

_graphs: dict[str, dict[str, Any]] = {}
# Each value: {"name": str, "graph": TopologicalGraph, "last_solutions": list[Solution]}

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")


# ── Graph lifecycle ─────────────────────────────────────────────

def create_graph(name: str = "Untitled",
                 template_name: str | None = None) -> str:
    """Create a new graph, optionally from a template. Returns graph_id."""
    graph_id = str(uuid.uuid4())[:8]

    if template_name:
        tpl = load_template(template_name)
        if tpl is not None:
            restored = deserialize_tca_state(tpl)
            graph = restored["graph"]
        else:
            graph = TopologicalGraph()
    else:
        graph = TopologicalGraph()

    _graphs[graph_id] = {
        "name": name if name != "Untitled" and name else (
            template_name or "Untitled"
        ),
        "graph": graph,
        "last_solutions": [],
    }
    return graph_id


def get_graph(graph_id: str) -> TopologicalGraph | None:
    entry = _graphs.get(graph_id)
    return entry["graph"] if entry else None


def delete_graph(graph_id: str) -> bool:
    return _graphs.pop(graph_id, None) is not None


def list_graphs() -> list[dict[str, Any]]:
    return [
        {
            "graph_id": gid,
            "name": entry["name"],
            "node_count": entry["graph"].node_count,
            "edge_count": entry["graph"].total_edge_count(),
        }
        for gid, entry in _graphs.items()
    ]


# ── Node/Edge CRUD ──────────────────────────────────────────────

def add_node(graph_id: str, label: str,
             node_id: str | None = None) -> TopologicalNode | None:
    graph = get_graph(graph_id)
    if graph is None:
        return None
    if node_id is None:
        node_id = label.lower().replace(" ", "_").replace("/", "_")[:30]
    # Avoid duplicate IDs.
    if node_id in graph.nodes:
        node_id = f"{node_id}_{str(uuid.uuid4())[:4]}"
    return graph.add_node(label=label, node_id=node_id)


def add_edge(graph_id: str, source_id: str, target_id: str,
             edge_type_str: str, weight: float = 1.0) -> bool:
    graph = get_graph(graph_id)
    if graph is None:
        return False
    et = EdgeType(edge_type_str)
    result = graph.add_edge(source_id, target_id, et, weight=weight)
    return result is not None


def delete_node(graph_id: str, node_id: str) -> bool:
    """Remove a node and clean up all edges pointing to it."""
    graph = get_graph(graph_id)
    if graph is None or node_id not in graph._nodes:
        return False
    # Step 1: Remove incoming edges from all other nodes.
    for nid, node in graph._nodes.items():
        if nid != node_id and node_id in node.edges:
            del node.edges[node_id]
    # Step 2: Remove the node itself (and all its outgoing edges).
    del graph._nodes[node_id]
    return True


def delete_edge(graph_id: str, source_id: str, target_id: str,
                edge_type_str: str | None = None) -> bool:
    """Remove edge(s) between two nodes."""
    graph = get_graph(graph_id)
    if graph is None:
        return False
    source = graph.get_node(source_id)
    if source is None or target_id not in source.edges:
        return False
    if edge_type_str:
        source.edges[target_id] = [
            r for r in source.edges[target_id]
            if r.edge_type.value != edge_type_str
        ]
        if not source.edges[target_id]:
            del source.edges[target_id]
    else:
        del source.edges[target_id]
    return True


# ── Graph state serialization (for frontend) ───────────────────

def serialize_graph_state(graph_id: str) -> dict[str, Any] | None:
    entry = _graphs.get(graph_id)
    if entry is None:
        return None
    graph = entry["graph"]
    nodes_out = []
    for nid, node in graph.nodes.items():
        edges_out = []
        for _target_id, rels in node.edges.items():
            for rel in rels:
                edges_out.append({
                    "target_id": rel.target_id,
                    "edge_type": rel.edge_type.value,
                    "weight": round(rel.weight, 3),
                    "grounded": rel.grounded,
                })
        nodes_out.append({"id": nid, "label": node.label, "edges": edges_out})
    return {
        "graph_id": graph_id,
        "name": entry["name"],
        "nodes": nodes_out,
        "node_count": graph.node_count,
        "edge_count": graph.total_edge_count(),
    }


# ── Analysis ────────────────────────────────────────────────────

def run_analysis(graph_id: str) -> dict[str, Any] | None:
    entry = _graphs.get(graph_id)
    if entry is None:
        return None
    graph = entry["graph"]

    if graph.node_count == 0:
        entry["last_solutions"] = []
        return {
            "node_count": 0,
            "edge_count": 0,
            "health": {
                "betweenness": {}, "clustering": {}, "path_diversity": 0.0,
                "cycles": [], "bridges": [], "isolated": [],
            },
            "confidence": {
                "confidence": 0.0, "path_diversity_score": 0.0,
                "cycle_penalty": 0.0, "grounding_ratio": 0.0,
                "convergence_speed": 0.0,
            },
            "solutions": [],
            "solution_count": 0,
        }

    # Health metrics — pass entire graph as the subgraph.
    health = compute_health(graph.nodes)
    # Confidence.
    conf = compute_confidence(graph.nodes, None, None, 1, 100)
    # Solutions.
    solver = TopologicalSolver(graph)
    solutions = solver.solve_all()
    entry["last_solutions"] = solutions

    return {
        "node_count": graph.node_count,
        "edge_count": graph.total_edge_count(),
        "health": {
            "betweenness": {k: round(v, 4) for k, v in health.betweenness.items()},
            "clustering": {k: round(v, 4) for k, v in health.clustering.items()},
            "path_diversity": round(health.path_diversity, 4),
            "cycles": health.cycles,
            "bridges": [list(b) for b in health.bridges],
            "isolated": health.isolated,
        },
        "confidence": {
            "confidence": round(conf.confidence, 4),
            "path_diversity_score": round(conf.path_diversity_score, 4),
            "cycle_penalty": round(conf.cycle_penalty, 4),
            "grounding_ratio": round(conf.grounding_ratio, 4),
            "convergence_speed": round(conf.convergence_speed, 4),
        },
        "solutions": [
            {
                "index": i,
                "problem_type": s.problem_type,
                "problem_description": s.problem_description,
                "action": s.action,
                "details": s.details,
                "reasoning": s.reasoning,
                "confidence": round(s.confidence, 3),
            }
            for i, s in enumerate(solutions)
        ],
        "solution_count": len(solutions),
    }


def apply_solution(graph_id: str, solution_index: int) -> bool:
    entry = _graphs.get(graph_id)
    if entry is None:
        return False
    solutions = entry.get("last_solutions", [])
    if solution_index < 0 or solution_index >= len(solutions):
        return False
    solver = TopologicalSolver(entry["graph"])
    return solver.apply_solution(solutions[solution_index])


# ── Templates ───────────────────────────────────────────────────

def load_template(template_name: str) -> dict | None:
    path = os.path.join(_TEMPLATES_DIR, f"{template_name}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_templates() -> list[dict[str, Any]]:
    results = []
    if not os.path.isdir(_TEMPLATES_DIR):
        return results
    for fname in sorted(os.listdir(_TEMPLATES_DIR)):
        if not fname.endswith(".json"):
            continue
        name = fname[:-5]
        path = os.path.join(_TEMPLATES_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        nodes = data.get("L2_graph", {}).get("nodes", [])
        results.append({
            "name": name,
            "description": data.get("description", ""),
            "node_count": len(nodes),
        })
    return results


# ── Export ──────────────────────────────────────────────────────

def export_state(graph_id: str) -> dict | None:
    graph = get_graph(graph_id)
    if graph is None:
        return None
    return serialize_tca_state(graph=graph)


def export_boot(graph_id: str) -> dict | None:
    graph = get_graph(graph_id)
    if graph is None:
        return None
    return export_boot_protocol(graph)
