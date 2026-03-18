"""
TCA Engine — The calculator.

Graph in → structural problems out. Deterministic. Zero AI.
Manages graph instances, runs analysis, applies solutions, exports.
"""

from __future__ import annotations

import uuid
from typing import Any

from tca.graph import EdgeType, TopologicalGraph, TopologicalNode
from tca.analyze import compute_health, compute_confidence
from tca.templates import apply_template

_graphs: dict[str, dict[str, Any]] = {}
_EDGE_TYPE_MAP: dict[str, EdgeType] = {e.value: e for e in EdgeType}


def create_graph(name: str = "Untitled",
                 template: str | None = None) -> dict:
    graph_id = str(uuid.uuid4())[:8]
    g = TopologicalGraph()

    if template:
        apply_template(g, template)

    _graphs[graph_id] = {
        "graph": g,
        "name": name,
        "template": template,
        "solutions": [],
    }
    return {"graph_id": graph_id}


def add_node(graph_id: str, label: str, node_id: str | None = None) -> dict:
    entry = _graphs.get(graph_id)
    if not entry:
        return {"error": f"Graph '{graph_id}' not found"}
    node = entry["graph"].add_node(label=label, node_id=node_id)
    return {"node_id": node.id}


def add_edge(graph_id: str, source: str, target: str,
             edge_type: str, weight: float = 1.0) -> dict:
    entry = _graphs.get(graph_id)
    if not entry:
        return {"error": f"Graph '{graph_id}' not found"}
    et = _EDGE_TYPE_MAP.get(edge_type)
    if et is None:
        return {"error": f"Unknown edge type: {edge_type}. Valid: {list(_EDGE_TYPE_MAP.keys())}"}
    rel = entry["graph"].add_edge(source, target, et, weight)
    if rel is None:
        return {"error": f"Source node '{source}' not found"}
    return {"ok": True}


def run_analysis(graph_id: str) -> dict | None:
    """Run full TCA analysis. Returns problems, questions, solutions."""
    entry = _graphs.get(graph_id)
    if not entry:
        return None

    g: TopologicalGraph = entry["graph"]
    nodes = g.nodes

    if not nodes:
        return {"node_count": 0, "edge_count": 0, "problems": [],
                "questions": [], "solutions": []}

    health = compute_health(nodes)
    conf = compute_confidence(nodes)

    problems = []
    questions = []
    solutions = []

    # Dead ends — nodes with no outgoing edges.
    for nid, node in nodes.items():
        if node.edge_count() == 0:
            problems.append({
                "type": "dead_end",
                "node": nid,
                "label": node.label,
                "description": f"'{node.label}' has no connections — isolated concept.",
            })
            solutions.append({
                "type": "connect_dead_end",
                "description": f"Connect '{node.label}' to related nodes.",
                "target_node": nid,
                "confidence": 0.7,
            })

    # Star topologies — bottleneck nodes.
    if health.betweenness:
        max_btwn = max(health.betweenness.values()) if health.betweenness else 0
        for nid, btwn in health.betweenness.items():
            if btwn > 0.3 and btwn == max_btwn:
                label = nodes[nid].label if nid in nodes else nid
                problems.append({
                    "type": "star_topology",
                    "node": nid,
                    "label": label,
                    "betweenness": round(btwn, 4),
                    "description": f"'{label}' is a bottleneck (betweenness={btwn:.3f}). "
                                   f"If removed, the graph fragments.",
                })
                solutions.append({
                    "type": "add_bypass",
                    "description": f"Add bypass edges around '{label}' to reduce fragility.",
                    "target_node": nid,
                    "confidence": 0.8,
                })

    # Feedback traps — cycles.
    for cycle in health.cycles:
        cycle_labels = [nodes[nid].label if nid in nodes else nid
                        for nid in cycle[:-1]]
        problems.append({
            "type": "feedback_trap",
            "nodes": cycle,
            "labels": cycle_labels,
            "description": f"Circular reasoning: {' → '.join(cycle_labels)}",
        })
        solutions.append({
            "type": "break_cycle",
            "description": f"Ground one edge in the cycle with external evidence.",
            "cycle": cycle,
            "confidence": 0.6,
        })

    # Ungrounded claims — SEEKS edges.
    for nid, node in nodes.items():
        seeks = node.get_edges_by_type(EdgeType.SEEKS)
        for edge in seeks:
            target_label = nodes[edge.target_id].label if edge.target_id in nodes else edge.target_id
            questions.append({
                "from": node.label,
                "to": target_label,
                "type": "SEEKS",
                "description": f"'{node.label}' seeks '{target_label}' — unresolved relationship.",
            })

    # Contradictions — REMOVES edges.
    for nid, node in nodes.items():
        removes = node.get_edges_by_type(EdgeType.REMOVES)
        for edge in removes:
            target_label = nodes[edge.target_id].label if edge.target_id in nodes else edge.target_id
            problems.append({
                "type": "contradiction",
                "from": node.label,
                "to": target_label,
                "description": f"'{node.label}' contradicts '{target_label}'.",
            })
            solutions.append({
                "type": "resolve_contradiction",
                "description": f"Resolve contradiction between '{node.label}' and '{target_label}'.",
                "confidence": 0.5,
            })

    # Isolated subgraphs.
    if health.isolated:
        for nid in health.isolated:
            label = nodes[nid].label if nid in nodes else nid
            problems.append({
                "type": "isolated",
                "node": nid,
                "label": label,
                "description": f"'{label}' is disconnected from the main graph.",
            })

    solutions.sort(key=lambda s: s.get("confidence", 0), reverse=True)
    entry["solutions"] = solutions

    return {
        "node_count": g.node_count,
        "edge_count": g.total_edge_count(),
        "confidence": round(conf.confidence, 4),
        "problems": problems,
        "questions": questions,
        "solutions": solutions,
        "health": {
            "cycles": len(health.cycles),
            "bridges": len(health.bridges),
            "isolated": len(health.isolated),
        },
    }


def apply_solution(graph_id: str, index: int) -> dict | None:
    """Apply a cached solution by index."""
    entry = _graphs.get(graph_id)
    if not entry:
        return None

    solutions = entry.get("solutions", [])
    if index < 0 or index >= len(solutions):
        return {"error": f"Solution index {index} out of range (0-{len(solutions)-1})"}

    sol = solutions[index]
    g: TopologicalGraph = entry["graph"]

    if sol["type"] == "connect_dead_end":
        target = sol["target_node"]
        best = None
        best_count = -1
        for nid, node in g.nodes.items():
            if nid != target and node.edge_count() > best_count:
                best = nid
                best_count = node.edge_count()
        if best:
            g.add_edge(target, best, EdgeType.SEEKS, 0.5)
            g.add_edge(best, target, EdgeType.SEEKS, 0.5)

    elif sol["type"] == "add_bypass":
        target = sol["target_node"]
        node = g.get_node(target)
        if node:
            neighbors = list(node.get_neighbors())
            for i in range(len(neighbors)):
                for j in range(i + 1, min(i + 3, len(neighbors))):
                    g.add_edge(neighbors[i], neighbors[j],
                               EdgeType.MIRRORS, 0.3)

    elif sol["type"] == "break_cycle":
        cycle = sol.get("cycle", [])
        for i in range(len(cycle) - 1):
            src = g.get_node(cycle[i])
            if src and cycle[i + 1] in src.edges:
                for rel in src.edges[cycle[i + 1]]:
                    rel.grounded = True
                break

    elif sol["type"] == "resolve_contradiction":
        pass  # Requires human input.

    return run_analysis(graph_id)


def export_state(graph_id: str) -> dict | None:
    """Export full graph state as JSON."""
    entry = _graphs.get(graph_id)
    if not entry:
        return None

    g: TopologicalGraph = entry["graph"]
    nodes_out = []
    edges_out = []

    for nid, node in g.nodes.items():
        nodes_out.append({"id": nid, "label": node.label})
        for target_id, rels in node.edges.items():
            for rel in rels:
                edges_out.append({
                    "source": nid,
                    "target": target_id,
                    "type": rel.edge_type.value,
                    "weight": round(rel.weight, 4),
                    "grounded": rel.grounded,
                })

    return {
        "graph_id": graph_id,
        "name": entry["name"],
        "template": entry.get("template"),
        "nodes": nodes_out,
        "edges": edges_out,
    }


def export_boot(graph_id: str) -> dict | None:
    """Export topology only — no labels, safe for sharing."""
    entry = _graphs.get(graph_id)
    if not entry:
        return None

    g: TopologicalGraph = entry["graph"]
    nodes_out = []
    edges_out = []

    for nid, node in g.nodes.items():
        nodes_out.append({"id": nid})
        for target_id, rels in node.edges.items():
            for rel in rels:
                edges_out.append({
                    "source": nid,
                    "target": target_id,
                    "type": rel.edge_type.value,
                    "weight": round(rel.weight, 4),
                })

    return {
        "graph_id": graph_id,
        "name": entry["name"],
        "format": "boot",
        "node_count": len(nodes_out),
        "edge_count": len(edges_out),
        "nodes": nodes_out,
        "edges": edges_out,
    }
