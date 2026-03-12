"""
TCA Phase 3: KD Grounding Cycles

Loads the full graph, picks 50 nodes from 3 different domains,
and runs grounding cycles: predict whether INHERITS edges are valid
by checking if neighbors share the same domain in the source data.

Validates edges, updates weights, reports grounding ratios.
"""

from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tca.L2_graph.topo_node import TopologicalNode, EdgeType, EdgeRelation
from tca.L2_graph.operations import TopologicalGraph
from tca.L3_grounding.prediction_loop import GroundingLoop


def load_full_graph(graph_path: str, nodes_path: str):
    """Load graph JSON and rebuild TopologicalGraph + node metadata.

    Returns (TopologicalGraph, node_metadata dict).
    """
    with open(graph_path) as f:
        graph_json = json.load(f)
    with open(nodes_path) as f:
        all_nodes = json.load(f)

    # Build node metadata index.
    meta = {}
    for n in all_nodes:
        meta[n["id"]] = n

    # Rebuild TopologicalGraph.
    g = TopologicalGraph()
    for node_entry in graph_json["nodes"]:
        g.add_node(
            label=meta.get(node_entry["id"], {}).get("content", "")[:80],
            node_id=node_entry["id"],
        )

    for edge in graph_json["edges"]:
        etype = EdgeType[edge["type"]]
        g.add_edge(edge["source"], edge["target"], etype,
                   weight=edge["weight"])
        # Add reverse edge.
        g.add_edge(edge["target"], edge["source"], etype,
                   weight=edge["weight"])

    return g, meta


def pick_grounding_nodes(meta: dict, domains: list[str],
                         per_domain: int = 17) -> list[str]:
    """Pick nodes from specified domains for grounding."""
    selected = []
    for domain in domains:
        domain_nodes = [nid for nid, m in meta.items()
                        if m.get("domain") == domain]
        selected.extend(domain_nodes[:per_domain])
    return selected


def count_grounded_edges(graph: TopologicalGraph) -> tuple[int, int]:
    """Count (grounded_edges, total_edges) across entire graph."""
    total = 0
    grounded = 0
    for node in graph.nodes.values():
        for rels in node.edges.values():
            for rel in rels:
                total += 1
                if rel.grounded:
                    grounded += 1
    return grounded, total


def run_grounding(graph: TopologicalGraph, meta: dict,
                  node_ids: list[str]) -> dict:
    """Run grounding cycles on selected nodes.

    For each node, check its INHERITS edges: if the neighbor shares
    the same domain, prediction = 1.0 (valid), actual = 1.0.
    If different domain, prediction = 1.0, actual = 0.0 (invalid).

    Returns stats dict.
    """
    loop = GroundingLoop(graph)

    grounded_before, total_before = count_grounded_edges(graph)
    edges_validated = 0
    edges_weakened = 0
    total_checked = 0

    for nid in node_ids:
        node = graph.get_node(nid)
        if node is None:
            continue
        node_domain = meta.get(nid, {}).get("domain", "")

        inherits_edges = node.get_edges_by_type(EdgeType.INHERITS)
        for rel in inherits_edges:
            target_domain = meta.get(rel.target_id, {}).get("domain", "")
            total_checked += 1

            # Prediction: use node's source confidence as prediction
            # (how confident we are the edge is valid).
            node_conf = meta.get(nid, {}).get("confidence", 0.5)
            prediction = min(0.9, node_conf * 0.85 + 0.1)
            # Actual: do they share domain?
            actual = 1.0 if target_domain == node_domain else 0.0

            result = loop.ground(nid, rel.target_id, "domain_check",
                                 prediction=prediction, actual=actual)

            if actual == 1.0:
                edges_validated += 1
            else:
                edges_weakened += 1

    grounded_after, total_after = count_grounded_edges(graph)

    # Compute average confidence before/after on grounded nodes.
    conf_values = []
    for nid in node_ids:
        m = meta.get(nid, {})
        conf_values.append(m.get("confidence", 0.5))
    avg_conf_before = sum(conf_values) / len(conf_values) if conf_values else 0

    # After grounding, confidence is same (it's source data), but
    # grounding ratio changes.
    grounding_ratio = grounded_after / total_after if total_after > 0 else 0

    return {
        "nodes_grounded": len(node_ids),
        "edges_checked": total_checked,
        "edges_validated": edges_validated,
        "edges_weakened": edges_weakened,
        "grounded_before": grounded_before,
        "grounded_after": grounded_after,
        "total_edges": total_after,
        "grounding_ratio": grounding_ratio,
        "avg_confidence_before": avg_conf_before,
    }


def main():
    data_dir = os.path.join(PROJECT_ROOT, "tca", "data")
    graph_path = os.path.join(data_dir, "full_graph.json")
    nodes_path = os.path.join(data_dir, "all_nodes.json")

    print("Loading full graph...")
    graph, meta = load_full_graph(graph_path, nodes_path)
    print(f"  Nodes: {graph.node_count}, Edges: {graph.total_edge_count()}")

    # Pick 3 domains with the most nodes.
    domain_counts = {}
    for m in meta.values():
        d = m.get("domain", "")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    top_domains = sorted(domain_counts.items(), key=lambda x: -x[1])[:3]
    domains = [d for d, _ in top_domains]
    print(f"\n  Grounding domains: {domains}")
    print(f"  Domain sizes: {[c for _, c in top_domains]}")

    # Pick 50 nodes (~17 per domain).
    node_ids = pick_grounding_nodes(meta, domains, per_domain=17)
    # Trim to exactly 50.
    node_ids = node_ids[:50]
    print(f"  Selected {len(node_ids)} nodes for grounding")

    print("\nRunning grounding cycles...")
    stats = run_grounding(graph, meta, node_ids)

    print(f"\n{'=' * 60}")
    print(f"  TCA GROUNDING RESULTS")
    print(f"{'=' * 60}")
    print(f"  Nodes grounded:          {stats['nodes_grounded']}")
    print(f"  Edges checked:           {stats['edges_checked']}")
    print(f"  Edges validated:         {stats['edges_validated']}")
    print(f"  Edges weakened:          {stats['edges_weakened']}")
    print(f"  Grounded edges before:   {stats['grounded_before']}")
    print(f"  Grounded edges after:    {stats['grounded_after']}")
    print(f"  Total edges:             {stats['total_edges']}")
    print(f"  Grounding ratio:         {stats['grounding_ratio']:.4f}")
    print(f"  Avg confidence (source): {stats['avg_confidence_before']:.3f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
