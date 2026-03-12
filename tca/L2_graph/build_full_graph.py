"""
TCA Phase 3: Build Full Graph from KD Data

Loads all 1,430 nodes from tca/data/all_nodes.json into L2's
TopologicalGraph with typed edges:
  - shared domain  → INHERITS edge (weight 0.3)
  - shared tags    → MIRRORS edge  (weight per overlap)
  - content refs   → EXPRESSES edge (weight 0.5)

Saves to tca/data/full_graph.json.
"""

from __future__ import annotations

import json
import os
import re
import sys

# Add project root to path for imports.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tca.L2_graph.topo_node import TopologicalNode, EdgeType, EdgeRelation


def load_nodes(path: str) -> list[dict]:
    """Load raw nodes from JSON."""
    with open(path) as f:
        return json.load(f)


def build_graph(raw_nodes: list[dict]) -> dict:
    """Build topological graph from raw KD nodes.

    Returns dict with 'nodes' and 'edges' for serialization,
    plus 'topo_nodes' dict of TopologicalNode objects for runtime.
    """
    topo_nodes: dict[str, TopologicalNode] = {}
    edges: list[dict] = []

    # Index: domain -> list of node IDs
    domain_index: dict[str, list[str]] = {}
    # Index: tag -> list of node IDs
    tag_index: dict[str, list[str]] = {}
    # Index: id -> node data
    node_data: dict[str, dict] = {}

    # Pass 1: Create all TopologicalNodes and build indices.
    for raw in raw_nodes:
        nid = raw["id"]
        node = TopologicalNode(
            id=nid,
            label=raw.get("content", "")[:80],
        )
        topo_nodes[nid] = node
        node_data[nid] = raw

        domain = raw.get("domain", "")
        if domain:
            domain_index.setdefault(domain, []).append(nid)

        for tag in raw.get("tags", []):
            if tag and tag != "untagged":
                tag_index.setdefault(tag, []).append(nid)

    # Pass 2: Build INHERITS edges (shared domain).
    # Cap per-domain to avoid O(n^2) explosion on large domains.
    MAX_DOMAIN_EDGES = 50
    for domain, nids in domain_index.items():
        if len(nids) < 2:
            continue
        # Connect each node to up to MAX_DOMAIN_EDGES neighbors in same domain.
        for i, src in enumerate(nids):
            targets = nids[i + 1:i + 1 + MAX_DOMAIN_EDGES]
            for tgt in targets:
                topo_nodes[src].add_edge(tgt, EdgeType.INHERITS, weight=0.3)
                topo_nodes[tgt].add_edge(src, EdgeType.INHERITS, weight=0.3)
                edges.append({
                    "source": src, "target": tgt,
                    "type": "INHERITS", "weight": 0.3,
                })

    # Pass 3: Build MIRRORS edges (shared tags).
    # Weight = number_of_shared_tags / max(total_tags_a, total_tags_b).
    MAX_TAG_EDGES = 30
    for tag, nids in tag_index.items():
        if len(nids) < 2:
            continue
        for i, src in enumerate(nids):
            targets = nids[i + 1:i + 1 + MAX_TAG_EDGES]
            for tgt in targets:
                src_tags = set(node_data[src].get("tags", []))
                tgt_tags = set(node_data[tgt].get("tags", []))
                shared = src_tags & tgt_tags - {"untagged"}
                if not shared:
                    continue
                denom = max(len(src_tags), len(tgt_tags), 1)
                weight = min(1.0, len(shared) / denom)
                # Only add if not already connected by INHERITS with same target.
                existing_targets = {
                    rel.target_id
                    for rels in topo_nodes[src].edges.values()
                    for rel in rels
                    if rel.edge_type == EdgeType.MIRRORS
                }
                if tgt not in existing_targets:
                    topo_nodes[src].add_edge(tgt, EdgeType.MIRRORS, weight=weight)
                    topo_nodes[tgt].add_edge(src, EdgeType.MIRRORS, weight=weight)
                    edges.append({
                        "source": src, "target": tgt,
                        "type": "MIRRORS", "weight": round(weight, 3),
                    })

    # Pass 4: Build EXPRESSES edges (content reference detection).
    # Look for domain names mentioned in content of other-domain nodes.
    domain_names = set(domain_index.keys())
    for nid, raw in node_data.items():
        content_lower = raw.get("content", "").lower()
        node_domain = raw.get("domain", "")
        for ref_domain in domain_names:
            if ref_domain == node_domain:
                continue
            if len(ref_domain) < 4:
                continue  # Skip very short domain names (noise).
            # Check if domain name appears as a word in content.
            if re.search(r'\b' + re.escape(ref_domain.replace('-', '.')) + r'\b',
                         content_lower):
                # Connect to first node of referenced domain.
                ref_nids = domain_index[ref_domain][:3]
                for ref_nid in ref_nids:
                    existing = {
                        rel.target_id
                        for rels in topo_nodes[nid].edges.values()
                        for rel in rels
                        if rel.edge_type == EdgeType.EXPRESSES
                    }
                    if ref_nid not in existing:
                        topo_nodes[nid].add_edge(ref_nid, EdgeType.EXPRESSES,
                                                  weight=0.5)
                        edges.append({
                            "source": nid, "target": ref_nid,
                            "type": "EXPRESSES", "weight": 0.5,
                        })

    return {
        "topo_nodes": topo_nodes,
        "node_list": [{"id": n["id"], "domain": n.get("domain", ""),
                        "confidence": n.get("confidence", 0)}
                       for n in raw_nodes],
        "edges": edges,
    }


def save_graph(graph_data: dict, output_path: str) -> None:
    """Save graph to JSON (without TopologicalNode objects)."""
    serializable = {
        "nodes": graph_data["node_list"],
        "edges": graph_data["edges"],
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)


def print_stats(graph_data: dict) -> None:
    """Print graph statistics."""
    topo_nodes = graph_data["topo_nodes"]
    edges = graph_data["edges"]

    # Count edges per type.
    type_counts: dict[str, int] = {}
    for e in edges:
        t = e["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    total_edges = sum(n.edge_count() for n in topo_nodes.values())
    avg_edges = total_edges / len(topo_nodes) if topo_nodes else 0

    print(f"\n{'=' * 60}")
    print(f"  TCA FULL GRAPH STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Total nodes:          {len(topo_nodes):,}")
    print(f"  Total edges (stored): {len(edges):,}")
    print(f"  Total edges (bidir):  {total_edges:,}")
    print(f"  Average edges/node:   {avg_edges:.1f}")
    print(f"\n  Edges per type:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {etype:<12} {count:>8,}")
    print(f"{'=' * 60}\n")


def main():
    data_dir = os.path.join(PROJECT_ROOT, "tca", "data")
    input_path = os.path.join(data_dir, "all_nodes.json")
    output_path = os.path.join(data_dir, "full_graph.json")

    print(f"Loading nodes from {input_path}...")
    raw_nodes = load_nodes(input_path)
    print(f"  Loaded {len(raw_nodes)} nodes")

    print("Building graph...")
    graph_data = build_graph(raw_nodes)

    print(f"Saving to {output_path}...")
    save_graph(graph_data, output_path)

    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

    print_stats(graph_data)


if __name__ == "__main__":
    main()
