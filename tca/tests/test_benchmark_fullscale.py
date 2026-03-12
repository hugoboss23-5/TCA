"""
TCA Phase 3: Full-Scale Benchmark — Chestohedron vs NULL on 1,430 nodes

5 queries, each run twice (chestohedron router, NULL router).
Metrics: tick count, nodes activated, domains touched, bridges found,
         confidence, cycles detected.

PREDICTION: Chestohedron should outperform NULL on queries 3-5 (cross-domain)
but show minimal difference on 1-2 (simple/factual).
"""

from __future__ import annotations

import json
import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tca.L1_router.router import route as chestohedron_route
from tca.L2_graph.operations import TopologicalGraph
from tca.L2_graph.topo_node import EdgeType
from tca.L4_temporal.reasoning import reason
from tca.L5_metacognition.health import compute_health
from tca.L5_metacognition.confidence import compute_confidence


def null_route(input_text: str) -> dict[str, float]:
    """NULL router: all gates equal weight."""
    return {
        "MIRROR": 1/7, "INHERIT": 1/7, "BOUND": 1/7,
        "EXPRESS": 1/7, "VERIFY": 1/7, "REMOVE": 1/7, "DARASH": 1/7,
    }


def load_full_graph() -> TopologicalGraph:
    """Load the full 1,430-node graph from disk."""
    data_dir = os.path.join(PROJECT_ROOT, "tca", "data")
    graph_path = os.path.join(data_dir, "full_graph.json")
    nodes_path = os.path.join(data_dir, "all_nodes.json")

    with open(graph_path) as f:
        graph_json = json.load(f)
    with open(nodes_path) as f:
        all_nodes = json.load(f)

    # Build metadata index.
    meta = {}
    for n in all_nodes:
        meta[n["id"]] = n

    # Rebuild TopologicalGraph.
    g = TopologicalGraph()
    for node_entry in graph_json["nodes"]:
        nid = node_entry["id"]
        content = meta.get(nid, {}).get("content", "")
        g.add_node(label=content[:80], node_id=nid)

    for edge in graph_json["edges"]:
        etype = EdgeType[edge["type"]]
        g.add_edge(edge["source"], edge["target"], etype, weight=edge["weight"])
        g.add_edge(edge["target"], edge["source"], etype, weight=edge["weight"])

    return g, meta


# The 5 benchmark queries.
QUERIES = [
    ("Q1: What is KD?",
     "What is KD?"),
    ("Q2: Reservoir comp",
     "How does reservoir computing work?"),
    ("Q3: Topology+consc",
     "How does topology relate to consciousness?"),
    ("Q4: Chesto+econ",
     "What is the relationship between the chestohedron and economic measurement?"),
    ("Q5: TRINITY+CCN",
     "How should TRINITY evolve based on CCN benchmarks?"),
]


def run_single_query(graph: TopologicalGraph, meta: dict,
                     query: str, router_fn, max_ticks: int = 50) -> dict:
    """Run one query with a given router and collect metrics."""

    # Monkey-patch L1 router for this run.
    import tca.L4_temporal.reasoning as reasoning_mod
    original_route = reasoning_mod.l1_route
    reasoning_mod.l1_route = router_fn

    try:
        result = reason(query, graph, max_ticks=max_ticks)
    finally:
        reasoning_mod.l1_route = original_route

    # Compute metrics.
    activated_ids = [nid for nid, _ in result.activated_nodes]
    subgraph = graph.subgraph(activated_ids)

    # Count domains touched.
    domains_touched = set()
    for nid in activated_ids:
        d = meta.get(nid, {}).get("domain", "")
        if d:
            domains_touched.add(d)

    # Compute health and confidence.
    # Cap subgraph size for health computation (O(n^2) algorithms).
    MAX_HEALTH_NODES = 50
    if subgraph:
        if len(subgraph) > MAX_HEALTH_NODES:
            # Keep top-activated nodes only for health metrics.
            top_ids = sorted(subgraph.keys(),
                             key=lambda n: subgraph[n].activation,
                             reverse=True)[:MAX_HEALTH_NODES]
            health_subgraph = {nid: subgraph[nid] for nid in top_ids}
        else:
            health_subgraph = subgraph
        health = compute_health(health_subgraph)
        conf = compute_confidence(health_subgraph,
                                  ticks_used=result.ticks_used,
                                  max_ticks=max_ticks)
    else:
        from tca.L5_metacognition.health import HealthReport
        from tca.L5_metacognition.confidence import ConfidenceReport
        health = HealthReport()
        conf = ConfidenceReport(confidence=0, path_diversity_score=0,
                                cycle_penalty=0, grounding_ratio=0,
                                convergence_speed=1.0)

    return {
        "ticks": result.ticks_used,
        "nodes": len(result.activated_nodes),
        "domains": len(domains_touched),
        "bridges": health.bridge_count,
        "confidence": conf.confidence,
        "cycles": health.cycle_count,
        "grounding": conf.grounding_ratio,
    }


class TestBenchmarkFullScale(unittest.TestCase):
    """Full-scale benchmark: Chestohedron vs NULL on 1,430 nodes."""

    @classmethod
    def setUpClass(cls):
        print("\n  Loading full graph (1,430 nodes)...")
        cls.graph, cls.meta = load_full_graph()
        print(f"  Graph loaded: {cls.graph.node_count} nodes, "
              f"{cls.graph.total_edge_count()} edges")

    def test_fullscale_benchmark(self):
        """Run all 5 queries with both routers and print comparison table."""
        all_results = []

        for label, query in QUERIES:
            # Chestohedron router.
            chesto = run_single_query(self.graph, self.meta, query,
                                      chestohedron_route, max_ticks=50)
            chesto["label"] = label
            chesto["router"] = "CHESTOHEDRON"

            # Reset activations before NULL run.
            for node in self.graph.nodes.values():
                node.activation = 0.0

            # NULL router.
            null = run_single_query(self.graph, self.meta, query,
                                    null_route, max_ticks=50)
            null["label"] = label
            null["router"] = "NULL"

            # Reset for next query.
            for node in self.graph.nodes.values():
                node.activation = 0.0

            all_results.append((chesto, null))

        # Print results table.
        print(f"\n  {'=' * 100}")
        print(f"  TCA FULL-SCALE BENCHMARK: CHESTOHEDRON vs NULL (1,430 nodes)")
        print(f"  {'=' * 100}")
        print(f"  {'Query':<20} {'Router':<14} {'Ticks':>5} {'Nodes':>6} "
              f"{'Domains':>7} {'Bridges':>7} {'Cycles':>6} "
              f"{'Conf':>6} {'Ground':>6}")
        print(f"  {'-' * 100}")

        chesto_total = {"ticks": 0, "nodes": 0, "domains": 0,
                        "bridges": 0, "cycles": 0, "confidence": 0,
                        "grounding": 0}
        null_total = {"ticks": 0, "nodes": 0, "domains": 0,
                      "bridges": 0, "cycles": 0, "confidence": 0,
                      "grounding": 0}

        for c, n in all_results:
            print(f"  {c['label']:<20} {'CHESTO':<14} "
                  f"{c['ticks']:>5} {c['nodes']:>6} "
                  f"{c['domains']:>7} {c['bridges']:>7} "
                  f"{c['cycles']:>6} {c['confidence']:>6.3f} "
                  f"{c['grounding']:>6.3f}")
            print(f"  {'':<20} {'NULL':<14} "
                  f"{n['ticks']:>5} {n['nodes']:>6} "
                  f"{n['domains']:>7} {n['bridges']:>7} "
                  f"{n['cycles']:>6} {n['confidence']:>6.3f} "
                  f"{n['grounding']:>6.3f}")
            print(f"  {'-' * 100}")

            for k in chesto_total:
                chesto_total[k] += c[k]
                null_total[k] += n[k]

        nq = len(all_results)
        print(f"\n  AVERAGES (across {nq} queries):")
        print(f"    CHESTOHEDRON: ticks={chesto_total['ticks']/nq:.1f}  "
              f"nodes={chesto_total['nodes']/nq:.1f}  "
              f"domains={chesto_total['domains']/nq:.1f}  "
              f"bridges={chesto_total['bridges']/nq:.1f}  "
              f"cycles={chesto_total['cycles']/nq:.1f}  "
              f"conf={chesto_total['confidence']/nq:.3f}  "
              f"ground={chesto_total['grounding']/nq:.3f}")
        print(f"    NULL:         ticks={null_total['ticks']/nq:.1f}  "
              f"nodes={null_total['nodes']/nq:.1f}  "
              f"domains={null_total['domains']/nq:.1f}  "
              f"bridges={null_total['bridges']/nq:.1f}  "
              f"cycles={null_total['cycles']/nq:.1f}  "
              f"conf={null_total['confidence']/nq:.3f}  "
              f"ground={null_total['grounding']/nq:.3f}")

        # Prediction check: chestohedron should outperform on Q3-Q5.
        print(f"\n  PREDICTION CHECK:")
        print(f"    Expected: Chestohedron outperforms NULL on Q3-Q5, "
              f"minimal diff on Q1-Q2")
        for i, (c, n) in enumerate(all_results):
            label = c["label"]
            diff = c["confidence"] - n["confidence"]
            winner = "CHESTO" if diff > 0 else "NULL" if diff < 0 else "TIE"
            print(f"    {label}: Δconf={diff:+.3f}  → {winner}")

        print(f"  {'=' * 100}")

        # Ensure both routers produce results (no crashes).
        for c, n in all_results:
            self.assertGreaterEqual(c["ticks"], 0)
            self.assertGreaterEqual(n["ticks"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
