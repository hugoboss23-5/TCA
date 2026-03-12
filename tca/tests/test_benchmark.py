"""
TCA Phase 2: Chestohedron vs NULL Routing Benchmark

Replicates the Haiku+chestohedron vs Opus+NULL test at the TCA level.
The topology protocol should outperform NULL routing on complex queries.

5 queries of increasing complexity:
1. Simple factual: "What is topology?"
2. Domain-specific: "How does the chestohedron protocol work?"
3. Cross-domain: "How does topology relate to quantum computing?"
4. Bridging: "What connects neural architecture to political topology?"
5. Meta-recursive: "Is topology-over-substance itself a topological invariant?"
"""

import unittest

from tca.L1_router.router import route as chestohedron_route
from tca.L2_graph.operations import TopologicalGraph
from tca.L2_graph.topo_node import EdgeType
from tca.L4_temporal.reasoning import reason
from tca.L5_metacognition.health import compute_health
from tca.L5_metacognition.confidence import compute_confidence


# NULL router: all gates equal weight.
def null_route(input_text: str) -> dict[str, float]:
    return {
        "MIRROR": 1/7, "INHERIT": 1/7, "BOUND": 1/7,
        "EXPRESS": 1/7, "VERIFY": 1/7, "REMOVE": 1/7, "DARASH": 1/7,
    }


def _build_benchmark_graph() -> TopologicalGraph:
    """Build a rich graph with multiple clusters and bridges for benchmarking."""
    g = TopologicalGraph()

    # Cluster 1: Topology fundamentals
    g.add_node(label="topology mathematics shapes", node_id="topo")
    g.add_node(label="homeomorphism continuous deformation", node_id="homeo")
    g.add_node(label="invariant properties preserved", node_id="inv")
    g.add_node(label="exotic spheres dimension 7 milnor", node_id="exotic")
    g.add_node(label="topological insulator surface conducts bulk insulates", node_id="topo_ins")

    g.add_edge("topo", "homeo", EdgeType.EXPRESSES, weight=0.9, grounded=True)
    g.add_edge("topo", "inv", EdgeType.EXPRESSES, weight=0.8, grounded=True)
    g.add_edge("homeo", "inv", EdgeType.VERIFIES, weight=0.7, grounded=True)
    g.add_edge("topo", "exotic", EdgeType.EXPRESSES, weight=0.6)
    g.add_edge("exotic", "inv", EdgeType.VERIFIES, weight=0.5)
    g.add_edge("topo_ins", "inv", EdgeType.VERIFIES, weight=0.7, grounded=True)
    g.add_edge("topo", "topo_ins", EdgeType.EXPRESSES, weight=0.5)

    # Cluster 2: Chestohedron / Protocol
    g.add_node(label="chestohedron protocol seven gates cognitive", node_id="chesto")
    g.add_node(label="V9 protocol MIRROR INHERIT BOUND EXPRESS VERIFY REMOVE", node_id="v9")
    g.add_node(label="topology over substance protocol effect 10x substrate", node_id="tos")
    g.add_node(label="NULL routing baseline equal weights", node_id="null_base")
    g.add_node(label="physics benchmark NULL 17/20 chestohedron 20/20", node_id="physics")

    g.add_edge("chesto", "v9", EdgeType.EXPRESSES, weight=0.9, grounded=True)
    g.add_edge("chesto", "tos", EdgeType.VERIFIES, weight=0.8, grounded=True)
    g.add_edge("v9", "null_base", EdgeType.REMOVES, weight=0.7, grounded=True)
    g.add_edge("physics", "tos", EdgeType.VERIFIES, weight=0.9, grounded=True)
    g.add_edge("physics", "null_base", EdgeType.REMOVES, weight=0.6)
    g.add_edge("chesto", "physics", EdgeType.VERIFIES, weight=0.7)

    # Cluster 3: Quantum computing
    g.add_node(label="quantum computing qubits circuits", node_id="quantum")
    g.add_node(label="circle topology ring beats all-to-all", node_id="circle")
    g.add_node(label="parameterized circuit topology study", node_id="pct")
    g.add_node(label="over-entanglement harmful tetrahedron", node_id="overent")

    g.add_edge("quantum", "circle", EdgeType.EXPRESSES, weight=0.8, grounded=True)
    g.add_edge("quantum", "pct", EdgeType.VERIFIES, weight=0.7)
    g.add_edge("circle", "pct", EdgeType.VERIFIES, weight=0.8, grounded=True)
    g.add_edge("overent", "circle", EdgeType.REMOVES, weight=0.6)
    g.add_edge("quantum", "overent", EdgeType.EXPRESSES, weight=0.5)

    # Cluster 4: Neural architecture
    g.add_node(label="neural architecture CCN geometric cycling", node_id="neural")
    g.add_node(label="topology precision scaling law FPGA", node_id="tps")
    g.add_node(label="pre-motor neural activation anxiety claude", node_id="premotor")
    g.add_node(label="GCN 331776 fixed topology parameters chestohedron matrices", node_id="gcn")

    g.add_edge("neural", "tps", EdgeType.EXPRESSES, weight=0.8)
    g.add_edge("neural", "premotor", EdgeType.SEEKS, weight=0.5)
    g.add_edge("neural", "gcn", EdgeType.INHERITS, weight=0.7, grounded=True)
    g.add_edge("tps", "gcn", EdgeType.VERIFIES, weight=0.6)

    # Cluster 5: Political topology
    g.add_node(label="political topology constraint founding gate economic miracle", node_id="pol")
    g.add_node(label="amazon topology scaling codec 5 gates", node_id="amazon")
    g.add_node(label="renaissance technologies closed system topology", node_id="ren")

    g.add_edge("pol", "amazon", EdgeType.EXPRESSES, weight=0.7)
    g.add_edge("pol", "ren", EdgeType.EXPRESSES, weight=0.7)
    g.add_edge("amazon", "ren", EdgeType.MIRRORS, weight=0.5)

    # BRIDGES between clusters (cross-domain connections)
    g.add_edge("topo", "chesto", EdgeType.INHERITS, weight=0.6)  # topology -> chestohedron
    g.add_edge("circle", "tos", EdgeType.VERIFIES, weight=0.7)   # quantum circle -> topology over substance
    g.add_edge("topo", "quantum", EdgeType.SEEKS, weight=0.4)    # topology -> quantum
    g.add_edge("neural", "chesto", EdgeType.INHERITS, weight=0.5) # neural -> chestohedron
    g.add_edge("pol", "tos", EdgeType.VERIFIES, weight=0.5)      # political -> topology over substance
    g.add_edge("topo", "neural", EdgeType.SEEKS, weight=0.3)     # topology -> neural
    g.add_edge("tos", "inv", EdgeType.MIRRORS, weight=0.8, grounded=True)  # tos IS an invariant

    # Meta-recursive edge: topology-over-substance is itself topological
    g.add_node(label="self-referential topology finding topology chestohedron operating on itself", node_id="meta")
    g.add_edge("meta", "tos", EdgeType.VERIFIES, weight=0.9, grounded=True)
    g.add_edge("meta", "inv", EdgeType.MIRRORS, weight=0.7)
    g.add_edge("meta", "chesto", EdgeType.MIRRORS, weight=0.6)
    g.add_edge("tos", "meta", EdgeType.SEEKS, weight=0.5)

    return g


# The 5 benchmark queries.
QUERIES = [
    ("Simple", "What is topology?"),
    ("Domain", "How does the chestohedron protocol work?"),
    ("Cross-domain", "How does topology relate to quantum computing?"),
    ("Bridging", "What connects neural architecture to political topology?"),
    ("Meta-recursive", "Is topology-over-substance itself a topological invariant?"),
]


def _run_benchmark(graph: TopologicalGraph, router_fn, router_name: str,
                   max_ticks: int = 50) -> list[dict]:
    """Run all 5 queries with a given router and collect metrics."""
    results = []
    for label, query in QUERIES:
        # Use the specified router.
        gate_weights = router_fn(query)

        # Find entry nodes.
        entry_nodes = graph.find_entry(query)
        if not entry_nodes:
            results.append({
                "label": label, "query": query, "router": router_name,
                "ticks": 0, "confidence": 0.0, "nodes": 0, "bridges": 0,
            })
            continue

        # Run reasoning.
        result = reason(query, graph, max_ticks=max_ticks)

        # Compute metrics.
        subgraph = graph.subgraph([nid for nid, _ in result.activated_nodes])
        health = compute_health(subgraph)
        conf = compute_confidence(subgraph,
                                  ticks_used=result.ticks_used,
                                  max_ticks=max_ticks)

        results.append({
            "label": label,
            "query": query,
            "router": router_name,
            "ticks": result.ticks_used,
            "confidence": conf.confidence,
            "nodes": len(result.activated_nodes),
            "bridges": health.bridge_count,
            "cycles": health.cycle_count,
            "grounding": conf.grounding_ratio,
        })

    return results


class TestBenchmark(unittest.TestCase):
    """Benchmark: Chestohedron vs NULL routing."""

    def test_chestohedron_vs_null(self):
        """Compare chestohedron routing against NULL baseline."""
        graph = _build_benchmark_graph()

        # Run both routers.
        chesto_results = _run_benchmark(graph, chestohedron_route, "CHESTOHEDRON")
        null_results = _run_benchmark(graph, null_route, "NULL")

        # Print results table.
        print("\n  " + "=" * 90)
        print("  TCA BENCHMARK: CHESTOHEDRON vs NULL ROUTING")
        print("  " + "=" * 90)
        print(f"  {'Query':<15} {'Router':<14} {'Ticks':>5} {'Conf':>6} "
              f"{'Nodes':>5} {'Bridges':>7} {'Cycles':>6} {'Ground':>6}")
        print("  " + "-" * 90)

        for c, n in zip(chesto_results, null_results):
            print(f"  {c['label']:<15} {'CHESTOHEDRON':<14} "
                  f"{c['ticks']:>5} {c['confidence']:>6.3f} "
                  f"{c['nodes']:>5} {c['bridges']:>7} "
                  f"{c['cycles']:>6} {c['grounding']:>6.3f}")
            print(f"  {'':<15} {'NULL':<14} "
                  f"{n['ticks']:>5} {n['confidence']:>6.3f} "
                  f"{n['nodes']:>5} {n['bridges']:>7} "
                  f"{n['cycles']:>6} {n['grounding']:>6.3f}")
            print("  " + "-" * 90)

        # Compute averages.
        chesto_avg_conf = sum(r["confidence"] for r in chesto_results) / len(chesto_results)
        null_avg_conf = sum(r["confidence"] for r in null_results) / len(null_results)
        chesto_avg_nodes = sum(r["nodes"] for r in chesto_results) / len(chesto_results)
        null_avg_nodes = sum(r["nodes"] for r in null_results) / len(null_results)

        print(f"\n  AVERAGES:")
        print(f"    CHESTOHEDRON: confidence={chesto_avg_conf:.3f}, "
              f"nodes={chesto_avg_nodes:.1f}")
        print(f"    NULL:         confidence={null_avg_conf:.3f}, "
              f"nodes={null_avg_nodes:.1f}")

        # The topology protocol should show differentiated gate activation
        # on complex queries, leading to more targeted node activation.
        # On simple queries, both should perform similarly.

    def test_complex_queries_benefit_from_routing(self):
        """Queries with keyword signals show gate differentiation vs NULL."""
        from tca.L1_router.router import route as l1_route

        queries_with_spread = 0
        for label, query in QUERIES:
            chesto_weights = l1_route(query)
            null_weights = null_route(query)

            max_gate = max(chesto_weights, key=lambda g: chesto_weights[g])
            min_gate = min(chesto_weights, key=lambda g: chesto_weights[g])
            spread = chesto_weights[max_gate] - chesto_weights[min_gate]
            null_spread = max(null_weights.values()) - min(null_weights.values())

            if spread > null_spread:
                queries_with_spread += 1

        # At least 3 of 5 queries should trigger keyword patterns that
        # differentiate the chestohedron from NULL routing.
        self.assertGreaterEqual(queries_with_spread, 3,
                                f"At least 3/5 queries should show gate "
                                f"differentiation, got {queries_with_spread}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
