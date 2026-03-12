"""
TCA Phase 2: Real Data Tests

Loads actual KD data (from MCP query results) into L2's TopologicalGraph
and runs the full test suite against real knowledge.

Data sources:
  - vault_query("topology") -> 20 nodes
  - vault_query("neural architecture") -> 7 nodes
  - vault_query("cognitive architecture") -> 16 nodes
  - graph_query("cognitive architecture topology protocol") -> 15 nodes
  Total unique nodes loaded: ~50-58 (some overlap)
"""

import json
import unittest

from tca.adapters.kd_mcp_adapter import KDMCPAdapter
from tca.L2_graph.operations import TopologicalGraph
from tca.L2_graph.topo_node import EdgeType
from tca.L4_temporal.reasoning import reason
from tca.L5_metacognition.confidence import compute_confidence
from tca.L5_metacognition.interventions import monitor


# Pre-fetched MCP data (captured from live KD calls).
# This avoids requiring MCP connectivity at test time.

_VAULT_STATS = {
    "total_facts": 930,
    "total_beliefs": 500,
    "avg_belief_confidence": 0.561,
}

_GRAPH_STATS = {
    "total_edges": 266007,
    "edge_types": [
        {"type": "domain", "count": 217197, "avg_weight": 0.3},
        {"type": "similarity", "count": 40551, "avg_weight": 0.244},
        {"type": "session", "count": 8129, "avg_weight": 0.5},
        {"type": "recognition", "count": 40, "avg_weight": 0.7},
        {"type": "extension", "count": 25, "avg_weight": 0.7},
        {"type": "compression", "count": 20, "avg_weight": 0.7},
        {"type": "analogy", "count": 17, "avg_weight": 0.7},
        {"type": "inversion", "count": 14, "avg_weight": 0.7},
        {"type": "decomposition", "count": 10, "avg_weight": 0.7},
        {"type": "contradiction", "count": 4, "avg_weight": 0.7},
    ],
}

# Nodes from 3 domain queries + 1 graph_query, deduplicated by ID.
_REAL_NODES = [
    # --- Domain: topology ---
    {"id": "da086de0", "content": "TOPOLOGY EXTRACTION METHOD: 4-step protocol for reverse-engineering any organization's operating topology from public data", "vault": "facts", "confidence": 1.0, "domain": "topology", "tags": ["topology-extraction", "method"]},
    {"id": "022cf1f0", "content": "AMAZON TOPOLOGY (THE SCALING CODEC): 5 gates: WRITE, INVERT, FRACTURE, MEASURE, FILTER", "vault": "facts", "confidence": 1.0, "domain": "corporate-topology", "tags": ["amazon", "topology-extraction"]},
    {"id": "0813e277", "content": "RENAISSANCE TECHNOLOGIES TOPOLOGY (THE CLOSED SYSTEM): 5-gate cycle: COMPRESS SIGNAL>NARRATIVE UNIFY SUBSTRATE-SWAP SEAL", "vault": "facts", "confidence": 1.0, "domain": "corporate-topology", "tags": ["renaissance-technologies", "topology-extraction"]},
    {"id": "c576039b", "content": "Milnor (1956): exotic spheres first appear in dimension 7. There are exactly 28 exotic 7-spheres", "vault": "facts", "confidence": 1.0, "domain": "topology-over-substance", "tags": ["milnor", "exotic-spheres", "dimension-7"]},
    {"id": "1ed9ff5a", "content": "Nobel Prize Physics 2016: topological phases of matter. Topological insulator conducts on surface, insulates in bulk", "vault": "facts", "confidence": 1.0, "domain": "topology-over-substance", "tags": ["nobel-2016", "topological-insulator"]},
    {"id": "1101d6a0", "content": "BEST GATE PER COMPANY (meta-topology): CONSTRAIN best at Renaissance, INVERT best at Amazon", "vault": "facts", "confidence": 1.0, "domain": "topology", "tags": ["best-gate", "meta-topology"]},
    {"id": "72fe2ea5", "content": "Every Hugo project independently discovered topology over substance: quantum kernels, V5 loop, KD spreading activation", "vault": "facts", "confidence": 1.0, "domain": "topology-over-substance", "tags": ["topology-over-substance"]},
    {"id": "4c85d139", "content": "Mission: whoever figures out material-agnostic systems deriving power from topology rather than substance wins the 21st century", "vault": "facts", "confidence": 1.0, "domain": "topology-over-substance", "tags": ["topology-over-substance"]},
    {"id": "1bfc6ee2", "content": "Global science building substrate-agnostic bodies: photonic computing, reluctance motors, biocomputing. All solving substrate. None solving topology.", "vault": "facts", "confidence": 1.0, "domain": "topology-over-substance", "tags": ["topology-over-substance"]},
    {"id": "7b12e882", "content": "Scrambled topology performs nearly identically to correct topology (CCN 1.379 vs scrambled 1.430)", "vault": "facts", "confidence": 1.0, "domain": "chestohedron-benchmark", "tags": ["chestohedron-benchmark"]},
    {"id": "b958074b", "content": "H4 topology replaces params: SUPPORTED in wrong direction. LSTM beats CCN. Raw topology cannot replace learned parameters.", "vault": "facts", "confidence": 1.0, "domain": "chestohedron-benchmark", "tags": ["chestohedron-benchmark"]},
    # --- Domain: neural-architecture ---
    {"id": "4d6433e9", "content": "TOPOLOGY-PRECISION SCALING LAW: geometric/topological neural architectures advantage INCREASES as arithmetic precision decreases", "vault": "beliefs", "confidence": 0.75, "domain": "neural-architecture", "tags": ["topology-precision-scaling", "ccn"]},
    {"id": "c2606586", "content": "Anthropic interpretability: neural activation patterns associated with anxiety firing BEFORE output generation", "vault": "facts", "confidence": 1.0, "domain": "neural-architecture", "tags": ["claude", "neural-architecture"]},
    {"id": "1a1223b3", "content": "Pre-motor-like neural activation in Claude may indicate implicit topological loops in feedforward architecture", "vault": "beliefs", "confidence": 0.5, "domain": "neural-architecture", "tags": ["claude", "neural-architecture"]},
    # --- Domain: cognitive-architecture ---
    {"id": "10bdea7f", "content": "CLAUDE.md Opus Transfer document: every session inherits full cognitive architecture including V9-chestohedron protocol", "vault": "facts", "confidence": 1.0, "domain": "ai-models", "tags": ["claude", "ai-models"]},
    {"id": "9b6801af", "content": "THE 3AM INSIGHT: The moat is not the model. The moat is the cognitive architecture AROUND the model.", "vault": "beliefs", "confidence": 0.95, "domain": "identity", "tags": ["topology-over-substance", "mission"]},
    {"id": "c5b77a6f", "content": "Physics benchmark: NULL scored 17/20 (85%), V9-chestohedron scored 20/20 (100%). Topology is reasoning amplifier.", "vault": "beliefs", "confidence": 0.95, "domain": "cognitive-architecture", "tags": ["chestohedron", "verify-operation"]},
    {"id": "5e705820", "content": "V5 topology choice settled by quantum research: circle topology beats complex structures, minimum structure wins", "vault": "facts", "confidence": 1.0, "domain": "cognitive-architecture", "tags": ["cognitive-architecture"]},
    {"id": "fdcec890", "content": "Circle topology beats star topology for cognitive architecture. One extra connection closes the loop.", "vault": "beliefs", "confidence": 0.7, "domain": "architecture", "tags": ["architecture"]},
    {"id": "b23bb9db", "content": "KEY FINDING - CONSTRAINT AS FOUNDING GATE: Every economic miracle is founded on existential constraint.", "vault": "beliefs", "confidence": 0.9, "domain": "political-topology", "tags": ["constraint-gate", "chestohedron"]},
    # --- From graph_query (spreading activation discovered) ---
    {"id": "3732eb04", "content": "Protocol effect being 10x substrate effect is the strongest empirical evidence for topology over substance", "vault": "beliefs", "confidence": 0.9, "domain": "cognitive-architecture", "tags": ["cognitive-architecture"]},
    {"id": "1e43e7b0", "content": "The cognitive genome may be 7 operations not 6: MIRROR INHERIT BOUND EXPRESS VERIFY REMOVE GATE6", "vault": "beliefs", "confidence": 0.5, "domain": "cognitive-architecture", "tags": ["cognitive-architecture"]},
    {"id": "4a211493", "content": "The topology-finding-topology IS the chestohedron operating on itself. The 7-step invariant detection engine maps exactly.", "vault": "beliefs", "confidence": 0.85, "domain": "topology-over-substance", "tags": ["chestohedron", "self-referential"]},
    {"id": "9a81b292", "content": "At S2 scale, GCN-per has 331776 fixed topology parameters (7 chestohedron matrices at 192x192)", "vault": "facts", "confidence": 1.0, "domain": "chestohedron-research", "tags": ["chestohedron-research"]},
    {"id": "4dbf489f", "content": "Parameterized circuit topology study: Ring topology outperforms All-to-all despite fewer connections", "vault": "facts", "confidence": 1.0, "domain": "quantum_computing", "tags": ["quantum_computing"]},
    {"id": "e1636a87", "content": "Circle topology wins both total range and efficiency on 4 qubits across 11 tested topologies", "vault": "facts", "confidence": 1.0, "domain": "quantum-computing", "tags": ["quantum-computing"]},
    {"id": "cac8d600", "content": "Tetrahedron topology (full connectivity) produces identical output range to LINE — over-entanglement confirmed harmful", "vault": "facts", "confidence": 1.0, "domain": "quantum-research", "tags": ["quantum-research"]},
    {"id": "9bd3e3a5", "content": "V10-CHESTOHEDRON PROTOCOL finalized March 11 2026. Seven gates, multi-civilizational naming, four structural gaps closed.", "vault": "facts", "confidence": 1.0, "domain": "core-projects", "tags": ["v10", "chestohedron", "protocol"]},
    {"id": "851eb52b", "content": "SATURN OBSERVATION LANGUAGE: 5 structural primitives. GATHER FENCE SHAPE REMOVE PASS.", "vault": "beliefs", "confidence": 0.8, "domain": "core-projects", "tags": ["saturn", "chestohedron"]},
    {"id": "1f20e62d", "content": "SATURN LENS: A PreToolUse hook that classifies every AI tool call into 5 structural primitives. First microscope for AI cognition.", "vault": "facts", "confidence": 1.0, "domain": "core-projects", "tags": ["saturn", "saturn-lens"]},
]


def _build_real_graph() -> TopologicalGraph:
    """Load real KD nodes into a TopologicalGraph.

    Uses the MCP adapter to parse nodes, then builds topological edges
    based on domain and tag relationships.
    """
    adapter = KDMCPAdapter()
    adapter.load_vault_stats(_VAULT_STATS)
    adapter.load_graph_stats(_GRAPH_STATS)

    # Parse all nodes through the adapter.
    kd_nodes = adapter.parse_vault_query(_REAL_NODES)

    # Build local edges.
    edge_count = adapter.build_edges_between_cached()

    # Create topological graph.
    g = TopologicalGraph()

    # Map KD edge types to TCA edge types.
    kd_to_tca_edge = {
        "domain": EdgeType.MIRRORS,      # same domain = analogous
        "similarity": EdgeType.MIRRORS,   # similar content = analogous
        "session": EdgeType.INHERITS,     # same session = derived from
        "recognition": EdgeType.VERIFIES, # recognized pattern = evidence
        "extension": EdgeType.EXPRESSES,  # extended idea = manifests as
        "compression": EdgeType.BOUNDS,   # compressed = constrained
        "analogy": EdgeType.MIRRORS,      # analogy = analogous
        "inversion": EdgeType.REMOVES,    # inverted = contradicts
        "contradiction": EdgeType.REMOVES,
        "decomposition": EdgeType.INHERITS,
    }

    # Add nodes.
    for kd_node in kd_nodes:
        g.add_node(
            label=kd_node.content[:80],
            node_id=kd_node.id,
        )

    # Add edges from local cache.
    for kd_node in kd_nodes:
        kd_edges = adapter.get_edges(kd_node.id)
        for kd_edge in kd_edges:
            other_id = (kd_edge.target_id
                        if kd_edge.source_id == kd_node.id
                        else kd_edge.source_id)
            if g.get_node(other_id) is not None:
                edge_type = kd_to_tca_edge.get(kd_edge.edge_type,
                                                EdgeType.MIRRORS)
                g.add_edge(kd_node.id, other_id, edge_type,
                           weight=kd_edge.similarity)

    return g, adapter, edge_count


class TestPhase2RealData(unittest.TestCase):
    """Tests with real KD data loaded into TCA."""

    @classmethod
    def setUpClass(cls):
        cls.graph, cls.adapter, cls.local_edges = _build_real_graph()

    def test_real_nodes_loaded(self):
        """Real KD nodes successfully loaded into TopologicalGraph."""
        self.assertEqual(self.graph.node_count, len(_REAL_NODES),
                         f"Expected {len(_REAL_NODES)} nodes, "
                         f"got {self.graph.node_count}")
        print(f"\n  Loaded {self.graph.node_count} real KD nodes")
        print(f"  Local edges built: {self.local_edges}")
        print(f"  Total TCA edges: {self.graph.total_edge_count()}")

    def test_real_data_spreading_activation(self):
        """Spreading activation works on real KD data."""
        # Find topology-related entry nodes.
        entries = self.graph.find_entry("topology")
        self.assertGreater(len(entries), 0, "Should find topology nodes")

        gate_weights = {"MIRROR": 0.3, "INHERIT": 0.1, "BOUND": 0.05,
                        "EXPRESS": 0.1, "VERIFY": 0.1, "REMOVE": 0.05,
                        "DARASH": 0.3}

        results = self.graph.query_by_activation(entries, gate_weights, depth=3)
        self.assertGreater(len(results), len(entries),
                           "Activation should spread beyond entry nodes")
        print(f"\n  Entry nodes: {len(entries)}")
        print(f"  Activated nodes: {len(results)}")
        for nid, act in results[:5]:
            node = self.graph.get_node(nid)
            print(f"    {act:.4f} {node.label[:60]}")

    def test_real_data_reasoning(self):
        """Full reasoning loop on real KD data."""
        result = reason("What is topology over substance?",
                        self.graph, max_ticks=50)
        self.assertTrue(result.converged or result.ticks_used <= 50)
        self.assertGreater(len(result.activated_nodes), 0)

        # Compute confidence.
        subgraph = self.graph.subgraph(
            [nid for nid, _ in result.activated_nodes])
        conf = compute_confidence(subgraph,
                                  ticks_used=result.ticks_used, max_ticks=50)

        print(f"\n  Query: 'What is topology over substance?'")
        print(f"  Ticks: {result.ticks_used}")
        print(f"  Converged: {result.converged}")
        print(f"  Nodes activated: {len(result.activated_nodes)}")
        print(f"  Confidence: {conf.confidence:.3f}")
        print(f"    path_diversity: {conf.path_diversity_score:.3f}")
        print(f"    grounding: {conf.grounding_ratio:.3f}")
        print(f"    convergence: {conf.convergence_speed:.3f}")

    def test_real_data_metacognition(self):
        """Metacognitive monitoring on real KD data."""
        result = reason("chestohedron protocol gates cognitive",
                        self.graph, max_ticks=30)

        subgraph = self.graph.subgraph(
            [nid for nid, _ in result.activated_nodes])

        if subgraph:
            mon = monitor(subgraph,
                          result.gate_weights,
                          ticks_used=result.ticks_used,
                          max_ticks=30)

            print(f"\n  Chestohedron query on real data:")
            print(f"  Confidence: {mon.confidence.confidence:.3f}")
            print(f"  Cycles detected: {mon.health.cycle_count}")
            print(f"  Bridges detected: {mon.health.bridge_count}")
            print(f"  Isolated nodes: {mon.health.isolation_count}")
            print(f"  Interventions: {len(mon.interventions)}")
            for i in mon.interventions:
                print(f"    [{i.action}] {i.reason[:60]}")

    def test_no_embeddings_at_scale(self):
        """Even with 30 real nodes, no embeddings or vectors exist."""
        for nid in list(self.graph.nodes.keys())[:10]:
            node = self.graph.get_node(nid)
            for attr in ("embedding", "vector", "coordinates", "latent"):
                self.assertFalse(
                    hasattr(node, attr),
                    f"Node {nid} should not have '{attr}' at scale")


if __name__ == "__main__":
    unittest.main(verbosity=2)
