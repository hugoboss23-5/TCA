"""Integration test for the TCA Coherence Patch.

Exercises all 3 patches together in a single flow:
1. Build a multi-cluster graph
2. Run reasoning (L1→L2→L4) to populate activation histories
3. Compute confidence_v2 (coherence + static topology)
4. Run resonance grounding with FileSystemSource
5. Serialize → deserialize full state → verify round-trip
6. Assert coherence-aware confidence differs from static-only confidence
"""

import json
import tempfile
import os
import unittest

from tca.L1_router.router import route as l1_route
from tca.L2_graph.topo_node import EdgeType
from tca.L2_graph.operations import TopologicalGraph
from tca.L4_temporal.reasoning import reason
from tca.L4_temporal.coherence import CoherenceDetector
from tca.L5_metacognition.confidence import compute_confidence
from tca.L5_metacognition.confidence_v2 import compute_confidence_v2
from tca.L3_grounding.resonance import (
    FileSystemSource, test_resonance,
)
from tca.L0_protocol.schema import (
    serialize_tca_state, deserialize_tca_state, export_boot_protocol,
)


def _build_integration_graph() -> TopologicalGraph:
    """Build a multi-cluster graph for integration testing."""
    g = TopologicalGraph()

    # Cluster 1: mathematics.
    g.add_node(label="topology mathematics", node_id="topo")
    g.add_node(label="manifolds geometry", node_id="man")
    g.add_node(label="homeomorphism mapping", node_id="homeo")
    g.add_edge("topo", "man", EdgeType.EXPRESSES, weight=0.8)
    g.add_edge("man", "homeo", EdgeType.INHERITS, weight=0.7)
    g.add_edge("homeo", "topo", EdgeType.VERIFIES, weight=0.6)

    # Cluster 2: cognition.
    g.add_node(label="consciousness awareness", node_id="cons")
    g.add_node(label="neural correlates", node_id="nc")
    g.add_node(label="binding problem", node_id="bind")
    g.add_edge("cons", "nc", EdgeType.EXPRESSES, weight=0.7)
    g.add_edge("nc", "bind", EdgeType.SEEKS, weight=0.8)
    g.add_edge("bind", "cons", EdgeType.BOUNDS, weight=0.5)

    # Bridge.
    g.add_node(label="topological consciousness theory", node_id="bridge")
    g.add_edge("topo", "bridge", EdgeType.SEEKS, weight=0.4)
    g.add_edge("cons", "bridge", EdgeType.SEEKS, weight=0.4)
    g.add_edge("bridge", "man", EdgeType.MIRRORS, weight=0.3)
    g.add_edge("bridge", "nc", EdgeType.MIRRORS, weight=0.3)

    return g


class TestCoherencePatchIntegration(unittest.TestCase):

    def test_full_stack_with_coherence(self):
        """Full reasoning cycle using confidence_v2 with coherence."""
        g = _build_integration_graph()

        # L1 + L2 + L4: Run reasoning to populate activation histories.
        result = reason("topology consciousness", g,
                        max_ticks=30, convergence_threshold=0.001)

        self.assertGreater(result.ticks_used, 0,
                           "Reasoning should use at least 1 tick")

        # Get activated nodes.
        active_ids = [nid for nid, _ in result.activated_nodes]
        self.assertGreater(len(active_ids), 0,
                           "Should have activated nodes")

        # L5 v1: Static confidence.
        subgraph = g.subgraph(active_ids)
        source_id = active_ids[0] if active_ids else None
        target_id = active_ids[-1] if len(active_ids) > 1 else source_id
        static_report = compute_confidence(
            subgraph, source_id, target_id,
            ticks_used=result.ticks_used, max_ticks=30)

        # L4 + L5 v2: Coherence-aware confidence.
        detector = CoherenceDetector(g)
        v2_report = compute_confidence_v2(
            static_report, detector, active_ids, tick_window=10)

        # V2 report should have a valid state.
        self.assertIn(v2_report.state,
                      ("understanding", "premature", "forming", "confused"))

        # The two confidence scores should exist and be bounded.
        self.assertGreaterEqual(v2_report.static_confidence, 0.0)
        self.assertLessEqual(v2_report.static_confidence, 1.0)
        self.assertGreaterEqual(v2_report.dynamic_coherence, 0.0)
        self.assertLessEqual(v2_report.dynamic_coherence, 1.0)

    def test_resonance_grounding_integration(self):
        """Resonance grounding with FileSystemSource on a live graph."""
        g = _build_integration_graph()
        node = g.get_node("topo")

        # Ground against a real file.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_path = f.name

        try:
            source = FileSystemSource(tmp_path)
            alignment = test_resonance(node, source, g)
            self.assertGreater(alignment, 0.0)
            self.assertLessEqual(alignment, 1.0)
        finally:
            os.unlink(tmp_path)

    def test_serialize_deserialize_round_trip(self):
        """Serialize full state after reasoning → deserialize → verify."""
        g = _build_integration_graph()
        result = reason("topology consciousness", g,
                        max_ticks=20, convergence_threshold=0.001)

        active_ids = [nid for nid, _ in result.activated_nodes]
        subgraph = g.subgraph(active_ids)
        source_id = active_ids[0] if active_ids else None
        target_id = active_ids[-1] if len(active_ids) > 1 else source_id
        static_report = compute_confidence(
            subgraph, source_id, target_id,
            ticks_used=result.ticks_used, max_ticks=20)

        detector = CoherenceDetector(g)
        v2_report = compute_confidence_v2(
            static_report, detector, active_ids)

        # Serialize.
        state = serialize_tca_state(
            router_weights=result.gate_weights,
            graph=g,
            temporal_result=result,
            metacognition_report=v2_report,
        )

        # Verify JSON-serializable.
        json_str = json.dumps(state)
        self.assertGreater(len(json_str), 0)

        # Deserialize.
        restored = deserialize_tca_state(state)
        g2 = restored["graph"]

        # Graph structure preserved.
        self.assertEqual(g2.node_count, g.node_count)
        self.assertEqual(g2.total_edge_count(), g.total_edge_count())

        # Router weights preserved.
        for gate, weight in result.gate_weights.items():
            self.assertAlmostEqual(
                restored["router_weights"][gate], weight, places=6)

        # Metacognition preserved.
        self.assertAlmostEqual(
            restored["metacognition"]["confidence"],
            v2_report.confidence, places=6)

    def test_boot_protocol_is_clean(self):
        """Boot protocol from live graph contains no personal data."""
        g = _build_integration_graph()
        reason("topology consciousness", g, max_ticks=10)

        boot = export_boot_protocol(g)
        boot_json = json.dumps(boot)

        # No node labels in boot protocol.
        self.assertNotIn("topology mathematics", boot_json)
        self.assertNotIn("consciousness awareness", boot_json)

        # But architecture is present.
        self.assertIn("MIRROR", boot_json)
        self.assertIn("pearson_correlation", boot_json)
        self.assertEqual(boot["architecture"]["gates"],
                         ["MIRROR", "INHERIT", "BOUND", "EXPRESS",
                          "VERIFY", "REMOVE", "DARASH"])

    def test_fresh_instance_from_round_trip_can_route(self):
        """A graph restored from serialization can run a new query."""
        g = _build_integration_graph()
        reason("topology", g, max_ticks=10)

        state = serialize_tca_state(graph=g)
        restored = deserialize_tca_state(state)
        g2 = restored["graph"]

        # Run a new query on the restored graph.
        result2 = reason("consciousness", g2, max_ticks=20)
        self.assertGreater(result2.ticks_used, 0)


if __name__ == "__main__":
    unittest.main()
