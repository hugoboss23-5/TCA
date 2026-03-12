"""Tests for TCA Layer 0: Cognitive Boot Protocol."""

import json
import unittest

from tca.L2_graph.topo_node import EdgeType
from tca.L2_graph.operations import TopologicalGraph
from tca.L4_temporal.temporal_node import record_activation
from tca.L0_protocol.schema import (
    TCA_PROTOCOL_VERSION,
    serialize_tca_state,
    deserialize_tca_state,
    export_boot_protocol,
)


def _build_test_graph(n_nodes: int = 6) -> TopologicalGraph:
    """Build a test graph with N nodes and edges."""
    g = TopologicalGraph()
    for i in range(n_nodes):
        g.add_node(label=f"node_{i}", node_id=f"n{i}")
    # Chain edges.
    for i in range(n_nodes - 1):
        g.add_edge(f"n{i}", f"n{i+1}", EdgeType.INHERITS,
                   weight=0.5 + i * 0.1, grounded=(i % 2 == 0))
    # Cross edge.
    if n_nodes > 2:
        g.add_edge(f"n{n_nodes-1}", "n0", EdgeType.VERIFIES, weight=0.7)
    return g


class TestSerializeDeserialize(unittest.TestCase):

    def test_round_trip_graph(self):
        """Serialize → deserialize preserves graph structure."""
        g = _build_test_graph(6)

        # Add activation history.
        for node in g.nodes.values():
            node.activation = 0.5
            record_activation(node)
            node.activation = 0.3
            record_activation(node)

        weights = {"MIRROR": 0.2, "INHERIT": 0.3, "BOUND": 0.1,
                   "EXPRESS": 0.1, "VERIFY": 0.15, "REMOVE": 0.1,
                   "DARASH": 0.05}

        state = serialize_tca_state(
            router_weights=weights,
            graph=g,
        )

        # Deserialize.
        result = deserialize_tca_state(state)
        g2 = result["graph"]

        # Same number of nodes.
        self.assertEqual(g2.node_count, g.node_count)

        # Same edge count.
        self.assertEqual(g2.total_edge_count(), g.total_edge_count())

        # Router weights preserved.
        self.assertEqual(result["router_weights"], weights)

        # Activation history preserved.
        for nid in g.nodes:
            orig = g.get_node(nid)
            restored = g2.get_node(nid)
            self.assertIsNotNone(restored, f"Node {nid} should exist")
            self.assertAlmostEqual(restored.activation, orig.activation,
                                   places=6)
            orig_hist = getattr(orig, "_activation_history", [])
            restored_hist = getattr(restored, "_activation_history", [])
            self.assertEqual(len(restored_hist), len(orig_hist))

    def test_round_trip_edge_properties(self):
        """Edge types, weights, and grounded flags survive round-trip."""
        g = _build_test_graph(4)
        state = serialize_tca_state(graph=g)
        result = deserialize_tca_state(state)
        g2 = result["graph"]

        # Check first edge.
        n0 = g2.get_node("n0")
        self.assertIn("n1", n0.edges)
        edge = n0.edges["n1"][0]
        self.assertEqual(edge.edge_type, EdgeType.INHERITS)
        self.assertTrue(edge.grounded)  # i=0, 0%2==0


class TestBootProtocol(unittest.TestCase):

    def test_no_node_content(self):
        """Boot protocol contains no node content (labels, activations)."""
        g = _build_test_graph(10)
        boot = export_boot_protocol(g)

        # Convert to JSON string for content inspection.
        boot_json = json.dumps(boot)

        # Should NOT contain any node labels.
        for i in range(10):
            self.assertNotIn(f"node_{i}", boot_json,
                             f"Boot protocol should not contain node label 'node_{i}'")

    def test_protocol_version_present(self):
        """Protocol version is present and parseable."""
        g = _build_test_graph(3)
        boot = export_boot_protocol(g)

        self.assertIn("protocol_version", boot)
        version = boot["protocol_version"]
        parts = version.split(".")
        self.assertEqual(len(parts), 3, "Version should be semver (X.Y.Z)")

    def test_contains_gate_definitions(self):
        """Boot protocol includes all 7 gate definitions."""
        boot = export_boot_protocol(TopologicalGraph())
        gates = boot["architecture"]["gates"]
        self.assertEqual(len(gates), 7)
        self.assertIn("MIRROR", gates)
        self.assertIn("DARASH", gates)

    def test_contains_edge_types(self):
        """Boot protocol includes all 7 edge types."""
        boot = export_boot_protocol(TopologicalGraph())
        edge_types = boot["architecture"]["edge_types"]
        self.assertEqual(len(edge_types), 7)
        self.assertIn("MIRRORS", edge_types)
        self.assertIn("SEEKS", edge_types)

    def test_contains_coherence_spec(self):
        """Boot protocol includes coherence metric specification."""
        boot = export_boot_protocol(TopologicalGraph())
        coherence = boot["specifications"]["coherence"]
        self.assertEqual(coherence["method"], "pearson_correlation")
        self.assertEqual(coherence["direct_edge_discount"], 0.3)

    def test_contains_resonance_spec(self):
        """Boot protocol includes resonance interface specification."""
        boot = export_boot_protocol(TopologicalGraph())
        resonance = boot["specifications"]["resonance"]
        self.assertIn("FileSystemSource", resonance["v01_sources"])


class TestProtocolSize(unittest.TestCase):

    def test_serialized_under_1mb_for_100_nodes(self):
        """Serialized state is < 1MB for a 100-node graph."""
        g = _build_test_graph(100)

        # Add activation history to all nodes.
        for _ in range(20):
            for node in g.nodes.values():
                node.activation = 0.5
                record_activation(node)

        state = serialize_tca_state(graph=g)
        json_str = json.dumps(state)
        size_bytes = len(json_str.encode("utf-8"))

        self.assertLess(size_bytes, 1_000_000,
                        f"Serialized state should be < 1MB, "
                        f"got {size_bytes:,} bytes")


class TestProtocolVersionConstant(unittest.TestCase):

    def test_version_constant(self):
        """TCA_PROTOCOL_VERSION is set correctly."""
        self.assertEqual(TCA_PROTOCOL_VERSION, "0.1.0")


if __name__ == "__main__":
    unittest.main()
