"""Tests for TCA Layer 2: Topological Knowledge Graph."""

import unittest

from tca.L2_graph.topo_node import TopologicalNode, EdgeType
from tca.L2_graph.activation import spread
from tca.L2_graph.operations import TopologicalGraph
from tca.adapters.kd_adapter import KDLocalCache


# Uniform gate weights for tests that don't care about gate bias.
UNIFORM_GATES = {
    "MIRROR": 1 / 7, "INHERIT": 1 / 7, "BOUND": 1 / 7,
    "EXPRESS": 1 / 7, "VERIFY": 1 / 7, "REMOVE": 1 / 7, "DARASH": 1 / 7,
}


class TestL2Graph(unittest.TestCase):

    def test_nodes_exist_with_only_edges(self):
        """Create 10 nodes with only edges, no coordinates. Verify queryable."""
        g = TopologicalGraph()
        ids = []
        for i in range(10):
            n = g.add_node(label=f"concept_{i}", node_id=f"n{i}")
            ids.append(n.id)

        # Connect in a chain.
        for i in range(9):
            g.add_edge(f"n{i}", f"n{i+1}", EdgeType.MIRRORS, weight=0.8)

        self.assertEqual(g.node_count, 10)
        for nid in ids:
            node = g.get_node(nid)
            self.assertIsNotNone(node)
            # No coordinates, no embeddings — just ID, label, edges.
            self.assertNotHasAttr(node, "coordinates")
            self.assertNotHasAttr(node, "embedding")
            self.assertNotHasAttr(node, "vector")

    def assertNotHasAttr(self, obj, attr_name):
        self.assertFalse(hasattr(obj, attr_name),
                         f"Object should not have attribute '{attr_name}'")

    def test_spreading_activation_decreases_with_distance(self):
        """Activation from A reaches C through B. Decreases with distance."""
        g = TopologicalGraph()
        g.add_node(label="A", node_id="A")
        g.add_node(label="B", node_id="B")
        g.add_node(label="C", node_id="C")
        g.add_edge("A", "B", EdgeType.VERIFIES, weight=0.9)
        g.add_edge("B", "C", EdgeType.VERIFIES, weight=0.9)

        results = g.query_by_activation(["A"], UNIFORM_GATES, depth=3)
        activations = dict(results)

        self.assertIn("A", activations)
        self.assertIn("B", activations)
        self.assertIn("C", activations)
        self.assertGreater(activations["A"], activations["B"],
                           "Activation should decrease: A > B")
        self.assertGreater(activations["B"], activations["C"],
                           "Activation should decrease: B > C")

    def test_gate_weighted_activation(self):
        """VERIFY gate emphasizes VERIFIES edges over MIRRORS edges."""
        g = TopologicalGraph()
        g.add_node(label="source", node_id="src")
        g.add_node(label="verified_target", node_id="vt")
        g.add_node(label="mirrored_target", node_id="mt")
        g.add_edge("src", "vt", EdgeType.VERIFIES, weight=0.9)
        g.add_edge("src", "mt", EdgeType.MIRRORS, weight=0.9)

        # VERIFY gate high.
        verify_gates = dict(UNIFORM_GATES)
        verify_gates["VERIFY"] = 0.8
        verify_gates["MIRROR"] = 0.02
        # Re-normalize.
        total = sum(verify_gates.values())
        verify_gates = {k: v / total for k, v in verify_gates.items()}

        results = g.query_by_activation(["src"], verify_gates, depth=1)
        activations = dict(results)

        vt_act = activations.get("vt", 0)
        mt_act = activations.get("mt", 0)
        self.assertGreater(vt_act, mt_act,
                           f"VERIFIES target ({vt_act:.4f}) should have higher "
                           f"activation than MIRRORS target ({mt_act:.4f}) "
                           f"when VERIFY gate is high")

    def test_disconnected_clusters(self):
        """Two disconnected clusters. Activation doesn't cross."""
        g = TopologicalGraph()
        # Cluster 1.
        g.add_node(label="c1_a", node_id="c1a")
        g.add_node(label="c1_b", node_id="c1b")
        g.add_edge("c1a", "c1b", EdgeType.INHERITS, weight=0.9)

        # Cluster 2 (disconnected).
        g.add_node(label="c2_a", node_id="c2a")
        g.add_node(label="c2_b", node_id="c2b")
        g.add_edge("c2a", "c2b", EdgeType.INHERITS, weight=0.9)

        results = g.query_by_activation(["c1a"], UNIFORM_GATES, depth=5)
        activated_ids = {nid for nid, _ in results}

        self.assertIn("c1a", activated_ids)
        self.assertIn("c1b", activated_ids)
        self.assertNotIn("c2a", activated_ids,
                          "Cluster 2 should NOT be activated")
        self.assertNotIn("c2b", activated_ids,
                          "Cluster 2 should NOT be activated")

    def test_merge_from_kd(self):
        """Import from KD adapter produces valid topological nodes."""
        g = TopologicalGraph()
        kd = KDLocalCache()

        # Deposit into KD.
        kid1 = kd.deposit("topology is shape without coordinates",
                          vault="facts", confidence=0.9, node_id="kd1")
        kid2 = kd.deposit("graphs represent connections",
                          vault="facts", confidence=0.8, node_id="kd2")
        kd.add_edge("kd1", "kd2", similarity=0.6)

        # Query and merge.
        kd_results = kd.query("topology")
        merged = g.merge_from_kd(kd_results, kd)

        self.assertTrue(len(merged) > 0, "Should have merged at least 1 node")
        for mid in merged:
            node = g.get_node(mid)
            self.assertIsNotNone(node, f"Merged node {mid} should exist")
            self.assertIsInstance(node, TopologicalNode)

    def test_pure_edge_traversal_no_embeddings(self):
        """CRITICAL: Query a concept by following ONLY edges.
        No similarity search. No embeddings. No vector math."""
        g = TopologicalGraph()

        # Build a knowledge structure:
        # "reservoir" --INHERITS--> "recurrent_networks"
        # "recurrent_networks" --EXPRESSES--> "temporal_processing"
        # "temporal_processing" --VERIFIES--> "prediction"
        g.add_node(label="reservoir computing", node_id="reservoir")
        g.add_node(label="recurrent neural networks", node_id="rnn")
        g.add_node(label="temporal processing", node_id="temporal")
        g.add_node(label="prediction", node_id="pred")

        g.add_edge("reservoir", "rnn", EdgeType.INHERITS, weight=0.9)
        g.add_edge("rnn", "temporal", EdgeType.EXPRESSES, weight=0.8)
        g.add_edge("temporal", "pred", EdgeType.VERIFIES, weight=0.7)

        # Query: start from "reservoir", find path to "prediction".
        results = g.query_by_activation(["reservoir"], UNIFORM_GATES, depth=4)
        activated_ids = {nid for nid, _ in results}

        # "prediction" should be reachable through pure edge traversal.
        self.assertIn("pred", activated_ids,
                       "Should reach 'prediction' through edge traversal alone")

        # Verify: no numpy, no vectors, no embeddings used.
        # The TopologicalNode has no embedding/vector/coordinate attributes.
        node = g.get_node("reservoir")
        for attr in ("embedding", "vector", "coordinates", "latent"):
            self.assertFalse(hasattr(node, attr),
                             f"Node should not have '{attr}' attribute")


if __name__ == "__main__":
    unittest.main()
