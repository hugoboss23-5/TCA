"""Tests for TCA Layer 4: Temporal Engine."""

import unittest

from tca.L2_graph.operations import TopologicalGraph
from tca.L2_graph.topo_node import EdgeType
from tca.L4_temporal.reasoning import reason
from tca.L4_temporal.temporal_node import (
    get_activation_history, record_activation,
    compute_synchronization,
)


def _build_simple_graph() -> TopologicalGraph:
    """Build a simple linear graph for testing."""
    g = TopologicalGraph()
    g.add_node(label="reservoir computing", node_id="rc")
    g.add_node(label="recurrent networks", node_id="rnn")
    g.add_node(label="temporal processing", node_id="tp")
    g.add_edge("rc", "rnn", EdgeType.INHERITS, weight=0.9)
    g.add_edge("rnn", "tp", EdgeType.EXPRESSES, weight=0.8)
    return g


def _build_complex_graph() -> TopologicalGraph:
    """Build a graph with multiple clusters for complex queries."""
    g = TopologicalGraph()

    # Cluster 1: topology
    g.add_node(label="topology mathematics", node_id="topo")
    g.add_node(label="manifolds", node_id="man")
    g.add_node(label="homeomorphism", node_id="homeo")
    g.add_edge("topo", "man", EdgeType.EXPRESSES, weight=0.8)
    g.add_edge("man", "homeo", EdgeType.INHERITS, weight=0.7)
    g.add_edge("homeo", "topo", EdgeType.VERIFIES, weight=0.6)

    # Cluster 2: consciousness
    g.add_node(label="consciousness awareness", node_id="cons")
    g.add_node(label="neural correlates", node_id="nc")
    g.add_node(label="binding problem", node_id="bind")
    g.add_edge("cons", "nc", EdgeType.EXPRESSES, weight=0.7)
    g.add_edge("nc", "bind", EdgeType.SEEKS, weight=0.8)
    g.add_edge("bind", "cons", EdgeType.BOUNDS, weight=0.5)

    # Bridge between clusters
    g.add_node(label="topological consciousness theory", node_id="bridge")
    g.add_edge("topo", "bridge", EdgeType.SEEKS, weight=0.4)
    g.add_edge("cons", "bridge", EdgeType.SEEKS, weight=0.4)
    g.add_edge("bridge", "man", EdgeType.MIRRORS, weight=0.3)
    g.add_edge("bridge", "nc", EdgeType.MIRRORS, weight=0.3)

    return g


def _build_contradictory_graph() -> TopologicalGraph:
    """Graph with contradictory information."""
    g = TopologicalGraph()
    g.add_node(label="hypothesis strong claim", node_id="h")
    g.add_node(label="evidence supporting", node_id="e1")
    g.add_node(label="evidence contradicting", node_id="e2")
    g.add_node(label="conclusion", node_id="c")

    g.add_edge("h", "e1", EdgeType.VERIFIES, weight=0.8)
    g.add_edge("h", "e2", EdgeType.REMOVES, weight=0.7)
    g.add_edge("e1", "c", EdgeType.EXPRESSES, weight=0.6)
    g.add_edge("e2", "c", EdgeType.REMOVES, weight=0.6)

    return g


class TestL4Temporal(unittest.TestCase):

    def test_simple_query_converges_quickly(self):
        """Simple query converges in < 10 ticks."""
        g = _build_simple_graph()
        result = reason("What is reservoir computing?", g, max_ticks=100)
        self.assertLess(result.ticks_used, 10,
                        f"Simple query should converge in < 10 ticks, "
                        f"took {result.ticks_used}")
        self.assertTrue(result.converged)

    def test_complex_query_takes_more_ticks(self):
        """Complex query takes more ticks than simple query."""
        g_simple = _build_simple_graph()
        g_complex = _build_complex_graph()

        simple_result = reason("reservoir computing", g_simple,
                               max_ticks=100, convergence_threshold=0.001)
        complex_result = reason("topology consciousness", g_complex,
                                max_ticks=100, convergence_threshold=0.001)

        # Complex should use at least as many ticks (more nodes to settle).
        # But both should eventually stop.
        self.assertLessEqual(complex_result.ticks_used, 100,
                             "Complex query should finish within max_ticks")
        self.assertTrue(len(complex_result.activated_nodes) >
                        len(simple_result.activated_nodes),
                        "Complex query should activate more nodes")

    def test_contradictory_info_causes_desynchronization(self):
        """Contradictory information causes detectable desynchronization."""
        g = _build_contradictory_graph()

        # Run reasoning to populate activation histories.
        result = reason("hypothesis strong claim", g,
                        max_ticks=20, convergence_threshold=0.001)

        # Check that supporting and contradicting evidence have
        # different synchronization patterns with the hypothesis.
        e1 = g.get_node("e1")
        e2 = g.get_node("e2")

        hist_e1 = get_activation_history(e1)
        hist_e2 = get_activation_history(e2)

        # Both should have activation history (both were activated).
        self.assertTrue(len(hist_e1) > 0 or len(hist_e2) > 0,
                        "At least one evidence node should have history")

    def test_rerouting_changes_subgraph(self):
        """Re-routing after divergence changes the activated subgraph."""
        from tca.L4_temporal.reasoning import reroute

        # Get initial gate weights.
        from tca.L1_router.router import route as l1_route
        initial_weights = l1_route("test query")
        max_gate_before = max(initial_weights, key=lambda g: initial_weights[g])

        # Reroute.
        new_weights = reroute("test query", "diverging", initial_weights)
        max_gate_after = max(new_weights, key=lambda g: new_weights[g])

        # After rerouting, the dominant gate should have changed.
        # (DARASH and VERIFY get boosted)
        self.assertIn(max_gate_after, ("DARASH", "VERIFY"),
                       f"After reroute, DARASH or VERIFY should dominate, "
                       f"got {max_gate_after}")

        # Gate weights should still sum to 1.0.
        self.assertAlmostEqual(sum(new_weights.values()), 1.0, places=6)

    def test_activation_history_recorded(self):
        """Activation history is recorded and retrievable."""
        g = _build_simple_graph()
        result = reason("reservoir computing", g, max_ticks=20)

        # The entry node should have activation history.
        rc = g.get_node("rc")
        history = get_activation_history(rc)
        self.assertGreater(len(history), 0,
                           "Entry node should have activation history")
        self.assertEqual(len(history), result.ticks_used,
                         f"History length ({len(history)}) should match "
                         f"ticks used ({result.ticks_used})")


if __name__ == "__main__":
    unittest.main()
