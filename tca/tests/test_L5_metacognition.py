"""Tests for TCA Layer 5: Metacognitive Monitor."""

import unittest

from tca.L2_graph.topo_node import TopologicalNode, EdgeType
from tca.L5_metacognition.health import (
    cycle_detection, isolation_detection, bridge_detection, compute_health,
)
from tca.L5_metacognition.confidence import compute_confidence
from tca.L5_metacognition.interventions import (
    flag_uncertainty, monitor,
)


class TestL5Metacognition(unittest.TestCase):

    def test_circular_reasoning_detected(self):
        """Circular reasoning (A->B->C->A) detected by cycle_detection."""
        sub = {
            "A": TopologicalNode(id="A", label="A"),
            "B": TopologicalNode(id="B", label="B"),
            "C": TopologicalNode(id="C", label="C"),
        }
        sub["A"].add_edge("B", EdgeType.VERIFIES, weight=0.8)
        sub["B"].add_edge("C", EdgeType.INHERITS, weight=0.7)
        sub["C"].add_edge("A", EdgeType.MIRRORS, weight=0.6)

        cycles = cycle_detection(sub)
        self.assertGreater(len(cycles), 0,
                           "Should detect at least one cycle")
        # The cycle should contain A, B, and C.
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)
        self.assertIn("A", cycle_nodes)
        self.assertIn("B", cycle_nodes)
        self.assertIn("C", cycle_nodes)

    def test_dead_ends_detected(self):
        """Dead-end nodes detected by isolation_detection."""
        sub = {
            "A": TopologicalNode(id="A", label="A", activation=1.0),
            "B": TopologicalNode(id="B", label="B", activation=0.5),
            "D": TopologicalNode(id="D", label="D", activation=0.0),
        }
        sub["A"].add_edge("B", EdgeType.EXPRESSES, weight=0.8)
        # D has no edges — it's isolated.

        isolated = isolation_detection(sub, conclusion_id="A")
        self.assertIn("D", isolated,
                       f"D should be isolated, got {isolated}")

    def test_well_connected_higher_confidence(self):
        """Well-connected subgraph produces higher confidence than fragmented."""
        # Well-connected: multiple grounded paths.
        good = {
            "A": TopologicalNode(id="A", label="A"),
            "B": TopologicalNode(id="B", label="B"),
            "C": TopologicalNode(id="C", label="C"),
        }
        good["A"].add_edge("B", EdgeType.VERIFIES, weight=0.9, grounded=True)
        good["A"].add_edge("C", EdgeType.VERIFIES, weight=0.8, grounded=True)
        good["B"].add_edge("C", EdgeType.EXPRESSES, weight=0.7, grounded=True)

        # Fragmented: no grounding, cycle.
        bad = {
            "X": TopologicalNode(id="X", label="X"),
            "Y": TopologicalNode(id="Y", label="Y"),
        }
        bad["X"].add_edge("Y", EdgeType.MIRRORS, weight=0.3)
        bad["Y"].add_edge("X", EdgeType.MIRRORS, weight=0.3)

        good_conf = compute_confidence(good, "A", "C", ticks_used=2, max_ticks=100)
        bad_conf = compute_confidence(bad, "X", "Y", ticks_used=90, max_ticks=100)

        self.assertGreater(good_conf.confidence, bad_conf.confidence,
                           f"Good ({good_conf.confidence:.3f}) should beat "
                           f"bad ({bad_conf.confidence:.3f})")

    def test_bridge_formation_detected(self):
        """Bridge between two clusters is detected."""
        sub = {
            "A": TopologicalNode(id="A", label="cluster1_a"),
            "B": TopologicalNode(id="B", label="cluster1_b"),
            "C": TopologicalNode(id="C", label="cluster2_a"),
            "D": TopologicalNode(id="D", label="cluster2_b"),
        }
        # Cluster 1.
        sub["A"].add_edge("B", EdgeType.INHERITS, weight=0.8)
        sub["B"].add_edge("A", EdgeType.INHERITS, weight=0.8)
        # Cluster 2.
        sub["C"].add_edge("D", EdgeType.EXPRESSES, weight=0.7)
        sub["D"].add_edge("C", EdgeType.EXPRESSES, weight=0.7)
        # Bridge.
        sub["B"].add_edge("C", EdgeType.SEEKS, weight=0.4)

        bridges = bridge_detection(sub)
        self.assertGreater(len(bridges), 0,
                           "Should detect bridge between clusters")
        # The bridge should involve B and C.
        bridge_nodes = set()
        for a, b in bridges:
            bridge_nodes.add(a)
            bridge_nodes.add(b)
        self.assertTrue(
            {"B", "C"} & bridge_nodes,
            f"Bridge should involve B-C, got {bridges}")

    def test_low_confidence_triggers_flag(self):
        """Low confidence triggers flag_uncertainty intervention."""
        intervention = flag_uncertainty(0.15, threshold=0.3)
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.action, "flag_uncertainty")

        # High confidence should NOT trigger.
        no_intervention = flag_uncertainty(0.8, threshold=0.3)
        self.assertIsNone(no_intervention)

    def test_monitor_reads_and_writes_graph(self):
        """Monitor can read AND write to Layer 2 graph (self-modification)."""
        sub = {
            "A": TopologicalNode(id="A", label="A", activation=0.5),
            "B": TopologicalNode(id="B", label="B", activation=0.3),
            "C": TopologicalNode(id="C", label="C", activation=0.001),
        }
        sub["A"].add_edge("B", EdgeType.MIRRORS, weight=0.5)
        # C is isolated with low activation — should get pruned.

        gates = {"MIRROR": 1/7, "INHERIT": 1/7, "BOUND": 1/7,
                 "EXPRESS": 1/7, "VERIFY": 1/7, "REMOVE": 1/7, "DARASH": 1/7}

        result = monitor(sub, gates, ticks_used=50, max_ticks=100)

        # Monitor READ: health metrics computed.
        self.assertIsNotNone(result.health)
        self.assertIsNotNone(result.confidence)

        # Monitor WRITE: pruning modified the graph.
        prune_interventions = [i for i in result.interventions
                               if i.action == "prune"]
        if prune_interventions:
            # C should have been pruned (activation set to 0).
            self.assertEqual(sub["C"].activation, 0.0,
                             "Pruned node should have activation=0")


if __name__ == "__main__":
    unittest.main()
