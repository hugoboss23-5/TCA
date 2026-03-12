"""Tests for TCA Layer 5: Confidence V2 (topology + coherence)."""

import unittest

from tca.L2_graph.topo_node import EdgeType
from tca.L2_graph.operations import TopologicalGraph
from tca.L2_graph.activation import spread
from tca.L4_temporal.temporal_node import record_activation
from tca.L4_temporal.coherence import CoherenceDetector
from tca.L5_metacognition.confidence import ConfidenceReport
from tca.L5_metacognition.confidence_v2 import (
    compute_confidence_v2, ConfidenceV2Report,
)


def _make_health(confidence: float) -> ConfidenceReport:
    """Create a ConfidenceReport with a specific confidence score."""
    return ConfidenceReport(
        confidence=confidence,
        path_diversity_score=confidence,
        cycle_penalty=0.0,
        grounding_ratio=confidence,
        convergence_speed=confidence,
    )


class TestConfidenceV2(unittest.TestCase):

    def test_premature_state_low_confidence(self):
        """Premature state: high static, low coherence → LOW confidence.

        This is the test that proves TCA catches hallucination-equivalent
        failures — static topology looks good, but dynamic coherence is
        absent, meaning the system converged without real understanding.
        """
        g = TopologicalGraph()
        g.add_node(label="a", node_id="A")
        g.add_node(label="b", node_id="B")
        # No meaningful activation history → low coherence.

        detector = CoherenceDetector(g)
        health = _make_health(0.8)  # High static confidence.

        result = compute_confidence_v2(health, detector, ["A", "B"])

        self.assertEqual(result.state, "premature")
        self.assertLess(result.confidence, 0.5,
                        f"Premature state should have low confidence, "
                        f"got {result.confidence:.3f}")
        self.assertIn("premature_convergence", result.flags)

    def test_understanding_state_high_confidence(self):
        """Understanding state: high static + high coherence → HIGH confidence."""
        g = TopologicalGraph()
        g.add_node(label="a", node_id="A")
        g.add_node(label="b", node_id="B")
        # No direct edge → emergent sync not discounted.

        # Build correlated activation histories manually.
        node_a = g.get_node("A")
        node_b = g.get_node("B")
        for i in range(15):
            val = 0.5 + 0.3 * (i % 3) / 2.0
            node_a.activation = val
            node_b.activation = val + 0.01  # Nearly identical rhythm.
            record_activation(node_a)
            record_activation(node_b)

        detector = CoherenceDetector(g)
        health = _make_health(0.8)

        result = compute_confidence_v2(health, detector, ["A", "B"],
                                       tick_window=10)

        self.assertEqual(result.state, "understanding")
        self.assertGreater(result.confidence, 0.5,
                           f"Understanding state should have high confidence, "
                           f"got {result.confidence:.3f}")
        self.assertEqual(result.flags, [])

    def test_forming_state_flags_continue(self):
        """Forming state: high coherence + low static → continue flag."""
        g = TopologicalGraph()
        g.add_node(label="a", node_id="A")
        g.add_node(label="b", node_id="B")

        node_a = g.get_node("A")
        node_b = g.get_node("B")
        for i in range(15):
            val = 0.5 + 0.2 * (i % 3) / 2.0
            node_a.activation = val
            node_b.activation = val + 0.01
            record_activation(node_a)
            record_activation(node_b)

        detector = CoherenceDetector(g)
        health = _make_health(0.3)  # Low static.

        result = compute_confidence_v2(health, detector, ["A", "B"],
                                       tick_window=10)

        self.assertEqual(result.state, "forming")
        self.assertIn("continue_reasoning", result.flags)

    def test_confused_state_zero_confidence(self):
        """Confused state: low coherence + low static → 0.0 confidence."""
        g = TopologicalGraph()
        g.add_node(label="a", node_id="A")
        g.add_node(label="b", node_id="B")

        detector = CoherenceDetector(g)
        health = _make_health(0.2)  # Low static.

        result = compute_confidence_v2(health, detector, ["A", "B"])

        self.assertEqual(result.state, "confused")
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("reroute", result.flags)

    def test_report_contains_all_fields(self):
        """ConfidenceV2Report contains static and dynamic scores."""
        g = TopologicalGraph()
        g.add_node(label="a", node_id="A")
        detector = CoherenceDetector(g)
        health = _make_health(0.6)

        result = compute_confidence_v2(health, detector, ["A"])

        self.assertIsInstance(result, ConfidenceV2Report)
        self.assertIsInstance(result.static_confidence, float)
        self.assertIsInstance(result.dynamic_coherence, float)
        self.assertIn(result.state,
                      ("understanding", "premature", "forming", "confused"))


if __name__ == "__main__":
    unittest.main()
