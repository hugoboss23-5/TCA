"""Tests for TCA Layer 3: Grounding Interface."""

import unittest

from tca.L2_graph.operations import TopologicalGraph
from tca.L2_graph.topo_node import EdgeType
from tca.L3_grounding.prediction_loop import GroundingLoop


class TestL3Grounding(unittest.TestCase):

    def _make_graph_and_loop(self):
        g = TopologicalGraph()
        g.add_node(label="predictor", node_id="p")
        g.add_node(label="target", node_id="t")
        g.add_edge("p", "t", EdgeType.VERIFIES, weight=1.0)
        return g, GroundingLoop(g)

    def test_correct_prediction_strengthens_edge(self):
        """Predict file exists, file does exist -> edge weight increases."""
        g, loop = self._make_graph_and_loop()
        node = g.get_node("p")
        original_weight = node.edges["t"][0].weight

        # Close prediction: pred=0.85, actual=0.9 -> delta=-0.05
        # abs(delta) < 0.2 -> strengthen
        result = loop.ground("p", "t", "file_exists",
                             prediction=0.85, actual=0.9)
        new_weight = node.edges["t"][0].weight
        self.assertGreater(new_weight, original_weight,
                           f"Edge should strengthen: {original_weight} -> {new_weight}")

    def test_wrong_prediction_weakens_edge(self):
        """Predict file exists, file doesn't exist -> edge weight decreases."""
        g, loop = self._make_graph_and_loop()
        node = g.get_node("p")
        original_weight = node.edges["t"][0].weight

        # Bad prediction: pred=0.9, actual=0.0 -> delta=0.9
        result = loop.ground("p", "t", "file_exists",
                             prediction=0.9, actual=0.0)
        new_weight = node.edges["t"][0].weight
        self.assertLess(new_weight, original_weight,
                        f"Edge should weaken: {original_weight} -> {new_weight}")

    def test_weight_saturation(self):
        """After 10 correct predictions, edge weight saturates (bounded)."""
        g, loop = self._make_graph_and_loop()
        node = g.get_node("p")

        for _ in range(50):  # Way more than 10 to ensure saturation.
            loop.ground("p", "t", "file_exists",
                        prediction=0.85, actual=0.9)

        weight = node.edges["t"][0].weight
        self.assertLessEqual(weight, 5.0,
                             f"Weight should saturate at max, got {weight}")
        self.assertGreater(weight, 1.0,
                           "Weight should have increased from 1.0")

    def test_learning_changes_prediction(self):
        """After wrong prediction, next prediction on same relationship differs."""
        g, loop = self._make_graph_and_loop()

        # First prediction.
        pred1 = loop.predict("p", "file_exists")

        # Wrong prediction recorded.
        loop.ground("p", "t", "file_exists", prediction=0.9, actual=0.0)

        # Second prediction should differ (learned from error).
        pred2 = loop.predict("p", "file_exists")
        self.assertNotAlmostEqual(pred1, pred2, places=2,
                                  msg=f"Prediction should change after error: "
                                  f"{pred1:.3f} vs {pred2:.3f}")

    def test_delta_computed_correctly(self):
        """Delta is computed correctly for each action type."""
        g, loop = self._make_graph_and_loop()

        test_cases = [
            (0.8, 1.0, -0.2),
            (0.9, 0.0, 0.9),
            (0.5, 0.5, 0.0),
            (0.3, 0.7, -0.4),
            (1.0, 0.0, 1.0),
        ]
        for pred, actual, expected_delta in test_cases:
            result = loop.ground("p", "t", "delta_test",
                                 prediction=pred, actual=actual)
            self.assertAlmostEqual(
                result.delta, expected_delta, places=6,
                msg=f"Delta for pred={pred}, actual={actual}: "
                f"expected {expected_delta}, got {result.delta}"
            )


if __name__ == "__main__":
    unittest.main()
