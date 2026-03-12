"""Tests for TCA Layer 3: Resonance Grounding Interface."""

import unittest
import tempfile
import os

from tca.L2_graph.topo_node import EdgeType
from tca.L2_graph.operations import TopologicalGraph
from tca.L3_grounding.resonance import (
    SignalSource, FileSystemSource, CodeExecutionSource,
    measure_resonance, derive_prediction_from_topology,
    compute_alignment, update_graph_from_resonance,
)


class TestFileSystemSource(unittest.TestCase):

    def test_correct_claim_high_alignment(self):
        """FileSystemSource with existing file → high alignment."""
        g = TopologicalGraph()
        g.add_node(label="file_claim", node_id="S")
        g.add_node(label="target", node_id="T")
        # VERIFIES edge = topology predicts positive (file exists).
        g.add_edge("S", "T", EdgeType.VERIFIES, weight=2.0)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_path = f.name

        try:
            source = FileSystemSource(tmp_path)
            node = g.get_node("S")
            alignment = measure_resonance(node, source, g)
            self.assertGreater(alignment, 0.5,
                               f"Correct claim should give high alignment, "
                               f"got {alignment:.3f}")
        finally:
            os.unlink(tmp_path)

    def test_incorrect_claim_low_alignment(self):
        """FileSystemSource with nonexistent file → low alignment."""
        g = TopologicalGraph()
        g.add_node(label="file_claim", node_id="S")
        g.add_node(label="target", node_id="T")
        # VERIFIES edge = topology predicts file exists.
        g.add_edge("S", "T", EdgeType.VERIFIES, weight=2.0)

        source = FileSystemSource("/nonexistent/path/xyz123.txt")
        node = g.get_node("S")
        alignment = measure_resonance(node, source, g)
        self.assertLess(alignment, 0.5,
                        f"Incorrect claim should give low alignment, "
                        f"got {alignment:.3f}")

    def test_edge_weight_decreases_after_low_alignment(self):
        """Edge weight decreases after low resonance alignment."""
        g = TopologicalGraph()
        g.add_node(label="claim", node_id="S")
        g.add_node(label="target", node_id="T")
        g.add_edge("S", "T", EdgeType.VERIFIES, weight=2.0)

        node = g.get_node("S")
        weight_before = node.edges["T"][0].weight

        source = FileSystemSource("/nonexistent/path/abc.txt")
        measure_resonance(node, source, g)

        weight_after = node.edges["T"][0].weight
        self.assertLess(weight_after, weight_before,
                        f"Edge should weaken after wrong prediction: "
                        f"{weight_before:.3f} -> {weight_after:.3f}")

    def test_edge_weight_increases_after_high_alignment(self):
        """Edge weight increases after high resonance alignment."""
        g = TopologicalGraph()
        g.add_node(label="claim", node_id="S")
        g.add_node(label="target", node_id="T")
        g.add_edge("S", "T", EdgeType.VERIFIES, weight=1.0)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_path = f.name

        try:
            node = g.get_node("S")
            weight_before = node.edges["T"][0].weight

            source = FileSystemSource(tmp_path)
            measure_resonance(node, source, g)

            weight_after = node.edges["T"][0].weight
            self.assertGreater(weight_after, weight_before,
                               f"Edge should strengthen after correct prediction: "
                               f"{weight_before:.3f} -> {weight_after:.3f}")
        finally:
            os.unlink(tmp_path)


class TestCustomSignalSource(unittest.TestCase):

    def test_custom_source_works_without_modifying_tca(self):
        """A new SignalSource subclass works with measure_resonance unchanged."""

        class TemperatureSource(SignalSource):
            """Hypothetical hardware source — proves interface is open."""
            def __init__(self, temp: float):
                self.temp = temp

            def measure(self) -> dict:
                return {"positive": self.temp > 20.0,
                        "source_type": "temperature"}

        g = TopologicalGraph()
        g.add_node(label="hypothesis", node_id="H")
        g.add_node(label="evidence", node_id="E")
        g.add_edge("H", "E", EdgeType.VERIFIES, weight=1.0)

        node = g.get_node("H")

        # Warm temperature → "warm" is truthy → aligns with VERIFIES.
        warm_source = TemperatureSource(25.0)
        alignment = measure_resonance(node, warm_source, g)
        self.assertGreater(alignment, 0.5)

        # Cold temperature → "warm" is False → misaligns with VERIFIES.
        cold_source = TemperatureSource(10.0)
        alignment = measure_resonance(node, cold_source, g)
        self.assertLess(alignment, 0.5)


class TestCodeExecutionSource(unittest.TestCase):

    def test_successful_code(self):
        """CodeExecutionSource with valid code → positive signal."""
        source = CodeExecutionSource("x = 1 + 1")
        signal = source.measure()
        self.assertTrue(signal["positive"])
        self.assertIsNone(signal["error"])

    def test_failing_code(self):
        """CodeExecutionSource with invalid code → negative signal."""
        source = CodeExecutionSource("raise ValueError('oops')")
        signal = source.measure()
        self.assertFalse(signal["positive"])
        self.assertIn("oops", signal["error"])


class TestPredictionDerivation(unittest.TestCase):

    def test_verifies_edges_predict_positive(self):
        """VERIFIES edges → predict positive outcome."""
        g = TopologicalGraph()
        g.add_node(label="x", node_id="X")
        g.add_node(label="y", node_id="Y")
        g.add_edge("X", "Y", EdgeType.VERIFIES, weight=3.0)

        pred = derive_prediction_from_topology(g.get_node("X"), g)
        self.assertTrue(pred["predicted_positive"])

    def test_removes_edges_predict_negative(self):
        """REMOVES edges → predict negative outcome."""
        g = TopologicalGraph()
        g.add_node(label="x", node_id="X")
        g.add_node(label="y", node_id="Y")
        g.add_edge("X", "Y", EdgeType.REMOVES, weight=3.0)

        pred = derive_prediction_from_topology(g.get_node("X"), g)
        self.assertFalse(pred["predicted_positive"])


if __name__ == "__main__":
    unittest.main()
