"""Tests for tca.analyze — structural analysis."""

from tca.graph import EdgeType, TopologicalGraph
from tca.analyze import (
    betweenness_centrality, clustering_coefficient, path_diversity,
    cycle_detection, bridge_detection, isolation_detection,
    compute_health, compute_confidence, compute_grounding_ratio,
)


def _make_star_graph():
    """A -> B, A -> C, A -> D (star topology, A is hub)."""
    g = TopologicalGraph()
    g.add_node(label="Hub", node_id="a")
    g.add_node(label="Spoke1", node_id="b")
    g.add_node(label="Spoke2", node_id="c")
    g.add_node(label="Spoke3", node_id="d")
    g.add_edge("a", "b", EdgeType.EXPRESSES)
    g.add_edge("a", "c", EdgeType.EXPRESSES)
    g.add_edge("a", "d", EdgeType.EXPRESSES)
    g.add_edge("b", "a", EdgeType.INHERITS)
    g.add_edge("c", "a", EdgeType.INHERITS)
    g.add_edge("d", "a", EdgeType.INHERITS)
    return g


def _make_cycle_graph():
    """A -> B -> C -> A (cycle)."""
    g = TopologicalGraph()
    g.add_node(label="A", node_id="a")
    g.add_node(label="B", node_id="b")
    g.add_node(label="C", node_id="c")
    g.add_edge("a", "b", EdgeType.EXPRESSES)
    g.add_edge("b", "c", EdgeType.EXPRESSES)
    g.add_edge("c", "a", EdgeType.EXPRESSES)
    return g


def test_betweenness_centrality_star():
    g = _make_star_graph()
    btwn = betweenness_centrality(g.nodes)
    assert btwn["a"] > btwn["b"]
    assert btwn["a"] > btwn["c"]


def test_clustering_coefficient():
    g = _make_star_graph()
    cc = clustering_coefficient(g.nodes)
    assert isinstance(cc, dict)
    assert all(0.0 <= v <= 1.0 for v in cc.values())


def test_cycle_detection_finds_cycle():
    g = _make_cycle_graph()
    cycles = cycle_detection(g.nodes)
    assert len(cycles) >= 1


def test_cycle_detection_no_cycle():
    g = TopologicalGraph()
    g.add_node(label="A", node_id="a")
    g.add_node(label="B", node_id="b")
    g.add_edge("a", "b", EdgeType.EXPRESSES)
    cycles = cycle_detection(g.nodes)
    assert len(cycles) == 0


def test_bridge_detection():
    g = TopologicalGraph()
    g.add_node(label="A", node_id="a")
    g.add_node(label="B", node_id="b")
    g.add_node(label="C", node_id="c")
    g.add_edge("a", "b", EdgeType.EXPRESSES)
    g.add_edge("b", "c", EdgeType.EXPRESSES)
    bridges = bridge_detection(g.nodes)
    assert len(bridges) >= 1


def test_path_diversity():
    g = TopologicalGraph()
    g.add_node(label="A", node_id="a")
    g.add_node(label="B", node_id="b")
    g.add_node(label="C", node_id="c")
    g.add_edge("a", "b", EdgeType.EXPRESSES)
    g.add_edge("a", "c", EdgeType.EXPRESSES)
    g.add_edge("b", "c", EdgeType.EXPRESSES)
    pd = path_diversity(g.nodes, "a", "c")
    assert pd > 0.0


def test_isolation_detection():
    g = TopologicalGraph()
    g.add_node(label="A", node_id="a")
    g.add_node(label="B", node_id="b")
    g.add_node(label="Isolated", node_id="iso")
    g.add_edge("a", "b", EdgeType.EXPRESSES)
    isolated = isolation_detection(g.nodes)
    assert "iso" in isolated


def test_compute_health():
    g = _make_cycle_graph()
    health = compute_health(g.nodes)
    assert health.has_cycles
    assert isinstance(health.betweenness, dict)


def test_compute_confidence():
    g = _make_star_graph()
    conf = compute_confidence(g.nodes)
    assert 0.0 <= conf.confidence <= 1.0


def test_grounding_ratio_none_grounded():
    g = TopologicalGraph()
    g.add_node(label="A", node_id="a")
    g.add_node(label="B", node_id="b")
    g.add_edge("a", "b", EdgeType.EXPRESSES)
    assert compute_grounding_ratio(g.nodes) == 0.0


def test_grounding_ratio_all_grounded():
    g = TopologicalGraph()
    g.add_node(label="A", node_id="a")
    g.add_node(label="B", node_id="b")
    g.add_edge("a", "b", EdgeType.EXPRESSES, grounded=True)
    assert compute_grounding_ratio(g.nodes) == 1.0
