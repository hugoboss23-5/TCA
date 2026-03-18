"""Tests for tca.graph — data structures."""

from tca.graph import EdgeType, EdgeRelation, TopologicalNode, TopologicalGraph


def test_edge_type_values():
    assert EdgeType.MIRRORS.value == "MIRRORS"
    assert EdgeType.SEEKS.value == "SEEKS"
    assert len(EdgeType) == 7


def test_edge_relation_clamps_weight():
    rel = EdgeRelation(target_id="x", edge_type=EdgeType.MIRRORS, weight=20.0)
    assert rel.weight == 10.0
    rel2 = EdgeRelation(target_id="x", edge_type=EdgeType.MIRRORS, weight=-5.0)
    assert rel2.weight == 0.0


def test_node_add_edge():
    node = TopologicalNode(id="a", label="A")
    node.add_edge("b", EdgeType.EXPRESSES, 0.8)
    node.add_edge("c", EdgeType.REMOVES, 0.5)
    assert node.edge_count() == 2
    assert node.get_neighbors() == {"b", "c"}


def test_node_edges_by_type():
    node = TopologicalNode(id="a", label="A")
    node.add_edge("b", EdgeType.EXPRESSES, 0.8)
    node.add_edge("c", EdgeType.REMOVES, 0.5)
    node.add_edge("d", EdgeType.EXPRESSES, 0.3)
    expr = node.get_edges_by_type(EdgeType.EXPRESSES)
    assert len(expr) == 2


def test_node_edge_type_distribution():
    node = TopologicalNode(id="a", label="A")
    node.add_edge("b", EdgeType.EXPRESSES, 0.8)
    node.add_edge("c", EdgeType.REMOVES, 0.5)
    dist = node.edge_type_distribution()
    assert dist[EdgeType.EXPRESSES] == 1
    assert dist[EdgeType.REMOVES] == 1
    assert dist[EdgeType.MIRRORS] == 0


def test_graph_add_nodes_and_edges():
    g = TopologicalGraph()
    g.add_node(label="A", node_id="a")
    g.add_node(label="B", node_id="b")
    g.add_edge("a", "b", EdgeType.EXPRESSES, 1.0)
    assert g.node_count == 2
    assert g.total_edge_count() == 1


def test_graph_add_edge_missing_source():
    g = TopologicalGraph()
    g.add_node(label="A", node_id="a")
    result = g.add_edge("missing", "a", EdgeType.EXPRESSES)
    assert result is None


def test_graph_get_node():
    g = TopologicalGraph()
    g.add_node(label="A", node_id="a")
    assert g.get_node("a") is not None
    assert g.get_node("missing") is None
