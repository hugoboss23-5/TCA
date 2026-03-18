"""Tests for tca.engine — full flow integration tests."""

from tca.engine import (
    create_graph, add_node, add_edge, run_analysis,
    apply_solution, export_state, export_boot,
)


def test_full_flow():
    """Create graph → add nodes/edges → analyze → get problems."""
    result = create_graph(name="Test")
    gid = result["graph_id"]

    add_node(gid, "Founder", "founder")
    add_node(gid, "Product", "product")
    add_node(gid, "Users", "users")
    add_node(gid, "Revenue", "revenue")
    add_node(gid, "Mission", "mission")

    add_edge(gid, "founder", "product", "EXPRESSES", 1.0)
    add_edge(gid, "product", "users", "SEEKS", 1.0)
    add_edge(gid, "users", "revenue", "SEEKS", 1.0)
    add_edge(gid, "revenue", "mission", "REMOVES", 1.0)

    analysis = run_analysis(gid)
    assert analysis is not None
    assert analysis["node_count"] == 5
    assert analysis["edge_count"] == 4
    assert len(analysis["problems"]) > 0
    assert len(analysis["questions"]) > 0


def test_template_loads():
    """All 6 templates load and produce analysis."""
    templates = ["economics", "tanakh", "us_geopolitics",
                 "china_geopolitics", "apple", "openai"]
    for name in templates:
        result = create_graph(name=name, template=name)
        gid = result["graph_id"]
        analysis = run_analysis(gid)
        assert analysis is not None, f"Template '{name}' returned None"
        assert analysis["node_count"] > 0, f"Template '{name}' has no nodes"
        assert analysis["edge_count"] > 0, f"Template '{name}' has no edges"
        assert len(analysis["problems"]) > 0, f"Template '{name}' found no problems"


def test_apply_solution():
    """Apply a solution and verify re-analysis."""
    result = create_graph(name="Test", template="economics")
    gid = result["graph_id"]
    analysis = run_analysis(gid)
    assert analysis is not None

    solutions = analysis["solutions"]
    if solutions:
        updated = apply_solution(gid, 0)
        assert updated is not None
        assert "node_count" in updated


def test_export_json():
    result = create_graph(name="Test", template="apple")
    gid = result["graph_id"]
    exported = export_state(gid)
    assert exported is not None
    assert "nodes" in exported
    assert "edges" in exported
    assert len(exported["nodes"]) > 0


def test_export_boot():
    result = create_graph(name="Test", template="apple")
    gid = result["graph_id"]
    exported = export_boot(gid)
    assert exported is not None
    assert exported["format"] == "boot"
    # Boot format has no labels
    for node in exported["nodes"]:
        assert "label" not in node


def test_dead_end_detection():
    """Isolated node with no edges should be flagged."""
    result = create_graph(name="Test")
    gid = result["graph_id"]
    add_node(gid, "Connected", "a")
    add_node(gid, "Also Connected", "b")
    add_node(gid, "Dead End", "dead")
    add_edge(gid, "a", "b", "EXPRESSES", 1.0)

    analysis = run_analysis(gid)
    dead_ends = [p for p in analysis["problems"] if p["type"] == "dead_end"]
    assert len(dead_ends) >= 1
    dead_labels = [p["label"] for p in dead_ends]
    assert "Dead End" in dead_labels


def test_contradiction_detection():
    """REMOVES edges should surface as contradictions."""
    result = create_graph(name="Test")
    gid = result["graph_id"]
    add_node(gid, "Growth", "growth")
    add_node(gid, "Sustainability", "sustain")
    add_edge(gid, "growth", "sustain", "REMOVES", 1.0)

    analysis = run_analysis(gid)
    contradictions = [p for p in analysis["problems"] if p["type"] == "contradiction"]
    assert len(contradictions) >= 1


def test_seeks_creates_questions():
    """SEEKS edges should create questions."""
    result = create_graph(name="Test")
    gid = result["graph_id"]
    add_node(gid, "Startup", "startup")
    add_node(gid, "PMF", "pmf")
    add_edge(gid, "startup", "pmf", "SEEKS", 1.0)

    analysis = run_analysis(gid)
    assert len(analysis["questions"]) >= 1


def test_invalid_edge_type():
    result = create_graph(name="Test")
    gid = result["graph_id"]
    add_node(gid, "A", "a")
    add_node(gid, "B", "b")
    err = add_edge(gid, "a", "b", "INVALID_TYPE", 1.0)
    assert "error" in err


def test_nonexistent_graph():
    analysis = run_analysis("nonexistent")
    assert analysis is None
