"""Test TCA MCP tools directly — no MCP transport needed."""

from api import engine


def test_template_loads_and_analyzes():
    result = engine.create_graph(name="Economics", template="economics")
    graph_id = result["graph_id"]
    analysis = engine.run_analysis(graph_id)
    assert analysis is not None, "Analysis should not be None"
    assert analysis["node_count"] > 5, f"Expected >5 nodes, got {analysis['node_count']}"
    solutions = analysis.get("solutions", [])
    assert len(solutions) > 0, "Should have at least one solution"
    print(f"PASS: economics template — {analysis['node_count']} nodes, "
          f"{analysis['edge_count']} edges, {len(solutions)} solutions")


def test_all_templates():
    templates = ["economics", "tanakh", "us_geopolitics",
                 "china_geopolitics", "apple", "openai"]
    for name in templates:
        result = engine.create_graph(name=name.title(), template=name)
        graph_id = result["graph_id"]
        analysis = engine.run_analysis(graph_id)
        assert analysis is not None, f"Template {name}: analysis is None"
        assert analysis["node_count"] > 5, f"Template {name}: too few nodes"
        print(f"PASS: {name} — {analysis['node_count']} nodes, "
              f"{len(analysis.get('solutions', []))} solutions")


def test_apply_solution():
    result = engine.create_graph(name="Test", template="economics")
    graph_id = result["graph_id"]
    analysis = engine.run_analysis(graph_id)
    initial = len(analysis.get("solutions", []))
    assert initial > 0, "Need at least one solution to test apply"
    applied = engine.apply_solution(graph_id, 0)
    assert applied is not None, "apply_solution returned None"
    assert "error" not in applied or applied.get("node_count", 0) > 0
    print(f"PASS: applied solution, solutions {initial} → "
          f"{len(applied.get('solutions', []))}")


def test_export_json():
    result = engine.create_graph(name="Test", template="economics")
    graph_id = result["graph_id"]
    state = engine.export_state(graph_id)
    assert state is not None, "export_state returned None"
    assert "nodes" in state, "Missing nodes in export"
    assert "edges" in state, "Missing edges in export"
    assert len(state["nodes"]) > 5, "Too few nodes in export"
    print(f"PASS: JSON export — {len(state['nodes'])} nodes, "
          f"{len(state['edges'])} edges")


def test_export_boot_clean():
    result = engine.create_graph(name="Test", template="economics")
    graph_id = result["graph_id"]
    boot = engine.export_boot(graph_id)
    assert boot is not None, "export_boot returned None"
    import json
    boot_str = json.dumps(boot)
    # Boot protocol should not have labels (private data).
    for node in boot["nodes"]:
        assert "label" not in node, f"Boot node has label: {node}"
    print(f"PASS: boot protocol clean — {boot['node_count']} nodes, "
          f"{boot['edge_count']} edges, no labels")


def test_solve():
    result = engine.create_graph(name="Test", template="openai")
    graph_id = result["graph_id"]
    analysis = engine.run_analysis(graph_id)
    solutions = analysis.get("solutions", [])
    assert len(solutions) > 0, "OpenAI template should have solutions"
    # Check solutions are sorted by confidence.
    confidences = [s.get("confidence", 0) for s in solutions]
    assert confidences == sorted(confidences, reverse=True), \
        "Solutions should be sorted by confidence descending"
    print(f"PASS: solve — {len(solutions)} solutions, "
          f"top confidence: {confidences[0]}")


if __name__ == "__main__":
    test_template_loads_and_analyzes()
    test_all_templates()
    test_apply_solution()
    test_export_json()
    test_export_boot_clean()
    test_solve()
    print("\nAll MCP tool tests passed.")
