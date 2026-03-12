"""Tests for the TCA Solution Engine."""

import unittest

from tca.L2_graph.operations import TopologicalGraph
from tca.L2_graph.topo_node import EdgeType
from tca.L5_metacognition.solutions import TopologicalSolver, Solution


def build_test_graph():
    """Build a graph with known problems for solution testing."""
    g = TopologicalGraph()

    # Dead end: wellbeing receives SEEKS but outputs nothing.
    g.add_node(label="Wellbeing", node_id="wellbeing")
    g.add_node(label="GDP", node_id="gdp")
    g.add_edge("gdp", "wellbeing", EdgeType.SEEKS)

    # Orphan source: Fed sends EXPRESSES but nothing points to it.
    g.add_node(label="Fed", node_id="fed")
    g.add_node(label="Currency", node_id="currency")
    g.add_edge("fed", "currency", EdgeType.EXPRESSES)

    # Feedback trap: Fiat ↔ Debt (MIRRORS both ways).
    g.add_node(label="Fiat", node_id="fiat")
    g.add_node(label="Debt", node_id="debt")
    g.add_edge("fiat", "debt", EdgeType.MIRRORS)
    g.add_edge("debt", "fiat", EdgeType.MIRRORS)

    # Contradiction: Ideology REMOVES Inequality.
    g.add_node(label="Ideology", node_id="ideology")
    g.add_node(label="Inequality", node_id="inequality")
    g.add_edge("ideology", "inequality", EdgeType.REMOVES)

    # Star center: 3 nodes all SEEK center.
    g.add_node(label="Center", node_id="center")
    g.add_node(label="A", node_id="a")
    g.add_node(label="B", node_id="b")
    g.add_node(label="C", node_id="c")
    g.add_edge("a", "center", EdgeType.SEEKS)
    g.add_edge("b", "center", EdgeType.SEEKS)
    g.add_edge("c", "center", EdgeType.SEEKS)

    return g


class TestDeadEndSolver(unittest.TestCase):
    def test_finds_dead_ends(self):
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_dead_ends()
        assert any("Wellbeing" in str(s.problem_description) or
                    "wellbeing" in str(s.details) for s in solutions)

    def test_proposes_outgoing_edge(self):
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_dead_ends()
        assert any(s.action == "add_edge" for s in solutions)

    def test_pairs_dead_ends_with_orphans(self):
        """Dead ends and orphans are natural bridges."""
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_dead_ends()
        assert any("orphan" in s.reasoning.lower() or
                    "origin" in s.reasoning.lower() or
                    "pair" in s.reasoning.lower()
                    for s in solutions)


class TestOrphanSolver(unittest.TestCase):
    def test_finds_orphans(self):
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_orphan_sources()
        assert any("Fed" in str(s.problem_description) or
                    "fed" in str(s.details) for s in solutions)

    def test_proposes_incoming_edge(self):
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_orphan_sources()
        assert any(s.action == "add_edge" for s in solutions)


class TestFeedbackTrapSolver(unittest.TestCase):
    def test_finds_trap(self):
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_feedback_traps()
        assert any("Fiat" in str(s.problem_description) or
                    "Debt" in str(s.problem_description) for s in solutions)

    def test_proposes_exit_edge(self):
        """The solution to a loop is an exit — not destroying the loop."""
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_feedback_traps()
        assert any("exit" in s.reasoning.lower() or
                    "external" in s.reasoning.lower() or
                    "outside" in s.reasoning.lower() or
                    "spiral" in s.reasoning.lower()
                    for s in solutions)


class TestStarSolver(unittest.TestCase):
    def test_finds_star(self):
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_star_topology(threshold=0.15)
        assert len(solutions) > 0

    def test_proposes_peripheral_bridge(self):
        """Solution to a star is bridges between periphery nodes."""
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_star_topology(threshold=0.15)
        assert any(s.action == "add_edge" for s in solutions)
        assert any("bypass" in s.reasoning.lower() or
                    "bridge" in s.reasoning.lower() or
                    "peripheral" in s.reasoning.lower()
                    for s in solutions)


class TestUngroundedSeeksSolver(unittest.TestCase):
    def test_finds_ungrounded_seeks(self):
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_ungrounded_seeks()
        assert any("GDP" in str(s.problem_description) or
                    "Wellbeing" in str(s.problem_description) for s in solutions)

    def test_proposes_intermediate_or_removal(self):
        """Either propose evidence or propose removing the edge."""
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_ungrounded_seeks()
        assert any(s.action in ("add_node", "add_edge", "remove_edge")
                    for s in solutions)


class TestContradictionSolver(unittest.TestCase):
    def test_finds_contradictions(self):
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_contradictions()
        assert any("Ideology" in str(s.problem_description) or
                    "Inequality" in str(s.problem_description) for s in solutions)

    def test_identifies_structural_lie(self):
        """When a narrative node REMOVES evidence of system output, flag it."""
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_contradictions()
        assert any("lie" in s.reasoning.lower() or
                    "deny" in s.reasoning.lower() or
                    "acknowledge" in s.reasoning.lower()
                    for s in solutions)


class TestSolveAll(unittest.TestCase):
    def test_returns_sorted_by_confidence(self):
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_all()
        for i in range(len(solutions) - 1):
            assert solutions[i].confidence >= solutions[i + 1].confidence

    def test_covers_all_problem_types(self):
        g = build_test_graph()
        solver = TopologicalSolver(g)
        solutions = solver.solve_all()
        types_found = {s.problem_type for s in solutions}
        # Should find at least 4 of 6 problem types in our test graph.
        assert len(types_found) >= 4

    def test_solve_report_is_readable(self):
        g = build_test_graph()
        solver = TopologicalSolver(g)
        report = solver.solve_and_report()
        assert "PROBLEM:" in report
        assert "SOLUTION:" in report
        assert "REASON:" in report


class TestApplySolution(unittest.TestCase):
    def test_applying_solution_modifies_graph(self):
        """When a solution is applied, the graph actually changes."""
        g = build_test_graph()
        original_edge_count = g.total_edge_count()
        solver = TopologicalSolver(g)
        solutions = solver.solve_all()
        add_solutions = [s for s in solutions if s.action == "add_edge"]
        if add_solutions:
            result = solver.apply_solution(add_solutions[0])
            assert result is True
            assert g.total_edge_count() > original_edge_count

    def test_applied_solution_reduces_problems(self):
        """After applying solutions, re-analysis should find fewer problems."""
        g = build_test_graph()
        solver = TopologicalSolver(g)
        initial_solutions = solver.solve_all()
        initial_count = len(initial_solutions)

        for s in initial_solutions:
            if s.confidence > 0.7:
                solver.apply_solution(s)

        new_solutions = solver.solve_all()
        assert len(new_solutions) <= initial_count


if __name__ == "__main__":
    unittest.main()
