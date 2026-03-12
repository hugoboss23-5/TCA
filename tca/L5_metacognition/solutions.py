"""
TCA Solution Engine

Every topological problem has a topological inverse that IS the solution.
The same analysis that finds the disease prescribes the cure.

This does NOT generate text answers. It generates STRUCTURAL MODIFICATIONS
to the graph — new edges, new nodes, removed edges — that resolve the
topological pathology. The human decides whether to apply them.

Solutions are PROPOSALS, not actions. Each solution says:
"If you add THIS edge / node / connection, THIS problem resolves."
The human (or L5 metacognition) decides whether to accept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tca.L2_graph.operations import TopologicalGraph
from tca.L2_graph.topo_node import EdgeType, TopologicalNode


# Inverse edge type map: if edges of type X flow IN, propose type Y flowing OUT.
_DEAD_END_INVERSE: dict[EdgeType, EdgeType] = {
    EdgeType.SEEKS: EdgeType.VERIFIES,      # sought → should prove something
    EdgeType.EXPRESSES: EdgeType.MIRRORS,    # receives output → should reflect
    EdgeType.VERIFIES: EdgeType.EXPRESSES,   # proven → should produce
    EdgeType.MIRRORS: EdgeType.EXPRESSES,    # mirrored → should express
    EdgeType.INHERITS: EdgeType.EXPRESSES,   # inherited into → should express
    EdgeType.BOUNDS: EdgeType.VERIFIES,      # constrained → should verify
    EdgeType.REMOVES: EdgeType.BOUNDS,       # attacked → should constrain
}

# Inverse for orphans: if edges of type X flow OUT, propose type Y flowing IN.
_ORPHAN_INVERSE: dict[EdgeType, EdgeType] = {
    EdgeType.EXPRESSES: EdgeType.INHERITS,   # creates → must be created by
    EdgeType.BOUNDS: EdgeType.VERIFIES,      # constrains → authority must be verified
    EdgeType.REMOVES: EdgeType.BOUNDS,       # destroys → must be constrained
    EdgeType.MIRRORS: EdgeType.INHERITS,     # reflects → must derive from
    EdgeType.VERIFIES: EdgeType.BOUNDS,      # proves → must be bounded
    EdgeType.SEEKS: EdgeType.VERIFIES,       # wants → must be verified
    EdgeType.INHERITS: EdgeType.VERIFIES,    # derives → must be verified
}

# For star bypass: when two peripherals share the same edge type to center.
_STAR_BRIDGE_TYPE: dict[EdgeType, EdgeType] = {
    EdgeType.SEEKS: EdgeType.MIRRORS,        # both seek same → they mirror each other
    EdgeType.MIRRORS: EdgeType.INHERITS,     # both mirror same → shared ancestry
    EdgeType.INHERITS: EdgeType.MIRRORS,     # both inherit same → they mirror
    EdgeType.EXPRESSES: EdgeType.MIRRORS,    # both express into same → they mirror
    EdgeType.VERIFIES: EdgeType.MIRRORS,     # both verify same → they mirror
    EdgeType.BOUNDS: EdgeType.MIRRORS,       # both bound same → they mirror
    EdgeType.REMOVES: EdgeType.BOUNDS,       # both attack same → need boundary
}


@dataclass
class Solution:
    """A proposed structural modification to the graph."""
    problem_type: str          # "dead_end", "orphan", "star", "trap", "ungrounded", "contradiction"
    problem_description: str   # Human-readable description of what's wrong
    action: str                # "add_edge", "add_node", "remove_edge", "redirect_edge"
    details: dict              # The specific modification proposed
    reasoning: str             # WHY this inverse resolves the problem
    confidence: float          # How structurally certain this solution is (0-1)


def _build_incoming_map(graph: TopologicalGraph
                        ) -> dict[str, list[tuple[str, EdgeType, float]]]:
    """Build a map of node_id -> list of (source_id, edge_type, weight) for incoming edges."""
    incoming: dict[str, list[tuple[str, EdgeType, float]]] = {
        nid: [] for nid in graph.nodes
    }
    for nid, node in graph.nodes.items():
        for target_id, rels in node.edges.items():
            if target_id in incoming:
                for rel in rels:
                    incoming[target_id].append((nid, rel.edge_type, rel.weight))
    return incoming


def _outgoing_edge_types(node: TopologicalNode) -> dict[EdgeType, int]:
    """Count outgoing edges by type."""
    counts: dict[EdgeType, int] = {}
    for rels in node.edges.values():
        for rel in rels:
            counts[rel.edge_type] = counts.get(rel.edge_type, 0) + 1
    return counts


class TopologicalSolver:
    """Generates solutions by inverting topological problems.

    Every diagnosis has an inverse:

    DEAD END (absorbs, never outputs)
    → Propose outgoing edges. A dead end that gains outgoing edges
      becomes a throughput node.

    ORPHAN SOURCE (sends edges, no origin)
    → Propose incoming edges. An orphan that gains incoming edges
      becomes accountable.

    STAR TOPOLOGY (one node concentrates all inflow)
    → Propose bridges between peripheral nodes that bypass the center.

    FEEDBACK TRAP (A→B→A with no exit)
    → Propose an exit edge from one loop node to something outside.

    UNGROUNDED SEEKS (desired but unproven connection)
    → Propose an intermediate verifier, or remove the edge.

    CONTRADICTION (A REMOVES B but both coexist)
    → Propose resolution: one wins, bridge node, or scope separation.
    """

    def __init__(self, graph: TopologicalGraph):
        self.graph = graph

    def solve_dead_ends(self) -> list[Solution]:
        """For every dead-end node, propose outgoing edges."""
        incoming = _build_incoming_map(self.graph)
        solutions: list[Solution] = []

        # Find orphan IDs for pairing.
        orphan_ids = set()
        for nid, node in self.graph.nodes.items():
            if node.edge_count() > 0 and not incoming[nid]:
                orphan_ids.add(nid)

        for nid, node in self.graph.nodes.items():
            has_incoming = len(incoming[nid]) > 0
            has_outgoing = node.edge_count() > 0

            if has_incoming and not has_outgoing:
                # This is a dead end.
                # Determine dominant incoming edge type.
                type_counts: dict[EdgeType, int] = {}
                for _, etype, _ in incoming[nid]:
                    type_counts[etype] = type_counts.get(etype, 0) + 1
                dominant_in = max(type_counts, key=lambda t: type_counts[t])
                proposed_out = _DEAD_END_INVERSE.get(dominant_in, EdgeType.EXPRESSES)

                # Try to pair with an orphan.
                best_target = None
                for oid in orphan_ids:
                    if oid != nid:
                        best_target = oid
                        break

                if best_target is not None:
                    target_label = self.graph.get_node(best_target).label
                    solutions.append(Solution(
                        problem_type="dead_end",
                        problem_description=(
                            f"'{node.label}' ({nid}) absorbs edges but outputs nothing"
                        ),
                        action="add_edge",
                        details={
                            "node_id": nid,
                            "node_label": node.label,
                            "target_id": best_target,
                            "target_label": target_label,
                            "edge_type": proposed_out.value,
                        },
                        reasoning=(
                            f"Dead end paired with orphan source '{target_label}'. "
                            f"Incoming {dominant_in.value} inverts to outgoing "
                            f"{proposed_out.value}. The dead end becomes throughput."
                        ),
                        confidence=0.7,
                    ))
                else:
                    # No orphan to pair with — propose generic outgoing edge.
                    solutions.append(Solution(
                        problem_type="dead_end",
                        problem_description=(
                            f"'{node.label}' ({nid}) absorbs edges but outputs nothing"
                        ),
                        action="add_edge",
                        details={
                            "node_id": nid,
                            "node_label": node.label,
                            "edge_type": proposed_out.value,
                            "target_id": None,
                        },
                        reasoning=(
                            f"Incoming {dominant_in.value} inverts to outgoing "
                            f"{proposed_out.value}. Needs a target — what does "
                            f"'{node.label}' produce or verify?"
                        ),
                        confidence=0.5,
                    ))

        return solutions

    def solve_orphan_sources(self) -> list[Solution]:
        """For every orphan source, propose incoming edges."""
        incoming = _build_incoming_map(self.graph)
        solutions: list[Solution] = []

        # Find dead-end IDs for pairing.
        dead_end_ids = set()
        for nid, node in self.graph.nodes.items():
            if incoming[nid] and node.edge_count() == 0:
                dead_end_ids.add(nid)

        for nid, node in self.graph.nodes.items():
            has_incoming = len(incoming[nid]) > 0
            has_outgoing = node.edge_count() > 0

            if has_outgoing and not has_incoming:
                # This is an orphan source.
                out_types = _outgoing_edge_types(node)
                dominant_out = max(out_types, key=lambda t: out_types[t])
                proposed_in = _ORPHAN_INVERSE.get(dominant_out, EdgeType.VERIFIES)

                # Try to pair with a dead end.
                best_source = None
                for did in dead_end_ids:
                    if did != nid:
                        best_source = did
                        break

                if best_source is not None:
                    source_label = self.graph.get_node(best_source).label
                    solutions.append(Solution(
                        problem_type="orphan",
                        problem_description=(
                            f"'{node.label}' ({nid}) sends edges but has no origin"
                        ),
                        action="add_edge",
                        details={
                            "node_id": nid,
                            "node_label": node.label,
                            "source_id": best_source,
                            "source_label": source_label,
                            "edge_type": proposed_in.value,
                        },
                        reasoning=(
                            f"Orphan paired with dead end '{source_label}'. "
                            f"Outgoing {dominant_out.value} inverts to incoming "
                            f"{proposed_in.value}. The orphan becomes accountable."
                        ),
                        confidence=0.7,
                    ))
                else:
                    solutions.append(Solution(
                        problem_type="orphan",
                        problem_description=(
                            f"'{node.label}' ({nid}) sends edges but has no origin"
                        ),
                        action="add_edge",
                        details={
                            "node_id": nid,
                            "node_label": node.label,
                            "edge_type": proposed_in.value,
                            "source_id": None,
                        },
                        reasoning=(
                            f"Outgoing {dominant_out.value} inverts to incoming "
                            f"{proposed_in.value}. What creates '{node.label}'s "
                            f"authority?"
                        ),
                        confidence=0.5,
                    ))

        return solutions

    def solve_star_topology(self, threshold: float = 0.3) -> list[Solution]:
        """For star-topology centers, propose bypass bridges."""
        incoming = _build_incoming_map(self.graph)
        solutions: list[Solution] = []
        total_nodes = self.graph.node_count
        if total_nodes < 3:
            return solutions

        for nid in self.graph.nodes:
            in_count = len(incoming[nid])
            star_ratio = in_count / total_nodes
            if star_ratio <= threshold:
                continue

            node = self.graph.get_node(nid)
            # Gather peripherals: nodes that connect TO this center.
            peripherals: list[tuple[str, EdgeType]] = []
            for src_id, etype, _ in incoming[nid]:
                peripherals.append((src_id, etype))

            # Find pairs of peripherals that don't connect to each other.
            for i in range(len(peripherals)):
                for j in range(i + 1, len(peripherals)):
                    pid_a, etype_a = peripherals[i]
                    pid_b, etype_b = peripherals[j]
                    node_a = self.graph.get_node(pid_a)
                    node_b = self.graph.get_node(pid_b)

                    # Check if they already connect.
                    if pid_b in node_a.edges or pid_a in node_b.edges:
                        continue

                    # Determine bridge type from shared relationship to center.
                    bridge_type = _STAR_BRIDGE_TYPE.get(etype_a, EdgeType.MIRRORS)

                    solutions.append(Solution(
                        problem_type="star",
                        problem_description=(
                            f"'{node.label}' ({nid}) is a star center — "
                            f"{in_count}/{total_nodes} nodes flow through it"
                        ),
                        action="add_edge",
                        details={
                            "center_id": nid,
                            "center_label": node.label,
                            "source_id": pid_a,
                            "source_label": node_a.label,
                            "target_id": pid_b,
                            "target_label": node_b.label,
                            "edge_type": bridge_type.value,
                        },
                        reasoning=(
                            f"Peripheral bridge bypasses star center. "
                            f"'{node_a.label}' and '{node_b.label}' both connect "
                            f"to '{node.label}' but not each other. Bridge makes "
                            f"the star a mesh."
                        ),
                        confidence=0.6,
                    ))

        return solutions

    def solve_feedback_traps(self) -> list[Solution]:
        """For feedback loops, propose exit edges."""
        solutions: list[Solution] = []
        seen_pairs: set[tuple[str, str]] = set()

        for nid, node in self.graph.nodes.items():
            for target_id, rels in node.edges.items():
                target_node = self.graph.get_node(target_id)
                if target_node is None:
                    continue
                # Check for bidirectional edge (A→B and B→A).
                if nid not in target_node.edges:
                    continue
                pair = tuple(sorted([nid, target_id]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Determine loop edge types.
                forward_types = {r.edge_type for r in rels}
                back_rels = target_node.edges.get(nid, [])
                back_types = {r.edge_type for r in back_rels}

                # Find an external node not in the loop.
                external_id = None
                external_label = ""
                for eid in self.graph.nodes:
                    if eid != nid and eid != target_id:
                        external_id = eid
                        external_label = self.graph.get_node(eid).label
                        break

                if external_id is None:
                    continue

                # Choose exit edge type based on loop types.
                if EdgeType.MIRRORS in forward_types and EdgeType.MIRRORS in back_types:
                    exit_type = EdgeType.SEEKS
                    exit_reason = (
                        "Identity loop (MIRRORS↔MIRRORS). Exit via SEEKS: "
                        "what does this loop want that isn't more of itself? "
                        "The closed cycle becomes an open spiral."
                    )
                elif EdgeType.REMOVES in forward_types or EdgeType.REMOVES in back_types:
                    exit_type = EdgeType.INHERITS
                    exit_reason = (
                        "Mutual destruction loop. Exit via INHERITS to a common "
                        "ancestor outside the loop — transforms conflict into "
                        "creative tension."
                    )
                else:
                    exit_type = EdgeType.BOUNDS
                    exit_reason = (
                        "Production loop. Exit via BOUNDS from an external "
                        "constraint — what should limit this cycle? "
                        "The spiral needs a governor."
                    )

                solutions.append(Solution(
                    problem_type="trap",
                    problem_description=(
                        f"Feedback trap: '{node.label}' ↔ '{target_node.label}' "
                        f"({forward_types} / {back_types})"
                    ),
                    action="add_edge",
                    details={
                        "loop_node_a": nid,
                        "loop_label_a": node.label,
                        "loop_node_b": target_id,
                        "loop_label_b": target_node.label,
                        "exit_source": nid,
                        "exit_target": external_id,
                        "exit_target_label": external_label,
                        "edge_type": exit_type.value,
                    },
                    reasoning=exit_reason,
                    confidence=0.65,
                ))

        return solutions

    def solve_ungrounded_seeks(self) -> list[Solution]:
        """For ungrounded SEEKS edges, propose evidence paths or removal."""
        solutions: list[Solution] = []

        for nid, node in self.graph.nodes.items():
            seeks_edges = node.get_edges_by_type(EdgeType.SEEKS)
            for rel in seeks_edges:
                target_id = rel.target_id
                target_node = self.graph.get_node(target_id)
                if target_node is None:
                    continue

                # Search for evidence path: any C where nid→C and C--VERIFIES-->target.
                evidence_found = False
                for mid_id in node.get_neighbors():
                    mid_node = self.graph.get_node(mid_id)
                    if mid_node is None:
                        continue
                    if target_id in mid_node.edges:
                        for mid_rel in mid_node.edges[target_id]:
                            if mid_rel.edge_type == EdgeType.VERIFIES:
                                evidence_found = True
                                break
                    if evidence_found:
                        break

                if evidence_found:
                    # Evidence path exists — propose strengthening.
                    solutions.append(Solution(
                        problem_type="ungrounded",
                        problem_description=(
                            f"'{node.label}' SEEKS '{target_node.label}' — "
                            f"evidence path exists but SEEKS is ungrounded"
                        ),
                        action="add_edge",
                        details={
                            "source_id": nid,
                            "source_label": node.label,
                            "target_id": target_id,
                            "target_label": target_node.label,
                            "edge_type": EdgeType.VERIFIES.value,
                            "replace_seeks": True,
                        },
                        reasoning=(
                            "Evidence path found. Convert SEEKS to VERIFIES — "
                            "the connection has proof, stop wishing and start knowing."
                        ),
                        confidence=0.8,
                    ))
                else:
                    # No evidence path — propose intermediate or removal.
                    solutions.append(Solution(
                        problem_type="ungrounded",
                        problem_description=(
                            f"'{node.label}' SEEKS '{target_node.label}' — "
                            f"no evidence path exists"
                        ),
                        action="add_node",
                        details={
                            "source_id": nid,
                            "source_label": node.label,
                            "target_id": target_id,
                            "target_label": target_node.label,
                            "proposed_intermediate": (
                                f"[Evidence for {node.label}→{target_node.label}]"
                            ),
                        },
                        reasoning=(
                            f"No evidence that '{node.label}' leads to "
                            f"'{target_node.label}'. Either name the intermediate "
                            f"concept that would VERIFY this connection, or remove "
                            f"the SEEKS edge — it's a wish, not a connection."
                        ),
                        confidence=0.6,
                    ))

        return solutions

    def solve_contradictions(self) -> list[Solution]:
        """For REMOVES contradictions, propose resolution paths."""
        solutions: list[Solution] = []
        incoming = _build_incoming_map(self.graph)

        for nid, node in self.graph.nodes.items():
            removes_edges = node.get_edges_by_type(EdgeType.REMOVES)
            for rel in removes_edges:
                target_id = rel.target_id
                target_node = self.graph.get_node(target_id)
                if target_node is None:
                    continue

                # Check if mutual destruction (B also REMOVES A).
                mutual = False
                if nid in target_node.edges:
                    for back_rel in target_node.edges[nid]:
                        if back_rel.edge_type == EdgeType.REMOVES:
                            mutual = True
                            break

                # Check if anything VERIFIES both.
                verifiers_a = {src for src, et, _ in incoming[nid]
                               if et == EdgeType.VERIFIES}
                verifiers_b = {src for src, et, _ in incoming[target_id]
                               if et == EdgeType.VERIFIES}
                shared_verifiers = verifiers_a & verifiers_b

                # Check if target has other support.
                target_support = [
                    (src, et) for src, et, _ in incoming[target_id]
                    if src != nid and et != EdgeType.REMOVES
                ]

                if mutual:
                    # Mutual destruction — propose bridge node.
                    solutions.append(Solution(
                        problem_type="contradiction",
                        problem_description=(
                            f"Mutual destruction: '{node.label}' ↔ REMOVES ↔ "
                            f"'{target_node.label}'"
                        ),
                        action="add_node",
                        details={
                            "node_a_id": nid,
                            "node_a_label": node.label,
                            "node_b_id": target_id,
                            "node_b_label": target_node.label,
                            "proposed_bridge": (
                                f"[Common ground: {node.label} ∩ {target_node.label}]"
                            ),
                        },
                        reasoning=(
                            "Mutual destruction needs a bridge. What do both nodes "
                            "INHERIT from? A common ancestor transforms the "
                            "contradiction into creative tension."
                        ),
                        confidence=0.55,
                    ))
                elif shared_verifiers:
                    # Contextual contradiction — propose scope separation.
                    solutions.append(Solution(
                        problem_type="contradiction",
                        problem_description=(
                            f"'{node.label}' REMOVES '{target_node.label}' but "
                            f"both verified by {shared_verifiers}"
                        ),
                        action="add_edge",
                        details={
                            "node_a_id": nid,
                            "node_a_label": node.label,
                            "node_b_id": target_id,
                            "node_b_label": target_node.label,
                            "edge_type": EdgeType.BOUNDS.value,
                            "scope_separation": True,
                        },
                        reasoning=(
                            "Both nodes are verified — the contradiction is "
                            "contextual. Propose BOUNDS edges to separate their "
                            "domains. They coexist in different scopes."
                        ),
                        confidence=0.6,
                    ))
                elif not target_support:
                    # Target has no support — structural lie or one side wins.
                    solutions.append(Solution(
                        problem_type="contradiction",
                        problem_description=(
                            f"'{node.label}' REMOVES '{target_node.label}' — "
                            f"target has no other support"
                        ),
                        action="remove_edge",
                        details={
                            "source_id": nid,
                            "source_label": node.label,
                            "target_id": target_id,
                            "target_label": target_node.label,
                            "replace_with": EdgeType.MIRRORS.value,
                        },
                        reasoning=(
                            f"'{node.label}' REMOVES '{target_node.label}' but "
                            f"the target has no defenders. This is a structural lie — "
                            f"the node denies what it cannot disprove. Replace REMOVES "
                            f"with MIRRORS to acknowledge instead of deny."
                        ),
                        confidence=0.7,
                    ))
                else:
                    # General contradiction.
                    solutions.append(Solution(
                        problem_type="contradiction",
                        problem_description=(
                            f"'{node.label}' REMOVES '{target_node.label}'"
                        ),
                        action="add_edge",
                        details={
                            "source_id": nid,
                            "source_label": node.label,
                            "target_id": target_id,
                            "target_label": target_node.label,
                            "edge_type": EdgeType.BOUNDS.value,
                        },
                        reasoning=(
                            "Contradiction with partial support. Propose BOUNDS "
                            "edges to acknowledge the tension without destroying "
                            "either side."
                        ),
                        confidence=0.5,
                    ))

        return solutions

    def solve_all(self) -> list[Solution]:
        """Run all solvers. Return solutions sorted by confidence descending."""
        solutions = []
        solutions.extend(self.solve_dead_ends())
        solutions.extend(self.solve_orphan_sources())
        solutions.extend(self.solve_star_topology())
        solutions.extend(self.solve_feedback_traps())
        solutions.extend(self.solve_ungrounded_seeks())
        solutions.extend(self.solve_contradictions())
        solutions.sort(key=lambda s: s.confidence, reverse=True)
        return solutions

    def apply_solution(self, solution: Solution) -> bool:
        """Apply a proposed solution to the graph.

        Returns True if applied, False if the action is unrecognized.

        CRITICAL: This modifies the graph. The human must approve.
        In autonomous mode, only apply solutions with confidence > 0.8.
        """
        action = solution.action
        details = solution.details

        if action == "add_edge":
            source_id = details.get("source_id") or details.get("node_id")
            target_id = details.get("target_id")
            if source_id is None or target_id is None:
                return False
            edge_type_str = details.get("edge_type", "MIRRORS")
            edge_type = EdgeType(edge_type_str)

            # If replacing a SEEKS edge, remove the old one first.
            if details.get("replace_seeks"):
                source_node = self.graph.get_node(source_id)
                if source_node and target_id in source_node.edges:
                    source_node.edges[target_id] = [
                        r for r in source_node.edges[target_id]
                        if r.edge_type != EdgeType.SEEKS
                    ]
                    if not source_node.edges[target_id]:
                        del source_node.edges[target_id]

            self.graph.add_edge(source_id, target_id, edge_type)
            return True

        elif action == "add_node":
            # Create the intermediate node and connect it.
            label = details.get("proposed_intermediate",
                                details.get("proposed_bridge", "bridge"))
            new_node = self.graph.add_node(label=label)
            source_id = details.get("source_id") or details.get("node_a_id")
            target_id = details.get("target_id") or details.get("node_b_id")
            if source_id:
                self.graph.add_edge(source_id, new_node.id, EdgeType.INHERITS)
            if target_id:
                self.graph.add_edge(new_node.id, target_id, EdgeType.VERIFIES)
            return True

        elif action == "remove_edge":
            source_id = details.get("source_id")
            target_id = details.get("target_id")
            if source_id is None or target_id is None:
                return False
            source_node = self.graph.get_node(source_id)
            if source_node is None or target_id not in source_node.edges:
                return False
            # Remove REMOVES edges to target.
            source_node.edges[target_id] = [
                r for r in source_node.edges[target_id]
                if r.edge_type != EdgeType.REMOVES
            ]
            if not source_node.edges[target_id]:
                del source_node.edges[target_id]
            # Add replacement edge if specified.
            replace_type = details.get("replace_with")
            if replace_type:
                self.graph.add_edge(source_id, target_id, EdgeType(replace_type))
            return True

        return False

    def solve_and_report(self) -> str:
        """Run all solvers and produce a human-readable report."""
        solutions = self.solve_all()
        lines = []
        for i, s in enumerate(solutions, 1):
            lines.append(f"{'─' * 50}")
            lines.append(f"#{i} [{s.problem_type.upper()}] confidence: {s.confidence:.2f}")
            lines.append(f"PROBLEM:  {s.problem_description}")
            lines.append(f"SOLUTION: {s.action} → {s.details}")
            lines.append(f"REASON:   {s.reasoning}")
        lines.append(f"{'─' * 50}")
        lines.append(f"Total: {len(solutions)} solutions proposed")
        return "\n".join(lines)
