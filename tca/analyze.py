"""
TCA Analyze — Structural analysis from pure topology.

Finds problems in any directed graph using graph algorithms:
  - betweenness centrality (bottleneck nodes)
  - clustering coefficient (echo chambers)
  - path diversity (robust vs fragile reasoning)
  - cycle detection (circular reasoning)
  - bridge detection (critical connections)
  - isolation detection (dead ends)

Zero learned parameters. All metrics computed from edge structure.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from tca.graph import TopologicalNode


# --- Health ---

@dataclass
class HealthReport:
    betweenness: dict[str, float] = field(default_factory=dict)
    clustering: dict[str, float] = field(default_factory=dict)
    path_diversity: float = 0.0
    cycles: list[list[str]] = field(default_factory=list)
    bridges: list[tuple[str, str]] = field(default_factory=list)
    isolated: list[str] = field(default_factory=list)

    @property
    def cycle_count(self) -> int:
        return len(self.cycles)

    @property
    def has_cycles(self) -> bool:
        return len(self.cycles) > 0

    @property
    def bridge_count(self) -> int:
        return len(self.bridges)

    @property
    def isolation_count(self) -> int:
        return len(self.isolated)


def _get_adjacency(subgraph: dict[str, TopologicalNode]
                   ) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {nid: set() for nid in subgraph}
    for nid, node in subgraph.items():
        for target_id in node.edges:
            if target_id in subgraph:
                adj[nid].add(target_id)
    return adj


def _get_undirected_adjacency(subgraph: dict[str, TopologicalNode]
                              ) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {nid: set() for nid in subgraph}
    for nid, node in subgraph.items():
        for target_id in node.edges:
            if target_id in subgraph:
                adj[nid].add(target_id)
                adj[target_id].add(nid)
    return adj


def betweenness_centrality(subgraph: dict[str, TopologicalNode]
                           ) -> dict[str, float]:
    """Brandes' algorithm. High centrality = bottleneck (single point of failure)."""
    nodes = list(subgraph.keys())
    adj = _get_adjacency(subgraph)
    centrality: dict[str, float] = {n: 0.0 for n in nodes}

    for s in nodes:
        stack: list[str] = []
        pred: dict[str, list[str]] = {n: [] for n in nodes}
        sigma: dict[str, int] = {n: 0 for n in nodes}
        sigma[s] = 1
        dist: dict[str, int] = {n: -1 for n in nodes}
        dist[s] = 0
        queue: deque[str] = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj.get(v, set()):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta: dict[str, float] = {n: 0.0 for n in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                centrality[w] += delta[w]

    n = len(nodes)
    if n > 2:
        factor = 1.0 / ((n - 1) * (n - 2))
        centrality = {k: v * factor for k, v in centrality.items()}

    return centrality


def clustering_coefficient(subgraph: dict[str, TopologicalNode]
                           ) -> dict[str, float]:
    """High clustering = tightly connected neighborhood (echo chamber risk)."""
    adj = _get_undirected_adjacency(subgraph)
    cc: dict[str, float] = {}

    for nid in subgraph:
        neighbors = adj[nid]
        k = len(neighbors)
        if k < 2:
            cc[nid] = 0.0
            continue
        links = 0
        nb_list = list(neighbors)
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                if nb_list[j] in adj[nb_list[i]]:
                    links += 1
        cc[nid] = (2.0 * links) / (k * (k - 1))

    return cc


def path_diversity(subgraph: dict[str, TopologicalNode],
                   source: str, target: str,
                   max_paths: int = 10) -> float:
    """Count distinct paths. Multiple paths = robust structure."""
    if source not in subgraph or target not in subgraph:
        return 0.0

    adj = _get_adjacency(subgraph)
    paths_found = 0

    stack: list[tuple[str, list[str]]] = [(source, [source])]
    while stack and paths_found < max_paths:
        current, path = stack.pop()
        if current == target and len(path) > 1:
            paths_found += 1
            continue
        for nb in adj.get(current, set()):
            if nb not in path:
                stack.append((nb, path + [nb]))

    return min(1.0, paths_found / max_paths)


def cycle_detection(subgraph: dict[str, TopologicalNode]
                    ) -> list[list[str]]:
    """Detect cycles (circular reasoning / feedback traps)."""
    adj = _get_adjacency(subgraph)
    cycles: list[list[str]] = []
    visited: set[str] = set()

    for start in subgraph:
        if start in visited:
            continue
        stack: list[tuple[str, list[str], set[str]]] = [
            (start, [start], {start})
        ]
        while stack:
            current, path, path_set = stack.pop()
            visited.add(current)
            for nb in adj.get(current, set()):
                if nb in path_set:
                    cycle_start = path.index(nb)
                    cycle = path[cycle_start:] + [nb]
                    min_idx = cycle[:-1].index(min(cycle[:-1]))
                    normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                    if normalized not in cycles:
                        cycles.append(normalized)
                elif nb not in visited:
                    stack.append((nb, path + [nb], path_set | {nb}))

    return cycles


def bridge_detection(subgraph: dict[str, TopologicalNode]
                     ) -> list[tuple[str, str]]:
    """Edges whose removal disconnects the graph. Critical connections."""
    adj = _get_undirected_adjacency(subgraph)
    nodes = list(subgraph.keys())
    bridges: list[tuple[str, str]] = []

    if len(nodes) < 2:
        return bridges

    edges_checked: set[tuple[str, str]] = set()
    for nid in nodes:
        for nb in adj[nid]:
            edge = tuple(sorted([nid, nb]))
            if edge in edges_checked:
                continue
            edges_checked.add(edge)

            adj[nid].discard(nb)
            adj[nb].discard(nid)

            reachable: set[str] = set()
            queue: deque[str] = deque([nid])
            reachable.add(nid)
            while queue:
                v = queue.popleft()
                for w in adj[v]:
                    if w not in reachable:
                        reachable.add(w)
                        queue.append(w)

            if nb not in reachable:
                bridges.append((nid, nb))

            adj[nid].add(nb)
            adj[nb].add(nid)

    return bridges


def isolation_detection(subgraph: dict[str, TopologicalNode],
                        conclusion_id: str | None = None
                        ) -> list[str]:
    """Nodes unreachable from the rest of the graph."""
    if not subgraph:
        return []

    adj = _get_undirected_adjacency(subgraph)

    if conclusion_id is None or conclusion_id not in subgraph:
        conclusion_id = max(subgraph,
                           key=lambda n: subgraph[n].activation)

    reachable: set[str] = set()
    queue: deque[str] = deque([conclusion_id])
    reachable.add(conclusion_id)
    while queue:
        v = queue.popleft()
        for w in adj[v]:
            if w not in reachable:
                reachable.add(w)
                queue.append(w)

    return [nid for nid in subgraph if nid not in reachable]


def compute_health(subgraph: dict[str, TopologicalNode],
                   source: str | None = None,
                   target: str | None = None) -> HealthReport:
    """Full structural health report."""
    report = HealthReport()
    report.betweenness = betweenness_centrality(subgraph)
    report.clustering = clustering_coefficient(subgraph)
    report.cycles = cycle_detection(subgraph)
    report.bridges = bridge_detection(subgraph)
    report.isolated = isolation_detection(subgraph)

    if source and target:
        report.path_diversity = path_diversity(subgraph, source, target)

    return report


# --- Confidence ---

@dataclass
class ConfidenceReport:
    confidence: float
    path_diversity_score: float
    cycle_penalty: float
    grounding_ratio: float

    W_PATH = 0.4
    W_CYCLE = 0.3
    W_GROUNDING = 0.3


def compute_grounding_ratio(subgraph: dict[str, TopologicalNode]) -> float:
    """Proportion of edges validated (grounded=True)."""
    total_edges = 0
    grounded_edges = 0
    for node in subgraph.values():
        for rels in node.edges.values():
            for rel in rels:
                total_edges += 1
                if rel.grounded:
                    grounded_edges += 1
    return grounded_edges / total_edges if total_edges > 0 else 0.0


def compute_confidence(subgraph: dict[str, TopologicalNode],
                       source: str | None = None,
                       target: str | None = None) -> ConfidenceReport:
    """Confidence computed from topology, not a learned scalar.

    confidence = path_diversity * 0.4 + (1 - cycle_ratio) * 0.3 + grounding_ratio * 0.3
    """
    if source and target:
        pd = path_diversity(subgraph, source, target)
    else:
        pd = 0.0

    cycles = cycle_detection(subgraph)
    total_nodes = len(subgraph)
    if total_nodes > 0:
        cycle_nodes: set[str] = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)
        cycle_ratio = len(cycle_nodes) / total_nodes
    else:
        cycle_ratio = 0.0

    gr = compute_grounding_ratio(subgraph)

    confidence = (
        pd * ConfidenceReport.W_PATH +
        (1.0 - cycle_ratio) * ConfidenceReport.W_CYCLE +
        gr * ConfidenceReport.W_GROUNDING
    )

    return ConfidenceReport(
        confidence=confidence,
        path_diversity_score=pd,
        cycle_penalty=cycle_ratio,
        grounding_ratio=gr,
    )
