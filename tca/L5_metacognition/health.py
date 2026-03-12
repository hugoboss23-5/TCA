"""
TCA Layer 5: Topological Health Metrics

Computed on the ACTIVE reasoning subgraph (not the whole graph):
  - betweenness_centrality: bottleneck nodes (fragile reasoning)
  - clustering_coefficient: tight connections (echo chamber risk)
  - path_diversity: multiple paths to conclusion (robust reasoning)
  - cycle_detection: loops (circular reasoning)
  - bridge_detection: new cross-cluster connections (insight)
  - isolation_detection: unreachable nodes (dead ends)

All metrics are computed from topology only — no embeddings,
no learned parameters.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from tca.L2_graph.topo_node import TopologicalNode


@dataclass
class HealthReport:
    """Topological health metrics for a reasoning subgraph."""
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
    """Build adjacency set from subgraph (directed edges)."""
    adj: dict[str, set[str]] = {nid: set() for nid in subgraph}
    for nid, node in subgraph.items():
        for target_id in node.edges:
            if target_id in subgraph:
                adj[nid].add(target_id)
    return adj


def _get_undirected_adjacency(subgraph: dict[str, TopologicalNode]
                              ) -> dict[str, set[str]]:
    """Build undirected adjacency from subgraph."""
    adj: dict[str, set[str]] = {nid: set() for nid in subgraph}
    for nid, node in subgraph.items():
        for target_id in node.edges:
            if target_id in subgraph:
                adj[nid].add(target_id)
                adj[target_id].add(nid)
    return adj


def betweenness_centrality(subgraph: dict[str, TopologicalNode]
                           ) -> dict[str, float]:
    """Compute betweenness centrality for each node.

    Uses Brandes' algorithm adapted for small graphs.
    High centrality = bottleneck node (fragile reasoning).
    """
    nodes = list(subgraph.keys())
    adj = _get_adjacency(subgraph)
    centrality: dict[str, float] = {n: 0.0 for n in nodes}

    for s in nodes:
        # BFS from s.
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

    # Normalize.
    n = len(nodes)
    if n > 2:
        factor = 1.0 / ((n - 1) * (n - 2))
        centrality = {k: v * factor for k, v in centrality.items()}

    return centrality


def clustering_coefficient(subgraph: dict[str, TopologicalNode]
                           ) -> dict[str, float]:
    """Compute clustering coefficient for each node.

    High clustering = tightly connected neighborhood (echo chamber risk).
    """
    adj = _get_undirected_adjacency(subgraph)
    cc: dict[str, float] = {}

    for nid in subgraph:
        neighbors = adj[nid]
        k = len(neighbors)
        if k < 2:
            cc[nid] = 0.0
            continue
        # Count edges between neighbors.
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
    """Count distinct paths from source to target.

    Returns normalized diversity: paths_found / max_paths.
    Multiple paths = robust reasoning.
    """
    if source not in subgraph or target not in subgraph:
        return 0.0

    adj = _get_adjacency(subgraph)
    paths_found = 0

    # DFS to find paths (with visited set per path).
    stack: list[tuple[str, list[str]]] = [(source, [source])]
    while stack and paths_found < max_paths:
        current, path = stack.pop()
        if current == target and len(path) > 1:
            paths_found += 1
            continue
        for nb in adj.get(current, set()):
            if nb not in path:  # Avoid revisiting in same path.
                stack.append((nb, path + [nb]))

    return min(1.0, paths_found / max_paths)


def cycle_detection(subgraph: dict[str, TopologicalNode]
                    ) -> list[list[str]]:
    """Detect cycles in the subgraph (circular reasoning).

    Returns list of cycles found (each cycle is a list of node IDs).
    """
    adj = _get_adjacency(subgraph)
    cycles: list[list[str]] = []
    visited: set[str] = set()

    for start in subgraph:
        if start in visited:
            continue
        # DFS looking for back edges.
        stack: list[tuple[str, list[str], set[str]]] = [
            (start, [start], {start})
        ]
        while stack:
            current, path, path_set = stack.pop()
            visited.add(current)
            for nb in adj.get(current, set()):
                if nb in path_set:
                    # Found cycle: extract it.
                    cycle_start = path.index(nb)
                    cycle = path[cycle_start:] + [nb]
                    # Normalize: start from lexically smallest.
                    min_idx = cycle[:-1].index(min(cycle[:-1]))
                    normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                    if normalized not in cycles:
                        cycles.append(normalized)
                elif nb not in visited:
                    stack.append((nb, path + [nb], path_set | {nb}))

    return cycles


def bridge_detection(subgraph: dict[str, TopologicalNode]
                     ) -> list[tuple[str, str]]:
    """Detect bridge edges (connections between otherwise separate clusters).

    A bridge is an edge whose removal disconnects the graph.
    Bridge formation = insight (new connection between concepts).
    """
    adj = _get_undirected_adjacency(subgraph)
    nodes = list(subgraph.keys())
    bridges: list[tuple[str, str]] = []

    if len(nodes) < 2:
        return bridges

    # For each edge, check if removing it disconnects the graph.
    edges_checked: set[tuple[str, str]] = set()
    for nid in nodes:
        for nb in adj[nid]:
            edge = tuple(sorted([nid, nb]))
            if edge in edges_checked:
                continue
            edges_checked.add(edge)

            # Temporarily remove edge and check connectivity.
            adj[nid].discard(nb)
            adj[nb].discard(nid)

            # BFS from nid.
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

            # Restore edge.
            adj[nid].add(nb)
            adj[nb].add(nid)

    return bridges


def isolation_detection(subgraph: dict[str, TopologicalNode],
                        conclusion_id: str | None = None
                        ) -> list[str]:
    """Detect isolated nodes with no path to the conclusion.

    If no conclusion_id given, uses the node with highest activation.
    """
    if not subgraph:
        return []

    adj = _get_undirected_adjacency(subgraph)

    # Determine conclusion node.
    if conclusion_id is None or conclusion_id not in subgraph:
        # Use highest-activation node.
        conclusion_id = max(subgraph,
                           key=lambda n: subgraph[n].activation)

    # BFS from conclusion.
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
    """Compute full health report for a reasoning subgraph."""
    report = HealthReport()
    report.betweenness = betweenness_centrality(subgraph)
    report.clustering = clustering_coefficient(subgraph)
    report.cycles = cycle_detection(subgraph)
    report.bridges = bridge_detection(subgraph)
    report.isolated = isolation_detection(subgraph)

    if source and target:
        report.path_diversity = path_diversity(subgraph, source, target)

    return report
