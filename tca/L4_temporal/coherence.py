"""
TCA Layer 4: Coherence Detector

Measures phase alignment of activation oscillations across nodes
in the active reasoning subgraph.

Coherence is NOT static graph health. It's dynamic:
- Do nodes oscillate in sync over time WITHOUT direct edges forcing it?
- Are activation patterns converging to a shared rhythm?
- Is the synchronization emergent (from topology) or forced (from direct connection)?

This is the primary optimization target of TCA.

Method: Pearson correlation of activation histories, with direct-edge
connections discounted (we care about EMERGENT sync, not FORCED sync).
"""

from __future__ import annotations

import math

from tca.L2_graph.operations import TopologicalGraph
from tca.L4_temporal.temporal_node import get_activation_history


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient between two equal-length series.

    Returns 0.0 if either series has zero variance (constant).
    """
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for i in range(n):
        dx = xs[i] - mean_x
        dy = ys[i] - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy

    denom = math.sqrt(den_x * den_y)
    if denom < 1e-12:
        return 0.0
    return num / denom


def _has_direct_edge(graph: TopologicalGraph,
                     id_a: str, id_b: str) -> bool:
    """Check if a direct edge exists between two nodes (either direction)."""
    node_a = graph.get_node(id_a)
    node_b = graph.get_node(id_b)
    if node_a is not None and id_b in node_a.edges:
        return True
    if node_b is not None and id_a in node_b.edges:
        return True
    return False


class CoherenceDetector:
    """Measures phase alignment of activation oscillations across nodes.

    Emergent synchronization = the system found something real.
    Forced synchronization = the system is just following a chain.
    No synchronization = noise.
    """

    def __init__(self, graph: TopologicalGraph) -> None:
        self.graph = graph

    def compute_phase_coherence(self, active_nodes: list[str],
                                tick_window: int = 10) -> float:
        """Compute mean pairwise Pearson correlation of activation histories.

        Direct-edge pairs are discounted by 0.3 (forced sync).
        Emergent sync (no direct edge) keeps full correlation value.

        Returns:
            Float in [0.0, 1.0]. 0.0 = pure noise, 1.0 = perfect sync.
        """
        if len(active_nodes) < 2:
            return 0.0

        correlations: list[float] = []

        for i in range(len(active_nodes)):
            node_i = self.graph.get_node(active_nodes[i])
            if node_i is None:
                continue
            hist_i = get_activation_history(node_i)
            if len(hist_i) < 2:
                continue
            window_i = hist_i[-tick_window:]

            for j in range(i + 1, len(active_nodes)):
                node_j = self.graph.get_node(active_nodes[j])
                if node_j is None:
                    continue
                hist_j = get_activation_history(node_j)
                if len(hist_j) < 2:
                    continue
                window_j = hist_j[-tick_window:]

                # Align to same length.
                min_len = min(len(window_i), len(window_j))
                if min_len < 2:
                    continue

                r = _pearson(window_i[-min_len:], window_j[-min_len:])

                # Discount forced sync (direct edge).
                if _has_direct_edge(self.graph,
                                    active_nodes[i], active_nodes[j]):
                    r *= 0.3

                correlations.append(r)

        if not correlations:
            return 0.0

        mean_r = sum(correlations) / len(correlations)
        return max(0.0, min(1.0, mean_r))

    def detect_phase_transition(self,
                                coherence_history: list[float]
                                ) -> tuple[bool, int]:
        """Detect rapid coherence jump — the 'insight moment'.

        A jump of > 0.2 within 3 consecutive ticks = phase transition.

        Returns:
            (detected: bool, tick_index: int). tick_index = -1 if not detected.
        """
        if len(coherence_history) < 3:
            return (False, -1)

        for i in range(2, len(coherence_history)):
            delta = coherence_history[i] - coherence_history[i - 2]
            if delta > 0.2:
                return (True, i)

        return (False, -1)

    def coherence_vs_convergence(self, coherence_score: float,
                                 convergence_score: float) -> str:
        """Classify the reasoning state from coherence and convergence.

        - High coherence + high convergence = genuine understanding
        - Low coherence + high convergence = premature crystallization (DANGEROUS)
        - High coherence + low convergence = insight forming, needs more ticks
        - Low coherence + low convergence = confusion, needs rerouting

        Threshold: 0.5 for both dimensions.
        """
        high_coherence = coherence_score >= 0.5
        high_convergence = convergence_score >= 0.5

        if high_coherence and high_convergence:
            return "understanding"
        if not high_coherence and high_convergence:
            return "premature"
        if high_coherence and not high_convergence:
            return "forming"
        return "confused"
