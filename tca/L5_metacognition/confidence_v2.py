"""
TCA Layer 5: Confidence V2 — Topology + Coherence

Confidence v1 uses static topology metrics only (path diversity, cycles,
grounding ratio, convergence speed). This is necessary but insufficient:
a system can converge on a wrong answer with healthy-looking topology.

Confidence v2 combines static topology health with dynamic phase coherence.
The critical case is "premature" — static says confident, dynamic says
incoherent. This is where current AI hallucinates. TCA catches it because
coherence overrides convergence.

No new learned parameters. Coherence is computed from activation histories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tca.L4_temporal.coherence import CoherenceDetector
from tca.L5_metacognition.confidence import ConfidenceReport


@dataclass
class ConfidenceV2Report:
    """Confidence breakdown with dynamic coherence."""
    confidence: float
    state: str  # "understanding" | "premature" | "forming" | "confused"
    static_confidence: float
    dynamic_coherence: float
    flags: list[str] = field(default_factory=list)


def compute_confidence_v2(
    health_metrics: ConfidenceReport,
    coherence_detector: CoherenceDetector,
    active_nodes: list[str],
    tick_window: int = 10,
) -> ConfidenceV2Report:
    """Compute confidence from topology + coherence.

    Static confidence comes from existing health metrics.
    Dynamic coherence comes from phase alignment of activation oscillations.
    The state classification determines how they combine:

    - "understanding": confidence = max(static, dynamic)
    - "premature": confidence = dynamic * 0.5  (OVERRIDE static)
    - "forming": confidence = dynamic * 0.8, flag to continue
    - "confused": confidence = 0.0, flag to reroute

    The premature case is the key insight — static says confident,
    dynamic says incoherent. This catches hallucination-equivalent failures.
    """
    static = health_metrics.confidence
    dynamic = coherence_detector.compute_phase_coherence(
        active_nodes, tick_window)
    state = coherence_detector.coherence_vs_convergence(dynamic, static)

    if state == "understanding":
        confidence = max(static, dynamic)
        flags: list[str] = []
    elif state == "premature":
        confidence = dynamic * 0.5
        flags = ["premature_convergence"]
    elif state == "forming":
        confidence = dynamic * 0.8
        flags = ["continue_reasoning"]
    else:  # confused
        confidence = 0.0
        flags = ["reroute"]

    return ConfidenceV2Report(
        confidence=confidence,
        state=state,
        static_confidence=static,
        dynamic_coherence=dynamic,
        flags=flags,
    )
