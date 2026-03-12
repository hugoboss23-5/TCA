"""
TCA Layer 1: Chestohedron Router

Routes incoming queries through 7 cognitive gates simultaneously.
All gates activate for every input — weights reflect relevance,
not binary on/off. Weights are normalized to sum to 1.0.

Gate semantics (Chestohedron doctrine):
  MIRROR  — reflection, analogy, "what is this like?"
  INHERIT — derivation, lineage, "where does this come from?"
  BOUND   — constraint, limitation, "what are the rules?"
  EXPRESS — output, manifestation, "how does this show up?"
  VERIFY  — evidence, validation, "is this true?"
  REMOVE  — negation, contradiction, "what's wrong here?"
  DARASH  — seeking, exploration, "what don't we know?"
"""

from __future__ import annotations

import re

from tca.adapters.jarvis_adapter import route as jarvis_route

GATES = ("MIRROR", "INHERIT", "BOUND", "EXPRESS", "VERIFY", "REMOVE", "DARASH")

# Keyword signals that boost specific gates.
# Each gate has a set of trigger patterns (case-insensitive).
_GATE_SIGNALS: dict[str, list[str]] = {
    "MIRROR": [
        r"\blike\b", r"\bsimilar\b", r"\banalog", r"\bcompare",
        r"\bmetaphor", r"\bresembl", r"\bremind", r"\bparallel",
        r"\bwhat\s+is\b",
    ],
    "INHERIT": [
        r"\bfrom\b", r"\bderiv", r"\borigin", r"\bhistor",
        r"\bcause", r"\bwhy\b", r"\bsource", r"\broot\b",
        r"\bbecause\b", r"\bfound(ed|ation)\b",
    ],
    "BOUND": [
        r"\brule", r"\blimit", r"\bconstraint", r"\bcannot\b",
        r"\bshould\s+not\b", r"\bmust\b", r"\bboundar",
        r"\brestrict", r"\brequire",
    ],
    "EXPRESS": [
        r"\bshow\b", r"\bexplain\b", r"\bdescribe\b", r"\btell\b",
        r"\bdefine\b", r"\bexample\b", r"\bdemonstrat",
        r"\billustrat", r"\bmanifest",
    ],
    "VERIFY": [
        r"\btrue\b", r"\bcorrect", r"\bvalid", r"\bprove\b",
        r"\bevidence\b", r"\bconfirm", r"\bcheck\b", r"\btest\b",
        r"\bverif", r"\bis\s+this\s+right",
    ],
    "REMOVE": [
        r"\bwrong\b", r"\bfalse\b", r"\bnot\b", r"\bcontradict",
        r"\brefut", r"\bdisprove", r"\binvalid", r"\berror",
        r"\bmistake", r"\bflaw",
    ],
    "DARASH": [
        r"\bexplor", r"\bwonder", r"\bwhat\s+if\b", r"\bcould\b",
        r"\bpossib", r"\bimagin", r"\bspeculat", r"\bhypothes",
        r"\bunknown", r"\bopen\s+question", r"\bhow\s+does\b",
        r"\bhow\s+do\b",
    ],
}

# Baseline activation — every gate gets at least this much.
_BASELINE = 0.05


def _score_gate(gate: str, text: str) -> float:
    """Count how many signal patterns match for a gate."""
    patterns = _GATE_SIGNALS[gate]
    score = 0.0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        score += len(matches)
    return score


def route(input_text: str) -> dict[str, float]:
    """Route input through all 7 gates simultaneously.

    Tries Jarvis first. If unavailable, uses keyword-based routing.

    Args:
        input_text: Raw query or input string.

    Returns:
        Dict mapping each gate name to a float weight.
        All 7 gates have nonzero values. Weights sum to 1.0.
    """
    # Try Jarvis first.
    jarvis_result = jarvis_route(input_text)
    if jarvis_result is not None:
        return jarvis_result

    # Compute raw scores from keyword signals.
    raw: dict[str, float] = {}
    for gate in GATES:
        raw[gate] = _BASELINE + _score_gate(gate, input_text)

    # Normalize to sum to 1.0.
    total = sum(raw.values())
    return {gate: weight / total for gate, weight in raw.items()}
