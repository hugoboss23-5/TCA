"""
TCA Layer 4: Internal Clock

Reasoning unfolds over internal time. Each tick represents one
round of spreading activation through the Layer 2 graph.

The clock tracks:
  - tick count
  - convergence (are activations stabilizing?)
  - early stopping when convergence detected
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConvergenceState:
    """State of convergence checking."""
    converged: bool = False
    diverging: bool = False
    activation_delta: float = float("inf")
    failure_signal: str = ""


class InternalClock:
    """Internal reasoning clock, independent of wall time.

    Each tick = one round of spreading activation.
    Tracks convergence to enable early stopping.
    """

    def __init__(self, max_ticks: int = 100,
                 convergence_threshold: float = 0.01,
                 divergence_threshold: float = 0.5) -> None:
        self.max_ticks = max_ticks
        self.convergence_threshold = convergence_threshold
        self.divergence_threshold = divergence_threshold

        self._tick: int = 0
        self._prev_activations: dict[str, float] = {}
        self._history: list[float] = []  # activation deltas per tick

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def reset(self) -> None:
        """Reset clock for a new reasoning cycle."""
        self._tick = 0
        self._prev_activations = {}
        self._history = []

    def advance(self, current_activations: dict[str, float]
                ) -> ConvergenceState:
        """Advance one tick and check convergence.

        Args:
            current_activations: Dict of node_id -> activation level
                                 after this tick's propagation.

        Returns:
            ConvergenceState indicating whether reasoning has
            converged, is diverging, or should continue.
        """
        self._tick += 1

        # Compute total activation change from previous tick.
        if self._prev_activations:
            all_ids = set(current_activations) | set(self._prev_activations)
            total_delta = sum(
                abs(current_activations.get(nid, 0.0)
                    - self._prev_activations.get(nid, 0.0))
                for nid in all_ids
            )
            # Normalize by number of nodes to make threshold scale-independent.
            if all_ids:
                avg_delta = total_delta / len(all_ids)
            else:
                avg_delta = 0.0
        else:
            avg_delta = float("inf")  # First tick, no comparison.

        self._history.append(avg_delta)
        self._prev_activations = dict(current_activations)

        state = ConvergenceState(activation_delta=avg_delta)

        # Check convergence.
        if avg_delta < self.convergence_threshold:
            state.converged = True

        # Check divergence (activation delta growing).
        if len(self._history) >= 3:
            recent = self._history[-3:]
            if all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
                if avg_delta > self.divergence_threshold:
                    state.diverging = True
                    state.failure_signal = (
                        f"Activations diverging: deltas {recent}")

        return state

    def has_budget(self) -> bool:
        """Whether there are ticks remaining."""
        return self._tick < self.max_ticks
