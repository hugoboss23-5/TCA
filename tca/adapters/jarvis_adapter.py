"""
Adapter for Jarvis Chestohedron Router.

Jarvis is Hugo's 7-gate cognitive router. It is NOT available in this
repository or as an MCP server. This adapter provides the expected
interface so TCA L1 can check for Jarvis availability and fall back
to its own routing logic when Jarvis is absent.

When Jarvis becomes available, implement `route()` to call Jarvis's
actual routing API and return real gate weights.
"""

GATES = ("MIRROR", "INHERIT", "BOUND", "EXPRESS", "VERIFY", "REMOVE", "DARASH")

JARVIS_AVAILABLE = False


def route(input_text: str) -> dict[str, float] | None:
    """Route input through Jarvis's 7 gates.

    Returns:
        dict mapping gate names to float weights, or None if Jarvis
        is not available.
    """
    if not JARVIS_AVAILABLE:
        return None
    # When Jarvis is connected, this will call its routing API.
    # For now, signal unavailability so L1 uses its own logic.
    return None
