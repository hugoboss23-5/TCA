"""
TCA Layer 3: Grounding Actions (Software-Only V1)

Concrete actions that ground predictions in reality:
  - file_exists: predict whether a file exists, check, update
  - code_runs: predict whether code succeeds, execute, update
  - search_returns: predict search relevance, search, update
  - tool_produces: predict output shape, run, update

Each action returns a PredictionResult with prediction, actual,
delta, and edges_updated.
"""

from __future__ import annotations

import os
import subprocess
import sys

from tca.L3_grounding.prediction_loop import GroundingLoop, PredictionResult


def file_exists(loop: GroundingLoop,
                source_id: str, target_id: str,
                path: str) -> PredictionResult:
    """Predict whether a file exists, then check.

    Args:
        loop: The grounding loop to update.
        source_id: Node making the prediction.
        target_id: Node representing the file/resource.
        path: File path to check.

    Returns:
        PredictionResult with prediction, actual, delta.
    """
    prediction = loop.predict(source_id, "file_exists")
    actual = 1.0 if os.path.exists(path) else 0.0
    return loop.ground(source_id, target_id, "file_exists",
                       prediction=prediction, actual=actual)


def code_runs(loop: GroundingLoop,
              source_id: str, target_id: str,
              code_string: str,
              timeout: float = 5.0) -> PredictionResult:
    """Predict whether code runs successfully, then execute.

    Args:
        loop: The grounding loop to update.
        source_id: Node making the prediction.
        target_id: Node representing the code/operation.
        code_string: Python code to execute.
        timeout: Max seconds to wait.

    Returns:
        PredictionResult with prediction, actual (1.0=success, 0.0=error), delta.
    """
    prediction = loop.predict(source_id, "code_runs")
    try:
        result = subprocess.run(
            [sys.executable, "-c", code_string],
            capture_output=True, text=True,
            timeout=timeout,
        )
        actual = 1.0 if result.returncode == 0 else 0.0
    except (subprocess.TimeoutExpired, OSError):
        actual = 0.0

    return loop.ground(source_id, target_id, "code_runs",
                       prediction=prediction, actual=actual)


def search_returns(loop: GroundingLoop,
                   source_id: str, target_id: str,
                   query: str,
                   corpus: list[str]) -> PredictionResult:
    """Predict whether a search returns relevant results.

    Args:
        loop: The grounding loop to update.
        source_id: Node making the prediction.
        target_id: Node representing the search target.
        query: Search query.
        corpus: List of strings to search through.

    Returns:
        PredictionResult. actual = fraction of corpus matching query keywords.
    """
    prediction = loop.predict(source_id, "search_returns")
    keywords = query.lower().split()
    if not corpus:
        actual = 0.0
    else:
        matches = sum(
            1 for item in corpus
            if any(kw in item.lower() for kw in keywords)
        )
        actual = matches / len(corpus)

    return loop.ground(source_id, target_id, "search_returns",
                       prediction=prediction, actual=actual)


def tool_produces(loop: GroundingLoop,
                  source_id: str, target_id: str,
                  tool_fn: callable,
                  tool_input: object,
                  expected_type: type | None = None) -> PredictionResult:
    """Predict whether a tool produces expected output type.

    Args:
        loop: The grounding loop to update.
        source_id: Node making the prediction.
        target_id: Node representing the tool.
        tool_fn: Callable to execute.
        tool_input: Input to pass to the tool.
        expected_type: Expected return type (if None, any non-None = success).

    Returns:
        PredictionResult. actual = 1.0 if output matches expected.
    """
    prediction = loop.predict(source_id, "tool_produces")
    try:
        output = tool_fn(tool_input)
        if expected_type is not None:
            actual = 1.0 if isinstance(output, expected_type) else 0.0
        else:
            actual = 1.0 if output is not None else 0.0
    except Exception:
        actual = 0.0

    return loop.ground(source_id, target_id, "tool_produces",
                       prediction=prediction, actual=actual)
