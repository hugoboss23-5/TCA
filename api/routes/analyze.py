"""TCA Calculator — Analysis routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import engine

router = APIRouter(tags=["analyze"])


@router.post("/graphs/{graph_id}/analyze")
def analyze_graph(graph_id: str):
    result = engine.run_analysis(graph_id)
    if result is None:
        raise HTTPException(404, "Graph not found")
    return result


@router.post("/graphs/{graph_id}/solutions/{index}/apply")
def apply_solution(graph_id: str, index: int):
    ok = engine.apply_solution(graph_id, index)
    if not ok:
        raise HTTPException(400, "Solution could not be applied")
    # Re-analyze after applying.
    result = engine.run_analysis(graph_id)
    state = engine.serialize_graph_state(graph_id)
    return {"graph": state, "analysis": result}
