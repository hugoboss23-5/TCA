"""TCA Calculator — Export routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import engine

router = APIRouter(tags=["export"])


@router.get("/graphs/{graph_id}/export/state")
def export_state(graph_id: str):
    result = engine.export_state(graph_id)
    if result is None:
        raise HTTPException(404, "Graph not found")
    return result


@router.get("/graphs/{graph_id}/export/boot")
def export_boot(graph_id: str):
    result = engine.export_boot(graph_id)
    if result is None:
        raise HTTPException(404, "Graph not found")
    return result
