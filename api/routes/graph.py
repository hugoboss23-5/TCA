"""TCA Calculator — Graph CRUD routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import engine
from api.models import (
    AddEdgeRequest,
    AddNodeRequest,
    CreateGraphRequest,
    GraphListItem,
    GraphStateResponse,
)

router = APIRouter(tags=["graph"])


@router.post("/graphs")
def create_graph(req: CreateGraphRequest):
    graph_id = engine.create_graph(name=req.name, template_name=req.template)
    return {"graph_id": graph_id, "name": req.name or req.template or "Untitled"}


@router.get("/graphs", response_model=list[GraphListItem])
def list_graphs():
    return engine.list_graphs()


@router.get("/graphs/{graph_id}")
def get_graph(graph_id: str):
    state = engine.serialize_graph_state(graph_id)
    if state is None:
        raise HTTPException(404, "Graph not found")
    return state


@router.delete("/graphs/{graph_id}")
def delete_graph(graph_id: str):
    if not engine.delete_graph(graph_id):
        raise HTTPException(404, "Graph not found")
    return {"status": "deleted"}


@router.post("/graphs/{graph_id}/nodes")
def add_node(graph_id: str, req: AddNodeRequest):
    node = engine.add_node(graph_id, req.label, req.node_id)
    if node is None:
        raise HTTPException(404, "Graph not found")
    return {"node_id": node.id, "label": node.label}


@router.delete("/graphs/{graph_id}/nodes/{node_id}")
def delete_node(graph_id: str, node_id: str):
    if not engine.delete_node(graph_id, node_id):
        raise HTTPException(404, "Graph or node not found")
    return {"status": "deleted"}


@router.post("/graphs/{graph_id}/edges")
def add_edge(graph_id: str, req: AddEdgeRequest):
    ok = engine.add_edge(
        graph_id, req.source_id, req.target_id,
        req.edge_type.value, req.weight,
    )
    if not ok:
        raise HTTPException(400, "Failed to add edge (graph or source node not found)")
    return {"status": "added"}


@router.delete("/graphs/{graph_id}/edges")
def delete_edge(graph_id: str, source_id: str, target_id: str,
                edge_type: str | None = None):
    if not engine.delete_edge(graph_id, source_id, target_id, edge_type):
        raise HTTPException(404, "Edge not found")
    return {"status": "deleted"}
