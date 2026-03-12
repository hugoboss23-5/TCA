"""TCA Calculator — Pydantic request/response models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EdgeTypeEnum(str, Enum):
    MIRRORS = "MIRRORS"
    INHERITS = "INHERITS"
    BOUNDS = "BOUNDS"
    EXPRESSES = "EXPRESSES"
    VERIFIES = "VERIFIES"
    REMOVES = "REMOVES"
    SEEKS = "SEEKS"


# ── Request models ──────────────────────────────────────────────

class CreateGraphRequest(BaseModel):
    name: str = "Untitled"
    template: str | None = None  # template name to load


class AddNodeRequest(BaseModel):
    label: str
    node_id: str | None = None


class AddEdgeRequest(BaseModel):
    source_id: str
    target_id: str
    edge_type: EdgeTypeEnum
    weight: float = 1.0


class ApplySolutionRequest(BaseModel):
    solution_index: int


# ── Response models ─────────────────────────────────────────────

class EdgeResponse(BaseModel):
    target_id: str
    edge_type: str
    weight: float
    grounded: bool


class NodeResponse(BaseModel):
    id: str
    label: str
    edges: list[EdgeResponse]


class GraphStateResponse(BaseModel):
    graph_id: str
    name: str
    nodes: list[NodeResponse]
    node_count: int
    edge_count: int


class GraphListItem(BaseModel):
    graph_id: str
    name: str
    node_count: int
    edge_count: int


class HealthResponse(BaseModel):
    betweenness: dict[str, float]
    clustering: dict[str, float]
    path_diversity: float
    cycles: list[list[str]]
    bridges: list[list[str]]
    isolated: list[str]


class ConfidenceResponse(BaseModel):
    confidence: float
    path_diversity_score: float
    cycle_penalty: float
    grounding_ratio: float
    convergence_speed: float


class SolutionResponse(BaseModel):
    index: int
    problem_type: str
    problem_description: str
    action: str
    details: dict[str, Any]
    reasoning: str
    confidence: float


class AnalysisResponse(BaseModel):
    node_count: int
    edge_count: int
    health: HealthResponse
    confidence: ConfidenceResponse
    solutions: list[SolutionResponse]
    solution_count: int
