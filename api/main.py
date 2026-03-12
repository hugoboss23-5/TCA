"""TCA Calculator — FastAPI application.

Run with: uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import analyze, export, graph, templates

app = FastAPI(
    title="TCA Calculator",
    description="Topological Cognitive Architecture — structural analysis for any system",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes (prefix /api).
app.include_router(graph.router, prefix="/api")
app.include_router(analyze.router, prefix="/api")
app.include_router(templates.router, prefix="/api")
app.include_router(export.router, prefix="/api")

# Serve frontend — must be LAST (catch-all for static files).
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
