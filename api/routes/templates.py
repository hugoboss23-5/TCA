"""TCA Calculator — Template routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import engine

router = APIRouter(tags=["templates"])


@router.get("/templates")
def list_templates():
    return {"templates": engine.list_templates()}


@router.get("/templates/{name}")
def get_template(name: str):
    tpl = engine.load_template(name)
    if tpl is None:
        raise HTTPException(404, f"Template '{name}' not found")
    return tpl
