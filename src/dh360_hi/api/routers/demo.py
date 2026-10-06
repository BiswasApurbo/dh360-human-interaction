"""Demonstration web page endpoint for Human Interaction authorization workflow."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="", tags=["demo"])

_DEMO_HTML_PATH = Path(__file__).resolve().parent.parent / "static" / "demo.html"


@router.get("/demo", response_class=HTMLResponse)
def get_demo_page() -> HTMLResponse:
    """Serve the static Human Interaction authorization workflow demo page."""
    html_content = _DEMO_HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)
