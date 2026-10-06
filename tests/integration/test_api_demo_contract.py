"""Integration tests for the demo UI endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app


def test_demo_endpoint_returns_html() -> None:
    """Verify GET /demo returns HTTP 200 with HTML content."""
    with TestClient(create_app()) as client:
        response = client.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Human Interaction Authorization Demo" in response.text
    assert "/authorization/sessions" in response.text
    assert "/interaction/events" in response.text
