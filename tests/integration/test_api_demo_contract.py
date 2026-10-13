"""Integration tests for the demo UI endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from setuptools.config.pyprojecttoml import apply_configuration
from setuptools.dist import Distribution

from dh360_hi.api.app import create_app
from dh360_hi.api.routers import demo as demo_router


def test_demo_endpoint_returns_html() -> None:
    """Verify GET /demo returns HTTP 200 with HTML content."""
    with TestClient(create_app()) as client:
        response = client.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Human Interaction Authorization Demo" in response.text
    assert "/authorization/sessions" in response.text
    assert "/interaction/events" in response.text


def test_demo_static_asset_is_packaged_and_servable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify setuptools package-data includes demo.html and GET /demo serves the packaged copy."""
    repo_root = Path(__file__).resolve().parents[2]
    site_packages = tmp_path / "site-packages"

    dist = apply_configuration(
        Distribution({"script_name": "setup.py"}),
        repo_root / "pyproject.toml",
    )
    build_py = dist.get_command_obj("build_py")
    build_py.build_lib = str(site_packages)  # type: ignore[attr-defined]
    build_py.ensure_finalized()
    build_py.run()

    packaged_demo_py = site_packages / "dh360_hi" / "api" / "routers" / "demo.py"
    packaged_demo_html = packaged_demo_py.parent.parent / "static" / "demo.html"

    assert packaged_demo_py.is_file()
    assert packaged_demo_html.is_file()

    monkeypatch.setattr(demo_router, "_DEMO_HTML_PATH", packaged_demo_html)

    with TestClient(create_app()) as client:
        response = client.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Human Interaction Authorization Demo" in response.text
