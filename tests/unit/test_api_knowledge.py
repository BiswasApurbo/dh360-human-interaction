"""API contract and security tests for POST /knowledge/reload."""

from __future__ import annotations

import secrets
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.backend.knowledge_service import KnowledgeService


def _configure_operator(
    monkeypatch: pytest.MonkeyPatch,
    scopes: str = "knowledge:manage",
) -> str:
    """Configure ephemeral operator credentials for API authentication."""
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_SCOPES", scopes)
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)
    return token


def test_knowledge_reload_requires_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling /knowledge/reload without bearer credentials returns 401."""
    _configure_operator(monkeypatch, "knowledge:manage")
    mock_ks = mock.MagicMock(spec=KnowledgeService)
    app = create_app(knowledge_service=mock_ks)

    with TestClient(app) as client:
        resp = client.post("/knowledge/reload")
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "Bearer"
        assert resp.json()["code"] == "UNAUTHENTICATED"
        mock_ks.reload.assert_not_called()


def test_knowledge_reload_requires_valid_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling /knowledge/reload with invalid token returns 401."""
    _configure_operator(monkeypatch, "knowledge:manage")
    mock_ks = mock.MagicMock(spec=KnowledgeService)
    app = create_app(knowledge_service=mock_ks)

    with TestClient(app) as client:
        resp = client.post(
            "/knowledge/reload",
            headers={"Authorization": "Bearer bad-token"},
        )
        assert resp.status_code == 401
        assert resp.json()["code"] == "UNAUTHENTICATED"
        mock_ks.reload.assert_not_called()


def test_knowledge_reload_requires_knowledge_manage_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling /knowledge/reload without knowledge:manage scope returns 403."""
    token = _configure_operator(monkeypatch, "persons:read,dialogue:interact")
    mock_ks = mock.MagicMock(spec=KnowledgeService)
    app = create_app(knowledge_service=mock_ks)

    with TestClient(app) as client:
        resp = client.post(
            "/knowledge/reload",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403
        assert resp.json()["code"] == "INSUFFICIENT_SCOPE"
        mock_ks.reload.assert_not_called()


def test_knowledge_reload_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling /knowledge/reload with valid scope triggers reload and returns counts."""
    token = _configure_operator(monkeypatch, "knowledge:manage")
    mock_ks = mock.MagicMock(spec=KnowledgeService)
    mock_ks.reload.return_value = 42
    app = create_app(knowledge_service=mock_ks)

    with TestClient(app) as client:
        resp = client.post(
            "/knowledge/reload",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reloaded"
        assert data["facts_count"] == 42
        assert isinstance(data["timestamp"], float)
        mock_ks.reload.assert_called_once()
