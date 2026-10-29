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


def test_knowledge_items_crud_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Full CRUD lifecycle on knowledge items verifies persistence and API responses."""
    csv_file = tmp_path / "test_knowledge.csv"
    csv_file.write_text(
        "Question (English),Answer (English),প্রশ্ন (বাংলা),উত্তর (বাংলা),Institution Type\n"
        "Initial Q,Initial A,প্রারম্ভিক প্রশ্ন,প্রারম্ভিক উত্তর,university\n",
        encoding="utf-8-sig",
    )
    token = _configure_operator(monkeypatch, "knowledge:manage")
    ks = KnowledgeService(csv_path=csv_file)
    app = create_app(knowledge_service=ks)

    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {token}"}

        # 1. List items
        resp = client.get("/knowledge/items", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == "kb-1"
        assert data["items"][0]["question_en"] == "Initial Q"

        # 2. Create item
        new_payload = {
            "question_en": "Where is the pharmacy?",
            "answer_en": "Ground floor next to emergency.",
            "question_bn": "ফার্মেসি কোথায়?",
            "answer_bn": "নিচতলায় জরুরি বিভাগের পাশে।",
            "institution_type": "clinic",
        }
        create_resp = client.post("/knowledge/items", json=new_payload, headers=headers)
        assert create_resp.status_code == 201
        created = create_resp.json()
        assert created["id"] == "kb-2"
        assert created["question_en"] == "Where is the pharmacy?"
        assert created["institution_type"] == "clinic"

        # 3. Retrieve item by ID
        get_resp = client.get("/knowledge/items/kb-2", headers=headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["answer_bn"] == "নিচতলায় জরুরি বিভাগের পাশে।"

        # 4. Search items
        search_resp = client.get("/knowledge/items?query=emergency", headers=headers)
        assert search_resp.status_code == 200
        assert search_resp.json()["total"] == 1
        assert search_resp.json()["items"][0]["id"] == "kb-2"

        # 5. Update item
        update_payload = {
            "answer_en": "Ground floor, open 24/7.",
        }
        update_resp = client.put("/knowledge/items/kb-2", json=update_payload, headers=headers)
        assert update_resp.status_code == 200
        assert update_resp.json()["answer_en"] == "Ground floor, open 24/7."
        assert update_resp.json()["question_en"] == "Where is the pharmacy?"

        # 6. Delete item
        del_resp = client.delete("/knowledge/items/kb-2", headers=headers)
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"

        # 7. Verify deletion
        list_after = client.get("/knowledge/items", headers=headers)
        assert list_after.json()["total"] == 1
        assert list_after.json()["items"][0]["id"] == "kb-1"

        # 8. Check backup file was created
        backup_file = csv_file.with_suffix(".csv.bak")
        assert backup_file.exists()


def test_knowledge_create_validation_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Empty question and answer pairs are rejected with HTTP 400."""
    csv_file = tmp_path / "test_knowledge.csv"
    csv_file.write_text(
        "Question (English),Answer (English),প্রশ্ন (বাংলা),উত্তর (বাংলা),Institution Type\n",
        encoding="utf-8-sig",
    )
    token = _configure_operator(monkeypatch, "knowledge:manage")
    ks = KnowledgeService(csv_path=csv_file)
    app = create_app(knowledge_service=ks)

    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {token}"}
        resp = client.post(
            "/knowledge/items",
            json={"question_en": "", "answer_en": "", "question_bn": "", "answer_bn": ""},
            headers=headers,
        )
        assert resp.status_code == 400
        assert "at least one complete Question and Answer" in resp.json()["detail"]
