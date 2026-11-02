"""Unit tests verifying multi-tenant data ownership and isolation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.backend.models import PersonStatus
from dh360_hi.backend.person_service import PersonService
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.contracts.dto import CreatePersonCommand


def _configure_operator(monkeypatch: pytest.MonkeyPatch, scopes: str) -> str:
    token = "test-operator-token"
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_SCOPES", scopes)
    return token


@pytest.fixture
def memory_db_manager() -> DatabaseManager:
    return DatabaseManager(":memory:", auto_init=True)


def test_schema_v5_tenant_columns(memory_db_manager: DatabaseManager) -> None:
    """Verify schema v5 includes tenant_id and robot_id columns on persons table."""
    assert memory_db_manager.get_schema_version() == 5
    with memory_db_manager.readonly_cursor() as cur:
        cur.execute("PRAGMA table_info(persons);")
        columns = [row[1] for row in cur.fetchall()]
        assert "tenant_id" in columns
        assert "robot_id" in columns


def test_person_service_tenant_isolation(memory_db_manager: DatabaseManager) -> None:
    """Verify PersonService filters people strictly by tenant_id."""
    repo = SQLitePersonRepository(memory_db_manager)
    service = PersonService(repo)

    # Create person for Tenant A
    p_a = service.create_person(
        display_name="User Admin A",
        tenant_id="tenant-a",
        robot_id="robot-1",
    )
    # Create person for Tenant B
    p_b = service.create_person(
        display_name="User Admin B",
        tenant_id="tenant-b",
        robot_id="robot-2",
    )
    # Create legacy person (unassigned tenant_id=None)
    p_legacy = service.create_person(
        display_name="Legacy User",
    )

    # Tenant A list
    list_a = service.list_people(tenant_id="tenant-a")
    names_a = [p.display_name for p in list_a]
    assert "User Admin A" in names_a
    assert "Legacy User" in names_a
    assert "User Admin B" not in names_a

    # Tenant B list
    list_b = service.list_people(tenant_id="tenant-b")
    names_b = [p.display_name for p in list_b]
    assert "User Admin B" in names_b
    assert "Legacy User" in names_b
    assert "User Admin A" not in names_b


def test_api_person_cross_tenant_access_prevention(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify FastAPI persons router blocks cross-tenant retrieval, update, and deletion."""
    token = _configure_operator(
        monkeypatch,
        "persons:read,persons:create,persons:update,persons:delete,biometrics:reset",
    )
    headers_a = {"Authorization": f"Bearer {token}", "X-Tenant-ID": "admin-a"}
    headers_b = {"Authorization": f"Bearer {token}", "X-Tenant-ID": "admin-b"}

    app = create_app()
    with TestClient(app) as client:
        # Admin A creates a person
        res_create = client.post(
            "/persons",
            headers=headers_a,
            json={"display_name": "Private Person A", "preferred_name": "PrivA"},
        )
        assert res_create.status_code == 201
        person_a_id = res_create.json()["person_id"]

        # Admin A can retrieve their person
        res_get_a = client.get(f"/persons/{person_a_id}", headers=headers_a)
        assert res_get_a.status_code == 200
        assert res_get_a.json()["display_name"] == "Private Person A"

        # Admin B CANNOT retrieve Admin A's person -> returns 404 (EntityNotFoundError)
        res_get_b = client.get(f"/persons/{person_a_id}", headers=headers_b)
        assert res_get_b.status_code == 404

        # Admin B CANNOT update Admin A's person -> returns 404
        res_patch_b = client.patch(
            f"/persons/{person_a_id}",
            headers=headers_b,
            json={"display_name": "Hacked Display Name"},
        )
        assert res_patch_b.status_code == 404

        # Admin B CANNOT reset biometrics of Admin A's person -> returns 404
        res_reset_b = client.delete(f"/persons/{person_a_id}/biometrics", headers=headers_b)
        assert res_reset_b.status_code == 404

        # Admin B CANNOT delete Admin A's person -> returns 404
        res_del_b = client.delete(f"/persons/{person_a_id}", headers=headers_b)
        assert res_del_b.status_code == 404

        # Admin A lists persons: Admin A sees their person
        res_list_a = client.get("/persons", headers=headers_a)
        ids_a = [p["person_id"] for p in res_list_a.json()]
        assert person_a_id in ids_a

        # Admin B lists persons: Admin B does NOT see Admin A's person
        res_list_b = client.get("/persons", headers=headers_b)
        ids_b = [p["person_id"] for p in res_list_b.json()]
        assert person_a_id not in ids_b


def test_api_knowledge_cross_tenant_access_prevention(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """Verify FastAPI knowledge router blocks cross-tenant access and mutation."""
    csv_file = tmp_path / "tenant_knowledge.csv"
    csv_file.write_text(
        "id,Question (English),Answer (English),প্রশ্ন (বাংলা),উত্তর (বাংলা),Institution Type,tenant_id\n"
        "kb-shared,Public Q,Public A,পাবলিক প্রশ্ন,পাবলিক উত্তর,general,\n",
        encoding="utf-8-sig",
    )
    token = _configure_operator(monkeypatch, "knowledge:manage")
    ks = KnowledgeService(csv_path=csv_file)
    app = create_app(knowledge_service=ks)

    headers_a = {"Authorization": f"Bearer {token}", "X-Tenant-ID": "admin-a"}
    headers_b = {"Authorization": f"Bearer {token}", "X-Tenant-ID": "admin-b"}

    with TestClient(app) as client:
        # Admin A creates a knowledge item
        res_create = client.post(
            "/knowledge/items",
            headers=headers_a,
            json={
                "question_en": "Tenant A Secret Q",
                "answer_en": "Tenant A Secret A",
            },
        )
        assert res_create.status_code == 201
        item_a_id = res_create.json()["id"]

        # Admin A can read their item
        res_get_a = client.get(f"/knowledge/items/{item_a_id}", headers=headers_a)
        assert res_get_a.status_code == 200

        # Admin B CANNOT read Admin A's item -> returns 404
        res_get_b = client.get(f"/knowledge/items/{item_a_id}", headers=headers_b)
        assert res_get_b.status_code == 404

        # Admin B CANNOT update Admin A's item -> returns 404
        res_put_b = client.put(
            f"/knowledge/items/{item_a_id}",
            headers=headers_b,
            json={"question_en": "Overwritten by B", "answer_en": "Overwritten"},
        )
        assert res_put_b.status_code == 404

        # Admin B CANNOT delete Admin A's item -> returns 404
        res_del_b = client.delete(f"/knowledge/items/{item_a_id}", headers=headers_b)
        assert res_del_b.status_code == 404

        # In list: Admin A sees item_a_id; Admin B does NOT see item_a_id
        list_a = client.get("/knowledge/items", headers=headers_a).json()["items"]
        assert any(item["id"] == item_a_id for item in list_a)

        list_b = client.get("/knowledge/items", headers=headers_b).json()["items"]
        assert not any(item["id"] == item_a_id for item in list_b)
