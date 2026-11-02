"""Tests for Phase 1 Emergency Security Remediation.

Verifies:
1. /sessions endpoints require INTERACTION_READ scope and reject unauthenticated requests.
2. /interaction/context and /interaction/summary/latest reject unauthenticated requests.
3. Requests with insufficient scope are rejected with HTTP 403 INSUFFICIENT_SCOPE.
4. Requests with valid INTERACTION_READ scope succeed with HTTP 200.
5. DatabaseManager enforces strict 0600 file permissions on database and auxiliary files.
"""

from __future__ import annotations

import secrets
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.backend.database import DatabaseManager


def _configure_operator(monkeypatch: pytest.MonkeyPatch, scopes: str) -> str:
    """Configure ephemeral operator credentials for test."""
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_SCOPES", scopes)
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)
    return token


@pytest.mark.parametrize(
    "path",
    [
        "/sessions",
        "/sessions/count",
        "/sessions/analytics",
        "/sessions/00000000-0000-4000-8000-000000000001",
        "/interaction/context",
        "/interaction/summary/latest",
    ],
)
def test_unauthenticated_requests_are_rejected_with_401(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All sensitive session and interaction endpoints reject unauthenticated callers."""
    _configure_operator(monkeypatch, "interaction:read")
    app = create_app()

    with TestClient(app) as client:
        # 1. Missing Authorization header -> 401
        res_missing = client.get(path)
        assert res_missing.status_code == 401
        assert res_missing.json()["code"] == "UNAUTHENTICATED"

        # 2. Invalid bearer token -> 401
        res_invalid = client.get(path, headers={"Authorization": "Bearer invalid_secret_token"})
        assert res_invalid.status_code == 401
        assert res_invalid.json()["code"] == "UNAUTHENTICATED"


@pytest.mark.parametrize(
    "path",
    [
        "/sessions",
        "/sessions/count",
        "/sessions/analytics",
        "/interaction/context",
        "/interaction/summary/latest",
    ],
)
def test_insufficient_scope_is_rejected_with_403(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Endpoints reject tokens that lack the required interaction:read scope."""
    # Token only has persons:read, lacking interaction:read
    token = _configure_operator(monkeypatch, "persons:read")
    app = create_app()

    with TestClient(app) as client:
        res = client.get(path, headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 403
        assert res.json()["code"] == "INSUFFICIENT_SCOPE"


def test_authenticated_interaction_read_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authenticated caller with interaction:read scope accesses endpoints successfully."""
    token = _configure_operator(monkeypatch, "interaction:read")
    headers = {"Authorization": f"Bearer {token}"}
    app = create_app()

    with TestClient(app) as client:
        res_ctx = client.get("/interaction/context", headers=headers)
        assert res_ctx.status_code == 200

        res_sum = client.get("/interaction/summary/latest", headers=headers)
        assert res_sum.status_code == 200

        res_sess = client.get("/sessions", headers=headers)
        assert res_sess.status_code == 200
        assert isinstance(res_sess.json(), list)

        res_cnt = client.get("/sessions/count", headers=headers)
        assert res_cnt.status_code == 200
        assert "count" in res_cnt.json()

        res_ana = client.get("/sessions/analytics", headers=headers)
        assert res_ana.status_code == 200
        assert "total_sessions" in res_ana.json()


def test_database_manager_enforces_0600_permissions(tmp_path: Path) -> None:
    """DatabaseManager enforces strict 0600 permissions on database file and auxiliary files."""
    db_file = tmp_path / "test_secure.db"

    # Initialize database
    mgr = DatabaseManager(db_path=db_file, auto_init=True)
    assert db_file.exists()

    # Verify main DB file is mode 0600
    file_mode = stat.S_IMODE(db_file.stat().st_mode)
    assert file_mode == 0o600, f"Expected mode 0600, got {oct(file_mode)}"

    # Trigger a write transaction to generate WAL/SHM activity
    with mgr.transaction() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, val TEXT);")
        cur.execute("INSERT INTO test_table (val) VALUES ('secure_data');")

    # Check that any created auxiliary files are also mode 0600
    for aux in [Path(f"{db_file}-wal"), Path(f"{db_file}-shm")]:
        if aux.exists():
            aux_mode = stat.S_IMODE(aux.stat().st_mode)
            assert aux_mode == 0o600, f"Expected {aux} mode 0600, got {oct(aux_mode)}"
