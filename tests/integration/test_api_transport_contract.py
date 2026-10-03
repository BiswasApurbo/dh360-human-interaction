"""API contract tests for the thin transport layer around the application facade."""

from __future__ import annotations

import secrets
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.backend.models import SessionRecord
from dh360_hi.contracts.errors import ServiceUnavailableError


def _configure_operator(monkeypatch: pytest.MonkeyPatch, scopes: str) -> str:
    """Configure one ephemeral operator credential for an API test."""
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_SCOPES", scopes)
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)
    return token


def test_transport_layer_exposes_m31_contract() -> None:
    """The thin HTTP layer should expose the approved service-backed endpoints."""
    app = create_app()

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        payload = health.json()
        assert "status" in payload
        assert "subsystems" in payload
        assert "timestamp_iso" in payload

        context = client.get("/interaction/context")
        assert context.status_code == 200
        context_payload = context.json()
        assert "state" in context_payload
        assert "identity_known" in context_payload

        latest_summary = client.get("/interaction/summary/latest")
        assert latest_summary.status_code == 200
        assert latest_summary.json() is None

        runtime_state = client.get("/runtime/state")
        assert runtime_state.status_code == 200
        assert runtime_state.json() in {"stopped", "running", "paused", "error"}

        metrics = client.get("/runtime/metrics")
        assert metrics.status_code == 503

        sessions = client.get("/sessions")
        assert sessions.status_code == 200
        assert sessions.json() == []

        count = client.get("/sessions/count")
        assert count.status_code == 200
        assert count.json() == {"count": 0}

        analytics = client.get("/sessions/analytics")
        assert analytics.status_code == 200
        analytics_payload = analytics.json()
        assert analytics_payload["total_sessions"] == 0
        assert analytics_payload["identified_sessions_count"] == 0
        assert analytics_payload["anonymous_sessions_count"] == 0


def test_session_detail_and_analytics_return_populated_dtos() -> None:
    """Historical session endpoints serialize populated application DTOs."""
    app = create_app()
    first_session_id = "00000000-0000-4000-8000-000000000001"
    anonymous_session_id = "00000000-0000-4000-8000-000000000002"

    with TestClient(app) as client:
        repository = app.state.service.session_repository
        assert repository is not None
        repository.save_session(
            SessionRecord(
                session_id=first_session_id,
                person_id=None,
                display_name=None,
                started_at=100.0,
                ended_at=110.0,
                duration_seconds=10.0,
                final_state="engaged",
                identity_was_known=False,
                identity_transitions_count=0,
            )
        )
        repository.save_session(
            SessionRecord(
                session_id=anonymous_session_id,
                person_id=None,
                display_name=None,
                started_at=200.0,
                ended_at=230.0,
                duration_seconds=30.0,
                final_state="farewell",
                identity_was_known=False,
                identity_transitions_count=0,
            )
        )

        detail = client.get(f"/sessions/{first_session_id}")
        assert detail.status_code == 200
        assert detail.json()["session_id"] == first_session_id
        assert detail.json()["person_id"] is None
        assert detail.json()["duration_seconds"] == 10.0

        missing = client.get("/sessions/00000000-0000-4000-8000-000000000099")
        assert missing.status_code == 404
        assert missing.json()["code"] == "ENTITY_NOT_FOUND"

        sessions = client.get("/sessions")
        assert sessions.status_code == 200
        assert len(sessions.json()) == 2

        analytics = client.get("/sessions/analytics")
        assert analytics.status_code == 200
        analytics_payload = analytics.json()
        assert analytics_payload["total_sessions"] == 2
        assert analytics_payload["total_duration_seconds"] == 40.0
        assert analytics_payload["identified_sessions_count"] == 0
        assert analytics_payload["anonymous_sessions_count"] == 2


@pytest.mark.parametrize(
    "path",
    [
        "/sessions/analytics?person_id=not-a-uuid",
        "/sessions/analytics?started_after=200&started_before=100",
        "/sessions/analytics?ended_after=200&ended_before=100",
        "/sessions/analytics?min_duration_seconds=20&max_duration_seconds=10",
        "/sessions/analytics?limit=0",
        "/sessions?person_id=not-a-uuid",
        "/sessions/count?person_id=not-a-uuid",
    ],
)
def test_invalid_session_queries_return_422(path: str) -> None:
    """Malformed and inconsistent session query parameters return client errors."""
    app = create_app()

    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 422
    assert response.json()["error"] == "VALIDATION_ERROR"


def test_person_routes_fail_closed_without_operator_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing auth configuration blocks person routes without changing M3.1 routes."""
    monkeypatch.delenv("DH360_HI_API__OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("DH360_HI_API__OPERATOR_SCOPES", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)

    with TestClient(create_app()) as client:
        response = client.get("/persons")
        health = client.get("/health")

    assert response.status_code == 503
    assert response.json()["code"] == "OPERATOR_AUTH_NOT_CONFIGURED"
    assert health.status_code == 200


def test_person_routes_reject_missing_and_invalid_bearer_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured person routes require a matching bearer token."""
    token = _configure_operator(monkeypatch, "persons:read,persons:create")
    invalid_token = secrets.token_urlsafe(32)

    with TestClient(create_app()) as client:
        missing = client.get("/persons")
        invalid = client.get(
            "/persons",
            headers={"Authorization": f"Bearer {invalid_token}"},
        )

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert missing.json()["code"] == "UNAUTHENTICATED"
    assert invalid.status_code == 401
    assert invalid.json()["code"] == "UNAUTHENTICATED"
    assert token not in missing.text
    assert token not in invalid.text


def test_person_routes_do_not_accept_tokens_in_query_or_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured token only authenticates through the Authorization header."""
    token = _configure_operator(monkeypatch, "persons:read,persons:create")

    with TestClient(create_app()) as client:
        query_token = client.get("/persons", params={"access_token": token})
        body_token = client.post(
            "/persons",
            json={"display_name": "Body credential", "operator_token": token},
        )

    assert query_token.status_code == 401
    assert query_token.json()["code"] == "UNAUTHENTICATED"
    assert body_token.status_code in {401, 422}
    assert body_token.status_code != 201
    assert token not in query_token.text
    assert token not in body_token.text


def test_operator_read_and_create_scopes_return_sanitized_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read/create permissions expose only the existing sanitized person DTO."""
    token = _configure_operator(monkeypatch, "persons:read,persons:create")
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(create_app()) as client:
        empty_list = client.get("/persons", headers=headers)
        created = client.post(
            "/persons",
            headers=headers,
            json={"display_name": "Operator-created profile", "preferred_name": "Operator"},
        )
        person_id = created.json()["person_id"]
        detail = client.get(f"/persons/{person_id}", headers=headers)
        populated_list = client.get("/persons", headers=headers)

    assert empty_list.status_code == 200
    assert empty_list.json() == []
    assert created.status_code == 201
    profile = created.json()
    assert profile["status"] == "active"
    assert profile["consent_status"] == "pending"
    assert set(profile) == {
        "person_id",
        "display_name",
        "preferred_name",
        "status",
        "consent_status",
        "consent_timestamp",
        "created_at",
        "updated_at",
    }
    assert detail.status_code == 200
    assert detail.json() == profile
    assert populated_list.status_code == 200
    assert populated_list.json() == [profile]


def test_person_create_requires_create_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """A read-only operator cannot create profiles."""
    token = _configure_operator(monkeypatch, "persons:read")

    with TestClient(create_app()) as client:
        response = client.post(
            "/persons",
            headers={"Authorization": f"Bearer {token}"},
            json={"display_name": "Unauthorized profile"},
        )

    assert response.status_code == 403
    assert response.json()["code"] == "INSUFFICIENT_SCOPE"


@pytest.mark.parametrize(
    "body",
    [
        {"display_name": "   "},
        {"display_name": "Forbidden consent", "consent_status": "granted"},
        {"display_name": "Forbidden note", "notes": "private"},
    ],
)
def test_person_create_rejects_invalid_and_unknown_fields(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, str],
) -> None:
    """Creation accepts only the strict profile-name request contract."""
    token = _configure_operator(monkeypatch, "persons:create")

    with TestClient(create_app()) as client:
        response = client.post(
            "/persons",
            headers={"Authorization": f"Bearer {token}"},
            json=body,
        )

    assert response.status_code == 422
    assert response.json()["error"] == "VALIDATION_ERROR"


def test_person_get_not_found_uses_existing_error_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing valid person IDs map to the existing application not-found response."""
    token = _configure_operator(monkeypatch, "persons:read")

    with TestClient(create_app()) as client:
        response = client.get(
            "/persons/00000000-0000-4000-8000-000000000099",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 404
    assert response.json()["code"] == "ENTITY_NOT_FOUND"


def test_person_mutations_require_their_specific_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Update scope does not grant consent management, delete, or biometric reset."""
    token = _configure_operator(monkeypatch, "persons:create,persons:update")
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(create_app()) as client:
        created = client.post("/persons", headers=headers, json={"display_name": "Casey"})
        person_id = created.json()["person_id"]
        updated = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={"display_name": "Casey Updated"},
        )
        consent_update = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={"consent_status": "granted"},
        )
        deletion = client.delete(f"/persons/{person_id}", headers=headers)
        reset = client.delete(f"/persons/{person_id}/biometrics", headers=headers)
        unknown_field = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={"notes": "not exposed"},
        )
        empty_update = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={},
        )
        invalid_status = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={"status": "suspended"},
        )
        blank_preferred_name = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={"preferred_name": "   "},
        )

    assert created.status_code == 201
    assert updated.status_code == 200
    assert updated.json()["display_name"] == "Casey Updated"
    assert consent_update.status_code == 403
    assert consent_update.json()["code"] == "INSUFFICIENT_SCOPE"
    assert deletion.status_code == 403
    assert reset.status_code == 403
    assert unknown_field.status_code == 422
    assert empty_update.status_code == 422
    assert invalid_status.status_code == 422
    assert blank_preferred_name.status_code == 422


def test_authorized_reset_retains_profile_and_delete_returns_not_found_afterward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset removes biometric records only; delete removes the profile afterward."""
    token = _configure_operator(
        monkeypatch,
        "persons:read,persons:create,persons:update,persons:delete,biometrics:reset,consent:manage",
    )
    headers = {"Authorization": f"Bearer {token}"}
    app = create_app()

    with TestClient(app) as client:
        created = client.post("/persons", headers=headers, json={"display_name": "Taylor"})
        assert created.status_code == 201
        person_id = created.json()["person_id"]
        app.state.service.person_service.add_person_embedding(
            person_id,
            np.ones(128, dtype=np.float32),
        )
        assert app.state.service._recognizer.gallery.get_entry(person_id) is None

        grant_consent = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={"consent_status": "granted"},
        )
        assert app.state.service._recognizer.gallery.get_entry(person_id) is not None
        withdraw_consent = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={"consent_status": "withdrawn"},
        )
        assert app.state.service._recognizer.gallery.get_entry(person_id) is None
        restore_consent = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={"consent_status": "granted"},
        )
        assert app.state.service._recognizer.gallery.get_entry(person_id) is not None
        deactivate = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={"status": "inactive"},
        )
        assert app.state.service._recognizer.gallery.get_entry(person_id) is None
        reset = client.delete(f"/persons/{person_id}/biometrics", headers=headers)
        profile_after_reset = client.get(f"/persons/{person_id}", headers=headers)
        embeddings_after_reset = app.state.service.person_service.get_person_embeddings(person_id)
        delete = client.delete(f"/persons/{person_id}", headers=headers)
        missing = client.get(f"/persons/{person_id}", headers=headers)

    assert grant_consent.status_code == 200
    assert grant_consent.json()["consent_status"] == "granted"
    assert withdraw_consent.status_code == 200
    assert withdraw_consent.json()["consent_status"] == "withdrawn"
    assert restore_consent.status_code == 200
    assert deactivate.status_code == 200
    assert deactivate.json()["status"] == "inactive"
    assert reset.status_code == 204
    assert profile_after_reset.status_code == 200
    assert profile_after_reset.json()["person_id"] == person_id
    assert embeddings_after_reset == []
    assert delete.status_code == 204
    assert missing.status_code == 404
    assert missing.json()["code"] == "ENTITY_NOT_FOUND"


def test_missing_person_mutations_return_existing_not_found_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Update, delete, and reset map missing records to the existing 404 contract."""
    token = _configure_operator(
        monkeypatch,
        "persons:update,persons:delete,biometrics:reset",
    )
    headers = {"Authorization": f"Bearer {token}"}
    person_id = "00000000-0000-4000-8000-000000000099"

    with TestClient(create_app()) as client:
        updated = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={"display_name": "Missing"},
        )
        deleted = client.delete(f"/persons/{person_id}", headers=headers)
        reset = client.delete(f"/persons/{person_id}/biometrics", headers=headers)

    for response in (updated, deleted, reset):
        assert response.status_code == 404
        assert response.json()["code"] == "ENTITY_NOT_FOUND"


def test_consent_update_rejected_when_gallery_policy_ignores_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consent withdrawal cannot be reported successful if gallery policy ignores it."""
    token = _configure_operator(
        monkeypatch,
        "persons:read,persons:create,persons:update,consent:manage",
    )
    monkeypatch.setenv("DH360_HI_REQUIRE_CONSENT_FOR_GALLERY", "false")
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(create_app()) as client:
        created = client.post("/persons", headers=headers, json={"display_name": "Pending"})
        person_id = created.json()["person_id"]
        response = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={"consent_status": "withdrawn"},
        )
        current = client.get(
            f"/persons/{person_id}",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 409
    assert response.json()["code"] == "CONFLICT_ERROR"
    assert current.status_code == 200
    assert current.json()["consent_status"] == "pending"


def test_mutation_service_unavailable_uses_existing_503_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safe-point failures retain the existing application service-unavailable mapping."""
    token = _configure_operator(monkeypatch, "persons:create,persons:update")
    headers = {"Authorization": f"Bearer {token}"}
    app = create_app()

    with TestClient(app) as client:
        created = client.post("/persons", headers=headers, json={"display_name": "Unavailable"})
        person_id = created.json()["person_id"]
        app.state.service.update_person = MagicMock(
            side_effect=ServiceUnavailableError("person_mutation", "safe point timeout")
        )
        response = client.patch(
            f"/persons/{person_id}",
            headers=headers,
            json={"display_name": "Unavailable Updated"},
        )

    assert response.status_code == 503
    assert response.json()["code"] == "SERVICE_UNAVAILABLE"
