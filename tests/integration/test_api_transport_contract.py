"""API contract tests for the thin transport layer around the application facade."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.backend.models import SessionRecord


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
