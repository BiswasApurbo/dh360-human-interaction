"""Contract tests for the M3.2.3 server-side camera enrollment HTTP surface."""

from __future__ import annotations

import secrets
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.contracts.dto import EnrollmentProgressDTO, EnrollmentResultDTO

_PERSON_ID = "11111111-1111-1111-1111-111111111111"
_MISSING_ID = "99999999-9999-9999-9999-999999999999"


def _configure_operator(monkeypatch: pytest.MonkeyPatch, scopes: str) -> str:
    """Configure one ephemeral operator credential for an enrollment API test."""
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_SCOPES", scopes)
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)
    return token


class _StubService:
    """Minimal service double exercising the enrollment router without a real pipeline."""

    def __init__(self) -> None:
        self._progress: EnrollmentProgressDTO | None = None
        self._config = SimpleNamespace(camera_index=0)

    @property
    def config(self) -> Any:
        """Server-side configuration consumed by the enrollment router."""
        return self._config

    def get_person(self, person_id: str) -> Any:
        return object() if person_id == _PERSON_ID else None

    def get_enrollment_progress(self) -> EnrollmentProgressDTO | None:
        return self._progress

    def start_enrollment(
        self, person_id: str, mode: str, capture_source: Any = None
    ) -> EnrollmentProgressDTO:
        _ = capture_source
        self._progress = EnrollmentProgressDTO(
            session_id="s-1",
            enrollment_batch_id="b-1",
            person_id=person_id,
            display_name="Test",
            mode=mode,
            state="collecting",
            samples_collected=0,
            target_samples=25,
            minimum_samples=5,
            total_attempts=0,
            is_viable=False,
            is_complete=False,
            current_direction="front",
            direction_counts={"front": 0, "right": 0, "left": 0, "up": 0, "down": 0},
            direction_targets={"front": 5, "right": 5, "left": 5, "up": 5, "down": 5},
        )
        return self._progress

    def step_enrollment(self) -> EnrollmentProgressDTO:
        assert self._progress is not None
        return self._progress

    def commit_enrollment(self) -> EnrollmentResultDTO:
        self._progress = None
        return EnrollmentResultDTO(
            session_id="s-1",
            enrollment_batch_id="b-1",
            person_id=_PERSON_ID,
            display_name="Test",
            mode="add_samples",
            is_success=True,
            state="completed",
            samples_enrolled=5,
            persisted_embedding_count=5,
            total_attempts=5,
        )

    def cancel_enrollment(self, reason: str = "User cancelled") -> bool:
        _ = reason
        self._progress = None
        return True


def test_start_requires_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Starting an enrollment session without valid credentials is rejected with 401."""
    _configure_operator(monkeypatch, "enrollment:manage")
    app = create_app()
    with TestClient(app) as client:
        response = client.post(f"/persons/{_PERSON_ID}/enrollment/sessions", json={})
    assert response.status_code == 401


def test_start_requires_enrollment_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """A credential without the enrollment:manage scope is rejected with 403."""
    token = _configure_operator(monkeypatch, "persons:read")
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            f"/persons/{_PERSON_ID}/enrollment/sessions",
            json={},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 403


def test_start_missing_person_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """Starting a session for an unknown person returns the standard not-found contract."""
    token = _configure_operator(monkeypatch, "enrollment:manage")
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            f"/persons/{_MISSING_ID}/enrollment/sessions",
            json={},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 404


def test_cancel_any_enrollment_session_returns_200(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _configure_operator(monkeypatch, "enrollment:manage")
    app = create_app()
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        client.app.state.service = _StubService()

        client.post(
            f"/persons/{_PERSON_ID}/enrollment/sessions",
            json={"mode": "add_samples"},
            headers=headers,
        )

        cancel_resp = client.delete("/persons/enrollment/sessions/current", headers=headers)
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["cancelled"] is True
        assert cancel_resp.json()["person_id"] == _PERSON_ID


def test_cancel_any_enrollment_session_returns_409_if_none(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _configure_operator(monkeypatch, "enrollment:manage")
    app = create_app()
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        resp = client.delete("/persons/enrollment/sessions/current", headers=headers)
        assert resp.status_code == 409
        assert "No active enrollment session" in resp.json()["detail"]


def test_session_endpoints_require_active_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Progress/step/commit/cancel all return 404 when no session is active."""
    token = _configure_operator(monkeypatch, "enrollment:manage")
    app = create_app()
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        calls = [
            ("get", f"/persons/{_PERSON_ID}/enrollment/sessions"),
            ("post", f"/persons/{_PERSON_ID}/enrollment/sessions/step"),
            ("post", f"/persons/{_PERSON_ID}/enrollment/sessions/commit"),
            ("post", f"/persons/{_PERSON_ID}/enrollment/sessions/cancel"),
        ]
        for method, path in calls:
            response = getattr(client, method)(path, headers=headers)
            assert response.status_code == 404, (method, path, response.status_code)


def test_full_session_lifecycle_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stub service drives the full start -> step -> commit HTTP lifecycle."""
    token = _configure_operator(monkeypatch, "enrollment:manage")
    app = create_app()
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        client.app.state.service = _StubService()

        start = client.post(
            f"/persons/{_PERSON_ID}/enrollment/sessions",
            json={"mode": "add_samples"},
            headers=headers,
        )
        assert start.status_code == 201
        start_data = start.json()
        assert start_data["state"] == "collecting"
        assert start_data["current_direction"] == "front"
        assert start_data["direction_targets"]["front"] == 5
        assert start_data["direction_counts"]["front"] == 0

        current = client.get(f"/persons/{_PERSON_ID}/enrollment/sessions", headers=headers)
        assert current.status_code == 200
        assert current.json()["current_direction"] == "front"

        step = client.post(f"/persons/{_PERSON_ID}/enrollment/sessions/step", headers=headers)
        assert step.status_code == 200

        commit = client.post(f"/persons/{_PERSON_ID}/enrollment/sessions/commit", headers=headers)
        assert commit.status_code == 200
        assert commit.json()["is_success"] is True


@pytest.mark.parametrize(
    "body",
    [
        {"capture": {"source": "webcam"}},
        {"capture": {"source": "video", "video_file": "/tmp/video.mp4"}},
        {"capture": {"source": "synthetic"}},
        {"capture_source": "video"},
        {"camera_index": 5},
        {"video_file": "/tmp/video.mp4"},
        {"source": "synthetic"},
    ],
)
def test_client_cannot_select_capture_source_returns_422(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
) -> None:
    """Client-supplied capture/source/camera/video/synthetic fields are rejected with 422."""
    token = _configure_operator(monkeypatch, "enrollment:manage")
    app = create_app()
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        client.app.state.service = _StubService()
        response = client.post(
            f"/persons/{_PERSON_ID}/enrollment/sessions",
            json=body,
            headers=headers,
        )
    assert response.status_code == 422
