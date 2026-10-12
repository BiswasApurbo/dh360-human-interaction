"""Integration tests for authorization API endpoints and composition wiring."""

from __future__ import annotations

import secrets
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.edge.authorization_controller import AuthorizationController
from dh360_hi.edge.session_manager import SessionConfig, SessionManager
from dh360_hi.vision.identity_state import IdentityState, TrackIdentityState
from dh360_hi.vision.target_selector import TargetCandidate, TargetSelectionResult
from dh360_hi.vision.two_rate_pipeline import TwoRateFrameResult

_AUTH_SESSIONS_PATH = "/authorization/sessions"
_AUTH_CURRENT_PATH = "/authorization/sessions/current"


def _configure_operator(monkeypatch: pytest.MonkeyPatch, scopes: str) -> str:
    """Configure one ephemeral operator credential for an API test."""
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_SCOPES", scopes)
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)
    return token


def _mk_session_manager() -> SessionManager:
    return SessionManager(config=SessionConfig(require_confirmed_presence=True))


def _mk_id_state(
    track_id: int,
    state: IdentityState = IdentityState.IDENTIFIED,
    person_id: str | None = None,
    display_name: str | None = None,
) -> TrackIdentityState:
    return TrackIdentityState(
        track_id=track_id,
        state=state,
        current_person_id=person_id,
        current_display_name=display_name,
    )


def _mk_frame(
    frame_id: int = 1,
    timestamp: float = 1.0,
    primary_track_id: int | None = None,
    identity_states: dict[int, TrackIdentityState] | None = None,
) -> TwoRateFrameResult:
    primary = None
    if primary_track_id is not None:
        primary = TargetCandidate(
            track_id=primary_track_id,
            score=1.0,
            centrality_score=1.0,
            size_score=1.0,
            identity_state_score=1.0,
            dwell_score=1.0,
            is_identified=True,
            identity_state=IdentityState.IDENTIFIED,
            person_id=None,
            display_name=None,
            bbox=(0.0, 0.0, 100.0, 100.0),
            center=(50.0, 50.0),
        )
    target_result = TargetSelectionResult(
        primary_target=primary,
        candidates=[primary] if primary else [],
        target_changed=False,
        previous_target_track_id=None,
        frame_id=frame_id,
    )
    return TwoRateFrameResult(
        frame_id=frame_id,
        timestamp=timestamp,
        detections=[],
        tracks=[],
        quality_results={},
        identity_states=identity_states or {},
        target_result=target_result,
        presence_transitions=[],
        completed_recognitions=[],
        ambiguous_crossing_track_ids=set(),
        pending_worker_tasks=0,
        frame_latency_ms=0.0,
        total_pipeline_latency_ms=0.0,
    )


class TestAuthorizationApiAuthentication:
    def test_missing_bearer_token_returns_401(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _configure_operator(monkeypatch, "authorization:manage")
        app = create_app()
        with TestClient(app) as client:
            resp_post = client.post(_AUTH_SESSIONS_PATH)
            assert resp_post.status_code == 401
            assert resp_post.headers.get("WWW-Authenticate") == "Bearer"

            resp_get = client.get(_AUTH_CURRENT_PATH)
            assert resp_get.status_code == 401
            assert resp_get.headers.get("WWW-Authenticate") == "Bearer"

            resp_del = client.delete(_AUTH_CURRENT_PATH)
            assert resp_del.status_code == 401
            assert resp_del.headers.get("WWW-Authenticate") == "Bearer"

            # Invalid / mismatch token (e.g. placeholder token) -> 401
            resp_invalid = client.delete(
                _AUTH_CURRENT_PATH,
                headers={"Authorization": "Bearer super-secret-demo-token"},
            )
            assert resp_invalid.status_code == 401
            assert resp_invalid.headers.get("WWW-Authenticate") == "Bearer"
            assert resp_invalid.json()["code"] == "UNAUTHENTICATED"

    def test_wrong_scope_returns_403(self, monkeypatch: pytest.MonkeyPatch) -> None:
        token = _configure_operator(monkeypatch, "interaction:read")
        app = create_app()
        headers = {"Authorization": f"Bearer {token}"}
        with TestClient(app) as client:
            resp_post = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert resp_post.status_code == 403
            assert resp_post.json()["code"] == "INSUFFICIENT_SCOPE"

            resp_get = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert resp_get.status_code == 403
            assert resp_get.json()["code"] == "INSUFFICIENT_SCOPE"

            resp_del = client.delete(_AUTH_CURRENT_PATH, headers=headers)
            assert resp_del.status_code == 403
            assert resp_del.json()["code"] == "INSUFFICIENT_SCOPE"

    def test_authorization_manage_scope_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=5.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}
        with TestClient(app) as client:
            resp = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert resp.status_code == 200
            assert resp.json() is None


class TestAuthorizationSessionLifecycleEndpoints:
    def test_controller_unavailable_returns_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage")
        app = create_app()  # Perception disabled, controller is None
        headers = {"Authorization": f"Bearer {token}"}
        with TestClient(app) as client:
            resp_post = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert resp_post.status_code == 503
            assert "not enabled" in resp_post.json()["detail"]

            resp_get = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert resp_get.status_code == 503
            assert "not enabled" in resp_get.json()["detail"]

    def test_post_starts_session_returns_201_with_sanitized_dto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=5.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}

        with TestClient(app) as client:
            resp = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert resp.status_code == 201
            payload = resp.json()

            # Verify public sanitized fields
            assert "authorization_id" in payload
            assert payload["status"] == "active"
            assert payload["person_id"] is None
            assert payload["display_name"] is None
            assert payload["ended_at_iso"] is None
            assert "created_at_iso" in payload
            assert "updated_at_iso" in payload

            # Verify privacy / no internal leaked fields
            forbidden_fields = [
                "track_id",
                "frame_id",
                "embeddings",
                "embedding",
                "bbox",
                "recognition",
                "RecognitionResult",
            ]
            for f in forbidden_fields:
                assert f not in payload

    def test_cancel_session_returns_200_and_ends_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=5.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}
        with TestClient(app) as client:
            client.post(_AUTH_SESSIONS_PATH, headers=headers)

            resp = client.delete("/authorization/sessions/current", headers=headers)
            assert resp.status_code == 200
            assert resp.json()["status"] == "cancelled"

            resp2 = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert resp2.status_code == 201

    def test_cancel_returns_409_when_no_active_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=5.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}
        with TestClient(app) as client:
            resp = client.delete("/authorization/sessions/current", headers=headers)
            assert resp.status_code == 409
            assert "No active authorization session" in resp.json()["detail"]

    def test_second_active_post_returns_409_conflict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=5.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}

        with TestClient(app) as client:
            resp1 = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert resp1.status_code == 201

            resp2 = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert resp2.status_code == 409
            assert "already exists" in resp2.json()["detail"]

    def test_get_current_before_session_returns_null(self, monkeypatch: pytest.MonkeyPatch) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=5.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}

        with TestClient(app) as client:
            resp = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert resp.status_code == 200
            assert resp.json() is None

    def test_get_current_after_start_returns_active_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=5.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}

        with TestClient(app) as client:
            start_resp = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert start_resp.status_code == 201
            start_payload = start_resp.json()

            get_resp = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert get_resp.status_code == 200
            get_payload = get_resp.json()
            assert get_payload == start_payload
            assert get_payload["status"] == "active"


class TestCompositionAndLifecycle:
    def test_same_controller_instance_used_by_router(self, monkeypatch: pytest.MonkeyPatch) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=5.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}

        with TestClient(app) as client:
            # Router must mutate the exact controller instance passed into create_app
            assert ctrl.get_current() is None
            client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert ctrl.get_current() is not None
            assert ctrl.get_current().status.value == "active"  # type: ignore[union-attr]

            id_state = _mk_id_state(
                1, IdentityState.IDENTIFIED, person_id="p-42", display_name="Bob"
            )
            frame = _mk_frame(primary_track_id=1, identity_states={1: id_state})
            sm.process_frame_result(frame)

            # GET through API router observes the frame-driven state transition
            resp = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["status"] == "authorized"
            assert payload["person_id"] == "p-42"
            assert payload["display_name"] == "Bob"

    def test_lifespan_closes_authorization_controller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=5.0)
        app = create_app(authorization_controller=ctrl)

        with mock.patch.object(ctrl, "close", wraps=ctrl.close) as mock_close:
            with TestClient(app):
                assert mock_close.call_count == 0
            # On exiting TestClient context, lifespan finally block runs
            assert mock_close.call_count == 1
            # Observer should be unregistered from SessionManager
            assert ctrl._on_frame not in sm._observers

    def test_no_perception_bypass_from_router(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify the router never directly calls runtime or SessionManager."""
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=5.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}

        with (
            mock.patch.object(sm, "process_frame_result") as mock_process,
            TestClient(app) as client,
        ):
            client.post(_AUTH_SESSIONS_PATH, headers=headers)
            client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert mock_process.call_count == 0

    def test_session_replacement_and_cancellation_lifecycle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify sessions can be cancelled, replaced, and safely handled
        when no session is active.
        """
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=10.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}

        with TestClient(app) as client:
            # 1. Start initial session
            r1 = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert r1.status_code == 201
            s1_id = r1.json()["authorization_id"]

            # 2. Attempting to start another session while active yields 409 Conflict
            r_conflict = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert r_conflict.status_code == 409
            assert "already exists" in r_conflict.json()["detail"]

            # 3. Cancel active session
            r_cancel = client.delete(_AUTH_CURRENT_PATH, headers=headers)
            assert r_cancel.status_code == 200
            assert r_cancel.json()["authorization_id"] == s1_id
            assert r_cancel.json()["status"] == "cancelled"

            # 4. Immediate replacement starts fresh session
            r2 = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert r2.status_code == 201
            s2_id = r2.json()["authorization_id"]
            assert s2_id != s1_id

            # 5. GET /authorization/sessions/current reflects new session
            r_curr = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert r_curr.status_code == 200
            assert r_curr.json()["authorization_id"] == s2_id
            assert r_curr.json()["status"] == "active"

            # 6. Cancel replacement session
            r_cancel2 = client.delete(_AUTH_CURRENT_PATH, headers=headers)
            assert r_cancel2.status_code == 200
            assert r_cancel2.json()["authorization_id"] == s2_id

            # 7. Repeated cancellation when no active session returns 409
            r_cancel_again = client.delete(_AUTH_CURRENT_PATH, headers=headers)
            assert r_cancel_again.status_code == 409
            assert "No active authorization session" in r_cancel_again.json()["detail"]

            # 8. Starting session after 409 cancel succeeds
            r3 = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert r3.status_code == 201
            assert r3.json()["authorization_id"] != s2_id

    def test_authorization_transition_emits_sse_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that authorization state transitions are published as SSE events."""
        import asyncio
        import contextlib
        import json
        from typing import Any

        token = _configure_operator(monkeypatch, "authorization:manage,interaction:read")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=5.0)
        app = create_app(authorization_controller=ctrl)

        class _StreamHarness:
            def __init__(self) -> None:
                self.chunks: list[bytes] = []
                self._disconnected = False
                self.status: int | None = None

            async def _receive(self) -> dict[str, Any]:
                if self._disconnected:
                    await asyncio.sleep(0)
                    return {"type": "http.disconnect"}
                await asyncio.sleep(0)
                return {"type": "http.request", "body": b"", "more_body": False}

            async def _send(self, message: dict[str, Any]) -> None:
                if message["type"] == "http.response.start":
                    self.status = int(message["status"])
                elif message["type"] == "http.response.body":
                    b = message.get("body", b"")
                    if b:
                        self.chunks.append(bytes(b))

            def text(self) -> str:
                return b"".join(self.chunks).decode()

        stream = _StreamHarness()
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/interaction/events",
            "raw_path": b"/interaction/events",
            "query_string": b"",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
            "state": {},
        }

        async def run_stream_and_trigger() -> None:
            task = asyncio.ensure_future(app(scope, stream._receive, stream._send))
            # Wait for stream to start and receive initial frame
            for _ in range(100):
                await asyncio.sleep(0.01)
                if len(stream.chunks) >= 1:
                    break

            # Start authorization session on the controller directly
            ctrl.start()

            # Feed frame to transition to AUTHORIZED
            id_state = _mk_id_state(
                1, IdentityState.IDENTIFIED, person_id="p-auth-1", display_name="Alice"
            )
            frame = _mk_frame(primary_track_id=1, identity_states={1: id_state})
            sm.process_frame_result(frame)

            # Wait for SSE frames
            for _ in range(100):
                await asyncio.sleep(0.01)
                if "authorization_updated" in stream.text():
                    break

            stream._disconnected = True
            for _ in range(50):
                if task.done():
                    break
                await asyncio.sleep(0.01)
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        asyncio.run(run_stream_and_trigger())

        assert stream.status == 200
        output = stream.text()
        assert "event: authorization_updated" in output
        # Find the authorization_updated event chunk
        lines = output.splitlines()
        found_auth_event = False
        for i, line in enumerate(lines):
            if line.strip() == "event: authorization_updated":
                data_line = lines[i + 1]
                assert data_line.startswith("data: ")
                payload = json.loads(data_line[len("data: ") :])
                assert payload["event_type"] == "authorization_updated"
                assert payload["status"] in ("active", "authorized")
                assert "authorization_id" in payload
                found_auth_event = True
                break
        assert found_auth_event, f"Did not find authorization_updated event in output: {output}"


class TestAuthorizedSessionContextEndpoints:
    """Tests for GET /authorization/sessions/{authorization_id}/context."""

    def test_authorized_session_returns_full_context_and_qa(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage,persons:create,persons:read")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=10.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}

        with TestClient(app) as client:
            # 1. Create person with full context
            create_resp = client.post(
                "/persons",
                headers=headers,
                json={
                    "display_name": "Dr. Gregory House",
                    "preferred_name": "House",
                    "role": "Head of Diagnostics",
                    "department": "Nephrology & Diagnostics",
                    "office_room": "Room 402",
                    "working_hours": "Mon-Fri 08:00-16:00",
                    "profile_description": "Specialist in diagnostic medicine.",
                    "qa_pairs": [
                        {"question": "Specialty?", "answer": "Diagnostics"},
                        {"question": "Pager?", "answer": "4412"},
                    ],
                },
            )
            assert create_resp.status_code == 201
            person_id = create_resp.json()["person_id"]

            # Verify public GET /persons/{id} does NOT leak context
            pub_resp = client.get(f"/persons/{person_id}", headers=headers)
            assert pub_resp.status_code == 200
            assert "profile_description" not in pub_resp.json()
            assert "qa_pairs" not in pub_resp.json()

            # 2. Start session
            start_resp = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert start_resp.status_code == 201
            auth_id = start_resp.json()["authorization_id"]

            # 3. Transition to AUTHORIZED via perception frame
            id_state = _mk_id_state(
                1, IdentityState.IDENTIFIED, person_id=person_id, display_name="House"
            )
            frame = _mk_frame(primary_track_id=1, identity_states={1: id_state})
            sm.process_frame_result(frame)

            curr = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert curr.json()["status"] == "authorized"

            # 4. Fetch privileged context
            ctx_resp = client.get(f"/authorization/sessions/{auth_id}/context", headers=headers)
            assert ctx_resp.status_code == 200
            data = ctx_resp.json()
            assert data["person_id"] == person_id
            assert data["display_name"] == "Dr. Gregory House"
            assert data["preferred_name"] == "House"
            assert data["role"] == "Head of Diagnostics"
            assert data["department"] == "Nephrology & Diagnostics"
            assert data["office_room"] == "Room 402"
            assert data["working_hours"] == "Mon-Fri 08:00-16:00"
            assert data["profile_description"] == "Specialist in diagnostic medicine."
            assert len(data["qa_pairs"]) == 2
            assert data["qa_pairs"][0] == {"question": "Specialty?", "answer": "Diagnostics"}
            assert data["qa_pairs"][1] == {"question": "Pager?", "answer": "4412"}

            # Verify no biometric or tracker internals leaked
            for forbidden in ("track_id", "frame_id", "embedding", "embeddings", "bbox"):
                assert forbidden not in data

    @pytest.mark.parametrize(
        "state,track_id,expected_status",
        [
            (None, None, "active"),
            (None, 1, "active"),  # active observation
            (IdentityState.UNREGISTERED, 1, "unknown"),
            (IdentityState.UNCERTAIN, 1, "uncertain"),
        ],
    )
    def test_non_authorized_states_return_403(
        self,
        monkeypatch: pytest.MonkeyPatch,
        state: IdentityState | None,
        track_id: int | None,
        expected_status: str,
    ) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=10.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}

        with TestClient(app) as client:
            start_resp = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            auth_id = start_resp.json()["authorization_id"]

            if state is not None and track_id is not None:
                id_state = _mk_id_state(track_id, state)
                frame = _mk_frame(primary_track_id=track_id, identity_states={track_id: id_state})
                sm.process_frame_result(frame)

            curr = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert curr.json()["status"] == expected_status

            ctx_resp = client.get(f"/authorization/sessions/{auth_id}/context", headers=headers)
            assert ctx_resp.status_code == 403
            assert "privileged context is restricted" in ctx_resp.json()["detail"]

    def test_lost_and_timeout_states_return_403(self, monkeypatch: pytest.MonkeyPatch) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=0.01)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}

        with TestClient(app) as client:
            start_resp = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            auth_id = start_resp.json()["authorization_id"]

            # Trigger timeout
            import time

            time.sleep(0.02)
            curr = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert curr.json()["status"] == "timeout"

            ctx_resp = client.get(f"/authorization/sessions/{auth_id}/context", headers=headers)
            assert ctx_resp.status_code == 403

    def test_missing_authorization_session_returns_404(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = _configure_operator(monkeypatch, "authorization:manage")
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=10.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}

        with TestClient(app) as client:
            resp = client.get(
                "/authorization/sessions/00000000-0000-0000-0000-000000000000/context",
                headers=headers,
            )
            assert resp.status_code == 404
            assert "not found" in resp.json()["detail"]

    def test_delete_person_cancels_matching_authorization_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = _configure_operator(
            monkeypatch,
            "authorization:manage,persons:create,persons:read,persons:delete",
        )
        sm = _mk_session_manager()
        ctrl = AuthorizationController(session_manager=sm, timeout_seconds=30.0)
        app = create_app(authorization_controller=ctrl)
        headers = {"Authorization": f"Bearer {token}"}

        with TestClient(app) as client:
            # 1. Create target person and unrelated person
            p_target = client.post(
                "/persons",
                headers=headers,
                json={"display_name": "Target Person"},
            ).json()["person_id"]

            p_unrelated = client.post(
                "/persons",
                headers=headers,
                json={"display_name": "Unrelated Person"},
            ).json()["person_id"]

            # 2. Start authorization session and identify target person
            start_resp = client.post(_AUTH_SESSIONS_PATH, headers=headers)
            assert start_resp.status_code == 201
            auth_id = start_resp.json()["authorization_id"]

            id_state = _mk_id_state(
                1, IdentityState.IDENTIFIED, person_id=p_target, display_name="Target Person"
            )
            sm.process_frame_result(_mk_frame(primary_track_id=1, identity_states={1: id_state}))

            curr = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert curr.status_code == 200
            assert curr.json()["status"] == "authorized"
            assert curr.json()["person_id"] == p_target

            # 3. Delete unrelated person: session must NOT be cancelled
            del_unrelated = client.delete(f"/persons/{p_unrelated}", headers=headers)
            assert del_unrelated.status_code == 204

            curr_after_unrelated = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert curr_after_unrelated.status_code == 200
            assert curr_after_unrelated.json()["status"] == "authorized"
            assert curr_after_unrelated.json()["person_id"] == p_target

            # 4. Delete target person: matching session must be cancelled
            del_target = client.delete(f"/persons/{p_target}", headers=headers)
            assert del_target.status_code == 204

            curr_after_target = client.get(_AUTH_CURRENT_PATH, headers=headers)
            assert curr_after_target.status_code == 200
            assert curr_after_target.json()["status"] == "cancelled"

            # 5. Privileged context is inaccessible
            ctx_resp = client.get(f"/authorization/sessions/{auth_id}/context", headers=headers)
            assert ctx_resp.status_code == 403
