"""Contract tests for the M3.2.4 live interaction SSE stream.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import contextlib
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app

from dh360_hi.contracts.integration import SessionStartedIntegrationEvent

_PATH = "/interaction/events"


def _configure_operator(monkeypatch: pytest.MonkeyPatch, scopes: str) -> str:
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_SCOPES", scopes)
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)
    return token


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_missing_token_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_operator(monkeypatch, "interaction:read")
    with TestClient(create_app()) as client:
        response = client.get(_PATH)
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHENTICATED"


def test_wrong_scope_returns_403(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _configure_operator(monkeypatch, "persons:read")
    with TestClient(create_app()) as client:
        response = client.get(_PATH, headers=_auth_headers(token))
    assert response.status_code == 403


def test_auth_unavailable_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DH360_HI_API__OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("DH360_HI_API__OPERATOR_SCOPES", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)
    with TestClient(create_app()) as client:
        response = client.get(_PATH)
    assert response.status_code == 503


def test_query_token_is_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _configure_operator(monkeypatch, "interaction:read")
    with TestClient(create_app()) as client:
        response = client.get(_PATH, params={"access_token": token})
    assert response.status_code == 401


def test_stream_path_rejects_post(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _configure_operator(monkeypatch, "interaction:read")
    with TestClient(create_app()) as client:
        response = client.post(_PATH, json={"token": token})
    assert response.status_code in (401, 403, 404, 405)


def test_existing_context_endpoint_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_operator(monkeypatch, "interaction:read")
    with TestClient(create_app()) as client:
        response = client.get("/interaction/context")
    assert response.status_code == 200

class _AsgiStream:
    """Minimal direct ASGI harness for the infinite SSE response body."""

    def __init__(self, app: Any) -> None:
        self._app = app
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.chunks: list[bytes] = []
        self._received_disconnect = False
        self._published = False

    async def _receive(self) -> dict[str, Any]:
        if self._received_disconnect:
            await asyncio.sleep(0)
            return {"type": "http.disconnect"}
        await asyncio.sleep(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(self, message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            self.status = int(message["status"])
            self.headers = {
                k.decode().lower(): v.decode() for k, v in message.get("headers", [])
            }
        elif message["type"] == "http.response.body":
            body = message.get("body", b"")
            if body:
                self.chunks.append(bytes(body))

    def disconnect(self) -> None:
        self._received_disconnect = True

    def text(self) -> str:
        return b"".join(self.chunks).decode()


def _scope(token: str, last_event_id: str | None = None) -> dict[str, Any]:
    headers = [(b"authorization", f"Bearer {token}".encode())]
    if last_event_id is not None:
        headers.append((b"last-event-id", last_event_id.encode()))
    return {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": _PATH,
        "raw_path": _PATH.encode(),
        "query_string": b"",
        "headers": headers,
        "state": {},
    }


def _run_stream(
    app: Any,
    scope: dict[str, Any],
    stream: _AsgiStream,
    publish: Any | None = None,
    stop_after_chunks: int = 2,
) -> None:
    async def runner() -> None:
        task = asyncio.ensure_future(app(scope, stream._receive, stream._send))
        for _ in range(1000):
            await asyncio.sleep(0.01)
            if publish is not None and len(stream.chunks) >= 1 and not stream._published:
                stream._published = True
                publish()
            if len(stream.chunks) >= stop_after_chunks or task.done():
                break
        stream.disconnect()
        for _ in range(1000):
            if task.done():
                break
            await asyncio.sleep(0.01)
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    asyncio.run(runner())

def test_sse_initial_context_frame_via_asgi(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _configure_operator(monkeypatch, "interaction:read")
    app = create_app()
    stream = _AsgiStream(app)
    _run_stream(app, _scope(token), stream)
    assert stream.status == 200
    assert stream.headers.get("content-type", "").startswith("text/event-stream")
    assert stream.headers.get("cache-control") == "no-cache"
    body = stream.text()
    assert "retry: 3000" in body
    assert "event: context" in body
    assert '\"state\"' in body
    assert app.state.event_hub.subscriber_count == 0

def test_sse_live_delivery_and_cleanup_via_asgi(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _configure_operator(monkeypatch, "interaction:read")
    app = create_app()
    stream = _AsgiStream(app)

    def publish_two() -> None:
        hub = app.state.event_hub
        hub.publish(SessionStartedIntegrationEvent(session_id="live-1"))
        hub.publish(SessionStartedIntegrationEvent(session_id="live-2"))

    _run_stream(app, _scope(token), stream, publish=publish_two, stop_after_chunks=4)
    body = stream.text()
    assert "event: context" in body
    assert "live-1" in body
    assert "live-2" in body
    ids = [int(ln[4:]) for ln in body.splitlines() if ln.startswith("id:")]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert app.state.event_hub.subscriber_count == 0

def test_sse_replay_since_last_event_id_via_asgi(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _configure_operator(monkeypatch, "interaction:read")
    app = create_app()
    hub = app.state.event_hub
    hub.publish(SessionStartedIntegrationEvent(session_id="replay-1"))
    hub.publish(SessionStartedIntegrationEvent(session_id="replay-2"))
    hub.publish(SessionStartedIntegrationEvent(session_id="replay-3"))
    stream = _AsgiStream(app)
    _run_stream(app, _scope(token, "1"), stream)
    body = stream.text()
    assert "replay-2" in body
    assert "replay-3" in body
    assert "event: context" in body
    assert app.state.event_hub.subscriber_count == 0

def test_sse_expired_replay_emits_resync_via_asgi(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _configure_operator(monkeypatch, "interaction:read")
    app = create_app()
    hub = app.state.event_hub
    for i in range(70):
        hub.publish(SessionStartedIntegrationEvent(session_id=f"ev-{i}"))
    stream = _AsgiStream(app)
    _run_stream(app, _scope(token, "1"), stream)
    body = stream.text()
    assert "event: resync" in body
    assert app.state.event_hub.subscriber_count == 0

def test_sse_payloads_contain_no_internal_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _configure_operator(monkeypatch, "interaction:read")
    app = create_app()
    stream = _AsgiStream(app)

    def publish_one() -> None:
        app.state.event_hub.publish(SessionStartedIntegrationEvent(session_id="priv-1"))

    _run_stream(app, _scope(token), stream, publish=publish_one, stop_after_chunks=3)
    lowered = stream.text().lower()
    for forbidden in ("track_id", "frame_id", "embedding", "landmark", "bbox"):
        assert forbidden not in lowered
    for line in stream.text().splitlines():
        if line.startswith("data:"):
            payload = json.loads(line[len("data:"):])
            assert "track_id" not in payload
            assert "frame_id" not in payload

