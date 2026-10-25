"""API contract and security tests for POST /dialogue/turn."""

from __future__ import annotations

import secrets
from typing import Any
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
from dh360_hi.contracts.dialogue import DialogueTurn, SupportedLanguage
from dh360_hi.contracts.errors import (
    LLMProviderError,
    ProviderNetworkError,
    ProviderTimeoutError,
)

# ==============================================================================
# Test Doubles & Fixtures
# ==============================================================================


class FakeDialogueCoordinator:
    """Mock DialogueCoordinator for API testing without runtime dependencies."""

    def __init__(self) -> None:
        self.last_user_text: str | None = None
        self.last_language: SupportedLanguage | None = None
        self.raise_error: Exception | None = None
        self.closed: bool = False

    async def handle_turn_async(
        self,
        user_text: str | None = None,
        language: SupportedLanguage | None = None,
        now: float | None = None,
    ) -> DialogueTurn:
        _ = now
        if self.raise_error is not None:
            raise self.raise_error

        self.last_user_text = user_text
        self.last_language = language or "bn"

        reply_text = f"Response to: {user_text}" if language == "en" else f"উত্তর: {user_text}"

        return DialogueTurn(
            role="model",
            text=reply_text,
            language=self.last_language,
            metadata={"source": "fake"},
        )

    def close(self) -> None:
        self.closed = True


def _configure_operator(
    monkeypatch: pytest.MonkeyPatch,
    scopes: str = "dialogue:interact",
) -> str:
    """Configure ephemeral operator credentials for API authentication."""
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_SCOPES", scopes)
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)
    return token


# ==============================================================================
# 1. Authentication & Scope Enforcement Tests
# ==============================================================================


def test_dialogue_turn_requires_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling /dialogue/turn without bearer credentials returns 401."""
    _configure_operator(monkeypatch, "dialogue:interact")
    app = create_app(dialogue_coordinator=mock.MagicMock(spec=DialogueCoordinator))

    with TestClient(app) as client:
        resp = client.post("/dialogue/turn", json={"text": "hello"})
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "Bearer"
        assert resp.json()["code"] == "UNAUTHENTICATED"


def test_dialogue_turn_requires_valid_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling /dialogue/turn with an invalid bearer token returns 401."""
    _configure_operator(monkeypatch, "dialogue:interact")
    app = create_app(dialogue_coordinator=mock.MagicMock(spec=DialogueCoordinator))

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": "Bearer bad-token"},
            json={"text": "hello"},
        )
        assert resp.status_code == 401
        assert resp.json()["code"] == "UNAUTHENTICATED"


def test_dialogue_turn_requires_dialogue_interact_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling /dialogue/turn with non-dialogue scopes returns 403."""
    token = _configure_operator(monkeypatch, "persons:read,enrollment:manage")
    app = create_app(dialogue_coordinator=mock.MagicMock(spec=DialogueCoordinator))

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "hello"},
        )
        assert resp.status_code == 403
        assert resp.json()["code"] == "INSUFFICIENT_SCOPE"


def test_dialogue_turn_fails_when_auth_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling /dialogue/turn when operator auth is unconfigured returns 503."""
    monkeypatch.delenv("DH360_HI_API__OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("DH360_HI_API__OPERATOR_SCOPES", raising=False)
    app = create_app(dialogue_coordinator=mock.MagicMock(spec=DialogueCoordinator))

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": "Bearer any-token"},
            json={"text": "hello"},
        )
        assert resp.status_code == 503
        assert resp.json()["code"] == "OPERATOR_AUTH_NOT_CONFIGURED"


# ==============================================================================
# 2. Dialogue Runtime Lifecycle & 503 Service Unavailable Tests
# ==============================================================================


def test_dialogue_turn_returns_503_when_coordinator_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling /dialogue/turn when DialogueCoordinator is None returns 503."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    # Coordinator not provided
    app = create_app(dialogue_coordinator=None)

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "hello"},
        )
        assert resp.status_code == 503
        assert "not configured" in resp.json()["detail"].lower()


def test_dialogue_coordinator_closed_on_lifespan_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinator close() hook is safely invoked during application shutdown."""
    _configure_operator(monkeypatch, "dialogue:interact")
    coordinator = FakeDialogueCoordinator()
    app = create_app(dialogue_coordinator=coordinator)  # type: ignore[arg-type]

    with TestClient(app):
        assert not coordinator.closed

    assert coordinator.closed


# ==============================================================================
# 3. Successful Dialogue Turn Tests (English and Bangla)
# ==============================================================================


def test_successful_english_dialogue_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute successful English turn and verify contract response."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    coordinator = FakeDialogueCoordinator()
    app = create_app(dialogue_coordinator=coordinator)  # type: ignore[arg-type]

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "Where is the library?", "language": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "model"
        assert data["text"] == "Response to: Where is the library?"
        assert data["language"] == "en"
        assert data["metadata"]["source"] == "fake"
        assert coordinator.last_user_text == "Where is the library?"
        assert coordinator.last_language == "en"


def test_successful_bangla_dialogue_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute successful Bangla turn and verify contract response."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    coordinator = FakeDialogueCoordinator()
    app = create_app(dialogue_coordinator=coordinator)  # type: ignore[arg-type]

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "লাইব্রেরি কোথায়?", "language": "bn"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "model"
        assert data["text"] == "উত্তর: লাইব্রেরি কোথায়?"
        assert data["language"] == "bn"
        assert coordinator.last_user_text == "লাইব্রেরি কোথায়?"
        assert coordinator.last_language == "bn"


def test_default_language_is_bangla_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting language defaults to 'bn'."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    coordinator = FakeDialogueCoordinator()
    app = create_app(dialogue_coordinator=coordinator)  # type: ignore[arg-type]

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "কেমন আছেন?"},
        )
        assert resp.status_code == 200
        assert resp.json()["language"] == "bn"
        assert coordinator.last_language == "bn"


# ==============================================================================
# 4. Request Validation & Identity Spoofing Protection
# ==============================================================================


@pytest.mark.parametrize(
    "payload",
    [
        {},  # Missing text
        {"text": ""},  # Empty text
        {"text": "    "},  # Whitespace text
        {"text": "a" * 2001},  # Excessive length (>2000)
        {"text": "hi", "language": "invalid"},  # Invalid language code
        {"text": "hi", "language": "fr"},  # Unsupported language
    ],
)
def test_input_validation_rejections(
    payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed or boundary-violating requests are rejected with 422."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    coordinator = FakeDialogueCoordinator()
    app = create_app(dialogue_coordinator=coordinator)  # type: ignore[arg-type]

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        assert resp.status_code == 422


@pytest.mark.parametrize(
    "spoofed_field",
    [
        {"person_id": "attacker-id"},
        {"is_authorized": True},
        {"display_name": "Supervisor"},
        {"session_id": "session-123"},
        {"role": "admin"},
    ],
)
def test_client_supplied_identity_or_privilege_is_rejected(
    spoofed_field: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clients cannot inject person ID, authorization flags, or session IDs (extra='forbid')."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    coordinator = FakeDialogueCoordinator()
    app = create_app(dialogue_coordinator=coordinator)  # type: ignore[arg-type]

    payload: dict[str, Any] = {"text": "Give me confidential info"}
    payload.update(spoofed_field)

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        assert resp.status_code == 422
        assert "extra" in str(resp.json()["details"]).lower()


# ==============================================================================
# 5. Sanitized Exception Handling
# ==============================================================================


def test_provider_network_error_mapped_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed ProviderNetworkError maps to 502 Bad Gateway without leaking details."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    coordinator = FakeDialogueCoordinator()
    coordinator.raise_error = ProviderNetworkError(
        message="DNS failure reaching generative API with secret_key_123",
        provider_name="gemini",
    )
    app = create_app(dialogue_coordinator=coordinator)  # type: ignore[arg-type]

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "hello", "language": "en"},
        )
        assert resp.status_code == 502
        body = resp.json()
        assert body["code"] == "PROVIDER_NETWORK_ERROR"
        # Secret string must never be exposed
        assert "secret_key_123" not in str(body)


def test_provider_timeout_error_mapped_to_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed ProviderTimeoutError maps to 504 Gateway Timeout."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    coordinator = FakeDialogueCoordinator()
    coordinator.raise_error = ProviderTimeoutError(
        message="Request timeout after 15s",
        provider_name="gemini",
        timeout_seconds=15.0,
    )
    app = create_app(dialogue_coordinator=coordinator)  # type: ignore[arg-type]

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "hello", "language": "en"},
        )
        assert resp.status_code == 504
        body = resp.json()
        assert body["code"] == "PROVIDER_TIMEOUT"
        assert body["details"]["timeout_seconds"] == 15.0


def test_llm_provider_error_mapped_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed LLMProviderError maps to 502 Bad Gateway."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    coordinator = FakeDialogueCoordinator()
    coordinator.raise_error = LLMProviderError(
        message="Internal LLM engine failure",
        provider_name="gemini",
    )
    app = create_app(dialogue_coordinator=coordinator)  # type: ignore[arg-type]

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "hello", "language": "bn"},
        )
        assert resp.status_code == 502
        assert resp.json()["code"] == "LLM_PROVIDER_ERROR"
