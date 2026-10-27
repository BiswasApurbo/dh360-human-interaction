"""Unit tests for voice dialogue API endpoints (/dialogue/turn/voice and /dialogue/interrupt)."""

from __future__ import annotations

import secrets
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.contracts.dialogue import DialogueTurn, SupportedLanguage


class VoiceFakeDialogueCoordinator:
    """Mock DialogueCoordinator recording voice turns and testing speech flows."""

    def __init__(self, default_response: str = "This is a voice response.") -> None:
        self.default_response = default_response
        self.last_user_text: str | None = None
        self.last_language: SupportedLanguage | None = None
        self.last_speak_response: bool | None = None
        self.raise_error: Exception | None = None
        self.interrupted: bool = False
        self.closed: bool = False

    async def handle_turn_async(
        self,
        user_text: str | None = None,
        language: SupportedLanguage | None = None,
        speak_response: bool = True,
        now: float | None = None,
    ) -> DialogueTurn:
        _ = now
        if self.raise_error is not None:
            raise self.raise_error

        self.last_user_text = user_text
        self.last_language = language or "bn"
        self.last_speak_response = speak_response

        recognized_prompt = user_text if user_text is not None else "স্পোকেন প্রশ্ন"
        metadata: dict[str, Any] = {
            "source": "fake_voice",
            "user_text": recognized_prompt,
        }
        if user_text is None:
            metadata["is_voice"] = True

        return DialogueTurn(
            role="model",
            text=self.default_response,
            language=self.last_language,
            metadata=metadata,
        )

    def interrupt(self) -> None:
        self.interrupted = True

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
# 1. Voice Turn Authentication & Security Tests
# ==============================================================================


def test_voice_turn_requires_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling /dialogue/turn/voice without credentials returns 401."""
    _configure_operator(monkeypatch, "dialogue:interact")
    app = create_app(dialogue_coordinator=VoiceFakeDialogueCoordinator())

    with TestClient(app) as client:
        resp = client.post("/dialogue/turn/voice", json={})
        assert resp.status_code == 401
        assert resp.json()["code"] == "UNAUTHENTICATED"


def test_voice_turn_requires_dialogue_interact_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling /dialogue/turn/voice without dialogue:interact returns 403."""
    token = _configure_operator(monkeypatch, "persons:read")
    app = create_app(dialogue_coordinator=VoiceFakeDialogueCoordinator())

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn/voice",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )
        assert resp.status_code == 403
        assert resp.json()["code"] == "INSUFFICIENT_SCOPE"


def test_voice_turn_extra_fields_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spoofed client fields on voice turn are rejected with 422."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    app = create_app(dialogue_coordinator=VoiceFakeDialogueCoordinator())

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn/voice",
            headers={"Authorization": f"Bearer {token}"},
            json={"language": "en", "is_authorized": True},
        )
        assert resp.status_code == 422


# ==============================================================================
# 2. Voice Turn Execution & Response Contract
# ==============================================================================


def test_voice_turn_executes_successfully(monkeypatch: pytest.MonkeyPatch) -> None:
    """Executing /dialogue/turn/voice returns recognized user_text and model answer."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    coord = VoiceFakeDialogueCoordinator(default_response="Voice answer text.")
    app = create_app(dialogue_coordinator=coord)

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn/voice",
            headers={"Authorization": f"Bearer {token}"},
            json={"language": "bn", "speak_response": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_text"] == "স্পোকেন প্রশ্ন"
        assert data["role"] == "model"
        assert data["text"] == "Voice answer text."
        assert data["language"] == "bn"
        assert data["metadata"]["is_voice"] is True
        assert coord.last_speak_response is True


def test_voice_turn_with_english_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """Voice turn in English correctly passes language hint."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    coord = VoiceFakeDialogueCoordinator()
    app = create_app(dialogue_coordinator=coord)

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/turn/voice",
            headers={"Authorization": f"Bearer {token}"},
            json={"language": "en", "speak_response": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "en"
        assert coord.last_language == "en"
        assert coord.last_speak_response is False


# ==============================================================================
# 3. Interrupt Endpoint Tests
# ==============================================================================


def test_interrupt_requires_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling /dialogue/interrupt without credentials returns 401."""
    _configure_operator(monkeypatch, "dialogue:interact")
    app = create_app(dialogue_coordinator=VoiceFakeDialogueCoordinator())

    with TestClient(app) as client:
        resp = client.post("/dialogue/interrupt")
        assert resp.status_code == 401


def test_interrupt_successfully_aborts_dialogue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling /dialogue/interrupt signals coordinator.interrupt()."""
    token = _configure_operator(monkeypatch, "dialogue:interact")
    coord = VoiceFakeDialogueCoordinator()
    app = create_app(dialogue_coordinator=coord)

    with TestClient(app) as client:
        resp = client.post(
            "/dialogue/interrupt",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "interrupted"
        assert coord.interrupted is True
