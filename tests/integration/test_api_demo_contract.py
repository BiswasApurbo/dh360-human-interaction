"""Integration tests for the demo UI endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from setuptools.config.pyprojecttoml import apply_configuration
from setuptools.dist import Distribution

from dh360_hi.api.app import create_app
from dh360_hi.api.routers import demo as demo_router


def test_demo_endpoint_returns_html() -> None:
    """Verify GET /demo returns HTTP 200 with HTML content."""
    with TestClient(create_app()) as client:
        response = client.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Human Interaction Authorization Demo" in response.text
    assert "/authorization/sessions" in response.text
    assert "/interaction/events" in response.text


def test_demo_static_asset_is_packaged_and_servable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify setuptools package-data includes demo.html and GET /demo serves the packaged copy."""
    repo_root = Path(__file__).resolve().parents[2]
    site_packages = tmp_path / "site-packages"

    dist = apply_configuration(
        Distribution({"script_name": "setup.py"}),
        repo_root / "pyproject.toml",
    )
    build_py = dist.get_command_obj("build_py")
    build_py.build_lib = str(site_packages)  # type: ignore[attr-defined]
    build_py.ensure_finalized()
    build_py.run()

    packaged_demo_py = site_packages / "dh360_hi" / "api" / "routers" / "demo.py"
    packaged_demo_html = packaged_demo_py.parent.parent / "static" / "demo.html"

    assert packaged_demo_py.is_file()
    assert packaged_demo_html.is_file()

    monkeypatch.setattr(demo_router, "_DEMO_HTML_PATH", packaged_demo_html)

    with TestClient(create_app()) as client:
        response = client.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Human Interaction Authorization Demo" in response.text


def test_demo_endpoint_contains_chat_interface_and_safe_rendering() -> None:
    """Verify demo UI includes chat interface elements, security scopes, and safe DOM binding."""
    with TestClient(create_app()) as client:
        response = client.get("/demo")
    assert response.status_code == 200
    html = response.text

    # Structure and elements
    assert 'id="chat-section"' in html
    assert 'id="chat-language"' in html
    assert 'id="chat-access-banner"' in html
    assert 'id="chat-history"' in html
    assert 'id="chat-input"' in html
    assert 'id="chat-send-btn"' in html
    assert 'id="chat-mic-btn"' in html
    assert 'id="chat-stop-btn"' in html
    assert 'id="chat-tts-toggle"' in html
    assert 'id="chat-voice-status"' in html
    assert 'id="chat-error-box"' in html
    assert "Conversational Assistant (Eliva)" in html
    assert "dialogue:interact" in html
    assert "toggleVoiceInput" in html
    assert "stopSpeechAndListening" in html

    # Safe text rendering verification: textContent is used for user and model text
    assert "bodyDiv.textContent = text" in html
    assert "div.textContent = text" in html

    # Verify no dangerous innerHTML injection for chat text
    assert "bodyDiv.innerHTML = text" not in html
    assert "msgDiv.innerHTML = text" not in html


def test_demo_chat_turn_contract_with_mock_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify POST /dialogue/turn behaves correctly with MockLLMProvider."""
    import secrets
    from unittest import mock

    from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
    from dh360_hi.backend.knowledge_service import KnowledgeService
    from dh360_hi.contracts.dialogue import MockLLMProvider
    from dh360_hi.edge.event_bus import EventBus

    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_SCOPES", "dialogue:interact")
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)

    bus = EventBus()
    knowledge_svc = KnowledgeService(person_service=mock.MagicMock())
    coord = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_svc,
        llm_provider=MockLLMProvider(),
    )
    app = create_app(dialogue_coordinator=coord)

    with TestClient(app) as client:
        # General question in English
        resp = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "What is this robot?", "language": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "model"
        assert "[MOCK]" in data["text"]
        assert data["metadata"].get("mock") is True

        # General question in Bangla
        resp_bn = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "তুমি কে?", "language": "bn"},
        )
        assert resp_bn.status_code == 200
        data_bn = resp_bn.json()
        assert data_bn["role"] == "model"
        assert "[MOCK]" in data_bn["text"]
        assert data_bn["metadata"].get("mock") is True


def test_demo_voice_turn_contract_with_stt_and_mock_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify POST /dialogue/turn/voice and /dialogue/interrupt contracts with STT."""
    import secrets
    from unittest import mock

    from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
    from dh360_hi.backend.knowledge_service import KnowledgeService
    from dh360_hi.contracts.dialogue import MockLLMProvider
    from dh360_hi.edge.event_bus import EventBus

    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_SCOPES", "dialogue:interact")
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)

    bus = EventBus()
    knowledge_svc = KnowledgeService(person_service=mock.MagicMock())
    mock_stt = mock.MagicMock()
    mock_stt.listen_utterance.return_value = "Who is Eliva?"
    mock_tts = mock.MagicMock()

    coord = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_svc,
        llm_provider=MockLLMProvider(),
        stt_provider=mock_stt,
        tts_provider=mock_tts,
    )
    app = create_app(dialogue_coordinator=coord)

    with TestClient(app) as client:
        # Voice turn in English
        resp = client.post(
            "/dialogue/turn/voice",
            headers={"Authorization": f"Bearer {token}"},
            json={"language": "en", "speak_response": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_text"] == "Who is Eliva?"
        assert data["role"] == "model"
        assert "[MOCK]" in data["text"]
        assert data["metadata"].get("is_voice") is True
        mock_tts.speak.assert_called_once()

        # Interrupt call
        resp_intr = client.post(
            "/dialogue/interrupt",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp_intr.status_code == 200
        assert resp_intr.json()["status"] == "interrupted"
        mock_tts.interrupt.assert_called_once()
