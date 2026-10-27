"""Unit tests for dh360-hi serve CLI dialogue composition root and modes."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
from dh360_hi.cli.main import main
from dh360_hi.contracts.dialogue import MockLLMProvider
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.integrations.eliva import (
    ElivaGeminiAdapter,
    ElivaGoogleTTSAdapter,
    ElivaSonioxAdapter,
)


def _setup_service_mock() -> mock.MagicMock:
    """Create a mock HumanInteractionService with necessary properties."""
    mock_svc = mock.MagicMock()
    mock_runtime = mock.MagicMock()
    mock_sm = mock.MagicMock()
    mock_runtime.session_manager = mock_sm
    mock_svc.runtime = mock_runtime
    mock_svc.event_bus = EventBus()
    mock_svc.person_service = mock.MagicMock()
    return mock_svc


def test_serve_auto_mode_without_gemini_key_wires_mock_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When GEMINI_API_KEY is unset, auto mode defaults to MockLLMProvider."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    db_file = str(tmp_path / "serve_dialogue.db")
    mock_svc = _setup_service_mock()

    with (
        mock.patch("uvicorn.run") as _mock_uvicorn,
        mock.patch("dh360_hi.cli.serve.create_interaction_service", return_value=mock_svc),
        mock.patch("dh360_hi.cli.serve.create_app") as mock_create_app,
    ):
        mock_create_app.return_value = mock.MagicMock()

        exit_code = main(["serve", "--source", "synthetic", "--db-path", db_file])
        assert exit_code == 0

        mock_create_app.assert_called_once()
        passed_coord = mock_create_app.call_args[1]["dialogue_coordinator"]
        assert isinstance(passed_coord, DialogueCoordinator)
        assert isinstance(passed_coord._llm_provider, MockLLMProvider)

        # Confirm turn execution returns explicit [MOCK] prefix
        turn = passed_coord.handle_turn(user_text="What is this robot?", language="en")
        assert turn.text.startswith("[MOCK]")


def test_serve_explicit_mock_flag_activates_mock_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing --mock-dialogue activates MockLLMProvider even if key is present."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    db_file = str(tmp_path / "serve_dialogue_mock.db")
    mock_svc = _setup_service_mock()

    with (
        mock.patch("uvicorn.run") as _mock_uvicorn,
        mock.patch("dh360_hi.cli.serve.create_interaction_service", return_value=mock_svc),
        mock.patch("dh360_hi.cli.serve.create_app") as mock_create_app,
    ):
        mock_create_app.return_value = mock.MagicMock()

        exit_code = main(
            ["serve", "--source", "synthetic", "--db-path", db_file, "--mock-dialogue"]
        )
        assert exit_code == 0

        passed_coord = mock_create_app.call_args[1]["dialogue_coordinator"]
        assert isinstance(passed_coord, DialogueCoordinator)
        assert isinstance(passed_coord._llm_provider, MockLLMProvider)

        turn = passed_coord.handle_turn(user_text="কে তুমি?", language="bn")
        assert turn.text.startswith("[MOCK]")


def test_serve_dialogue_mode_disabled(tmp_path: Path) -> None:
    """When --dialogue-mode disabled is provided, coordinator is None."""
    db_file = str(tmp_path / "serve_dialogue_disabled.db")
    mock_svc = _setup_service_mock()

    with (
        mock.patch("uvicorn.run") as _mock_uvicorn,
        mock.patch("dh360_hi.cli.serve.create_interaction_service", return_value=mock_svc),
        mock.patch("dh360_hi.cli.serve.create_app") as mock_create_app,
    ):
        mock_create_app.return_value = mock.MagicMock()

        exit_code = main(
            [
                "serve",
                "--source",
                "synthetic",
                "--db-path",
                db_file,
                "--dialogue-mode",
                "disabled",
            ]
        )
        assert exit_code == 0

        passed_coord = mock_create_app.call_args[1]["dialogue_coordinator"]
        assert passed_coord is None


def test_serve_dialogue_mode_live_wires_eliva_adapters(tmp_path: Path) -> None:
    """When --dialogue-mode live is set, Eliva cloud adapters are wired lazily."""
    db_file = str(tmp_path / "serve_dialogue_live.db")
    mock_svc = _setup_service_mock()

    with (
        mock.patch("uvicorn.run") as _mock_uvicorn,
        mock.patch("dh360_hi.cli.serve.create_interaction_service", return_value=mock_svc),
        mock.patch("dh360_hi.cli.serve.create_app") as mock_create_app,
    ):
        mock_create_app.return_value = mock.MagicMock()

        exit_code = main(
            [
                "serve",
                "--source",
                "synthetic",
                "--db-path",
                db_file,
                "--dialogue-mode",
                "live",
            ]
        )
        assert exit_code == 0

        passed_coord = mock_create_app.call_args[1]["dialogue_coordinator"]
        assert isinstance(passed_coord, DialogueCoordinator)
        assert isinstance(passed_coord._llm_provider, ElivaGeminiAdapter)
        assert isinstance(passed_coord._stt_provider, ElivaSonioxAdapter)
        assert isinstance(passed_coord._tts_provider, ElivaGoogleTTSAdapter)


def test_serve_shutdown_cleans_up_dialogue_coordinator(tmp_path: Path) -> None:
    """When the server terminates, dialogue_coordinator.close() is invoked."""
    db_file = str(tmp_path / "serve_dialogue_shutdown.db")
    mock_svc = _setup_service_mock()

    with (
        mock.patch("uvicorn.run") as _mock_uvicorn,
        mock.patch("dh360_hi.cli.serve.create_interaction_service", return_value=mock_svc),
        mock.patch("dh360_hi.cli.serve.create_app") as mock_create_app,
    ):
        mock_create_app.return_value = mock.MagicMock()

        exit_code = main(
            ["serve", "--source", "synthetic", "--db-path", db_file, "--mock-dialogue"]
        )
        assert exit_code == 0

        passed_coord = mock_create_app.call_args[1]["dialogue_coordinator"]
        # Coordinator should have been closed in the finally block
        # When closed, _session_state is None and subscriptions are empty
        assert passed_coord._subscription_ids == []


def test_wired_dialogue_endpoint_e2e_via_fastapi_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end verification of POST /dialogue/turn with wired mock coordinator in create_app."""
    from dh360_hi.api.app import create_app
    from dh360_hi.backend.knowledge_service import KnowledgeService

    token = "secret-operator-token"
    monkeypatch.setenv("DH360_HI_API_OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API_OPERATOR_SCOPES", "dialogue:interact")

    mock_svc = _setup_service_mock()
    bus = EventBus()
    knowledge_svc = KnowledgeService(person_service=mock_svc.person_service)
    coord = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_svc,
        llm_provider=MockLLMProvider(),
    )

    app = create_app(service=mock_svc, dialogue_coordinator=coord)

    with TestClient(app) as client:
        # 1. Unauthenticated request rejected
        resp_401 = client.post("/dialogue/turn", json={"text": "Hello"})
        assert resp_401.status_code == 401

        # 2. Authenticated request succeeded with explicit [MOCK] output
        resp_200 = client.post(
            "/dialogue/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "Hello Eliva", "language": "en"},
        )
        assert resp_200.status_code == 200
        data = resp_200.json()
        assert data["role"] == "model"
        assert "[MOCK]" in data["text"]
        assert "Hello Eliva" in data["text"]
