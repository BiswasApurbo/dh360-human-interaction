"""Unit tests for KBCON Robot Identity, Kiosk Display, and Scoped Knowledge."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.backend.robot_registry import RobotRegistry
from dh360_hi.contracts.authorization import AuthorizationSession, AuthorizationStatus
from dh360_hi.contracts.dialogue import MockLLMProvider
from dh360_hi.edge.event_bus import EventBus


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for robot config files."""
    config_dir = tmp_path / ".config" / "dh360"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def test_robot_registry_offline_cache_resolution(
    temp_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RobotRegistry must cleanly resolve robot identity from local cache when offline."""
    cache_file = temp_config_dir / "robot_config.json"
    cached_payload = {
        "robot_id": "KB-TEST-01",
        "name": "Research Unit 1",
        "language": "en",
        "mode": "healthcare",
        "tenant_id": "tenant-clinic-abc",
        "status": "online",
    }
    cache_file.write_text(json.dumps(cached_payload), encoding="utf-8")

    from dh360_hi.backend import robot_registry

    monkeypatch.setattr(robot_registry, "OFFLINE_CONFIG_CACHE_CANDIDATES", [cache_file])
    monkeypatch.setattr(
        robot_registry, "ROBOT_ID_FILE_CANDIDATES", [temp_config_dir / "robot.json"]
    )

    # Reset singleton instance for clean test
    RobotRegistry.reset_instance()
    registry = RobotRegistry.get_instance()

    robot = registry.get_active_robot()
    assert robot.robot_id == "KB-TEST-01"
    assert robot.name == "Research Unit 1"
    assert robot.language == "en"
    assert robot.mode == "healthcare"
    assert robot.tenant_id == "tenant-clinic-abc"


def test_knowledge_service_robot_scoping_and_isolation(tmp_path: Path) -> None:
    """Knowledge items scoped to a robot must not be leaked to a different robot."""
    csv_path = tmp_path / "knowledge.csv"
    ks = KnowledgeService(csv_path=csv_path)

    # 1. Add baseline institutional fact (robot_id is None)
    base_res = ks.add_fact(
        question_en="When is campus open?",
        answer_en="Everyday from 8am to 8pm.",
        institution_type="general",
        tenant_id="tenant-acme",
        robot_id=None,
    )
    base_id = base_res["id"]

    # 2. Add Robot A private fact
    fact_a_res = ks.add_fact(
        question_en="What is this robot model?",
        answer_en="This is Robot A (Education Specialist).",
        institution_type="general",
        tenant_id="tenant-acme",
        robot_id="ROBOT-A",
    )
    fact_a_id = fact_a_res["id"]

    # 3. Add Robot B private fact
    fact_b_res = ks.add_fact(
        question_en="What is this robot model?",
        answer_en="This is Robot B (Healthcare Specialist).",
        institution_type="general",
        tenant_id="tenant-acme",
        robot_id="ROBOT-B",
    )
    fact_b_id = fact_b_res["id"]

    # Robot A facts: should see baseline + Robot A fact, NEVER Robot B fact
    facts_robot_a = ks.get_all_facts(tenant_id="tenant-acme", robot_id="ROBOT-A")
    fact_a_ids = {f["id"] for f in facts_robot_a}
    assert base_id in fact_a_ids
    assert fact_a_id in fact_a_ids
    assert fact_b_id not in fact_a_ids

    # Robot B facts: should see baseline + Robot B fact, NEVER Robot A fact
    facts_robot_b = ks.get_all_facts(tenant_id="tenant-acme", robot_id="ROBOT-B")
    fact_b_ids = {f["id"] for f in facts_robot_b}
    assert base_id in fact_b_ids
    assert fact_b_id in fact_b_ids
    assert fact_a_id not in fact_b_ids

    # Context assembly test
    ctx_a = ks.build_knowledge_context(
        query="robot model",
        language="en",
        tenant_id="tenant-acme",
        robot_id="ROBOT-A",
    )
    assert "Robot A (Education Specialist)" in ctx_a
    assert "Robot B (Healthcare Specialist)" not in ctx_a


def test_dialogue_coordinator_mute_and_interrupt() -> None:
    """DialogueCoordinator must track mute state, interrupt audio, and suppress speech when muted."""
    mock_stt = MagicMock()
    mock_tts = MagicMock()
    mock_ks = MagicMock()
    mock_ks.build_knowledge_context.return_value = ""
    mock_ks.resolve_direct_kbcon_inquiry.return_value = None

    coordinator = DialogueCoordinator(
        event_bus=EventBus(),
        knowledge_service=mock_ks,
        stt_provider=mock_stt,
        tts_provider=mock_tts,
        llm_provider=MockLLMProvider(),
        default_language="en",
    )

    # Initial state
    assert coordinator.is_microphone_muted is False
    assert coordinator.current_state == "idle"

    # Mute microphone
    coordinator.mute_microphone()
    assert coordinator.is_microphone_muted is True
    assert coordinator.current_state == "muted"
    mock_stt.hold.assert_called_once()

    # Voice turn while muted should reject immediately without invoking STT
    turn = coordinator.handle_turn(user_text=None, language="en")
    assert turn.role == "model"
    assert turn.metadata.get("error") == "MICROPHONE_MUTED"
    mock_stt.listen_utterance.assert_not_called()

    # Unmute microphone
    coordinator.unmute_microphone()
    assert coordinator.is_microphone_muted is False
    assert coordinator.current_state == "idle"
    mock_stt.release.assert_called_once()

    # Interrupt
    coordinator.interrupt()
    mock_tts.interrupt.assert_called_once()


def test_kiosk_api_endpoints() -> None:
    """Verify FastAPI /kiosk endpoints return valid HTML, state, mute, and interrupt."""
    mock_service = MagicMock()
    mock_service.event_bus = EventBus()
    mock_service.runtime = None

    mock_ks = MagicMock()
    mock_coord = DialogueCoordinator(
        event_bus=mock_service.event_bus,
        knowledge_service=mock_ks,
        default_language="en",
    )

    auth_sess = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id="p-arif-1",
        display_name="Arif",
    )
    mock_auth = MagicMock()
    mock_auth.get_current.return_value = auth_sess

    app = create_app(
        service=mock_service,
        authorization_controller=mock_auth,
        dialogue_coordinator=mock_coord,
    )

    client = TestClient(app)

    # 1. GET /kiosk HTML page
    res_page = client.get("/kiosk")
    assert res_page.status_code == 200
    assert "KBCON Humanoid Display" in res_page.text
    assert "eyes-wrapper" in res_page.text
    assert "touch-zone-left" in res_page.text
    assert "touch-zone-right" in res_page.text

    # 2. GET /kiosk/state
    res_state = client.get("/kiosk/state")
    assert res_state.status_code == 200
    data = res_state.json()
    assert "robot_id" in data
    assert data["is_authorized"] is True
    assert data["display_name"] == "Arif"
    assert data["is_muted"] is False

    # 3. POST /kiosk/mute
    res_mute = client.get("/kiosk/state")
    assert res_mute.json()["is_muted"] is False

    res_toggle = client.post("/kiosk/mute")
    assert res_toggle.status_code == 200
    assert res_toggle.json()["is_muted"] is True
    assert mock_coord.is_microphone_muted is True

    # 4. POST /kiosk/interrupt
    res_interrupt = client.post("/kiosk/interrupt")
    assert res_interrupt.status_code == 200
    assert res_interrupt.json()["status"] == "success"
