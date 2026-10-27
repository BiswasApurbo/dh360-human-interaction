"""Unit and integration tests for Eliva knowledge retrieval, Gemini, and authorization.

Covers:
1. General question with no matching CSV entry receives a generated answer.
2. Company question uses relevant CSV context.
3. Registered person's actual profile and Q&A pairs can be retrieved during valid authorization.
4. Personal information remains blocked without authorization.
5. Authorized session survives ordinary pauses and multiple questions.
6. Logout revokes authorization on the backend.
7. Personal questions are blocked immediately after logout.
8. General questions continue working after logout.
9. Identity changes cannot transfer private access between people.
10. Pending responses cannot leak private information after logout or revocation.
11. Private context from one person is never supplied to another person's request.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.application.config import ApiConfig
from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.backend.person_service import PersonContextDTO, QAPair
from dh360_hi.contracts.authorization import AuthorizationSession, AuthorizationStatus
from dh360_hi.contracts.dialogue import (
    DialogueTurn,
    LLMProvider,
    SupportedLanguage,
)
from dh360_hi.contracts.events import PersonIdentifiedEvent
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.integrations.eliva.adapters import ElivaGeminiAdapter


class SpyLLMProvider(LLMProvider):
    """Test LLM provider that captures assembled context and returns predictable responses."""

    def __init__(self, response_text: str = "Generated response") -> None:
        self.response_text = response_text
        self.last_prompt: str | None = None
        self.last_context: str | None = None
        self.last_language: str | None = None
        self.call_count: int = 0

    def generate_response(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_context = filtered_context
        self.last_language = language
        return self.response_text


class MockAuthController:
    """Mock AuthorizationController for testing state transitions."""

    def __init__(self, initial_session: AuthorizationSession | None = None) -> None:
        self.current_session = initial_session

    def get_current(self) -> AuthorizationSession | None:
        return self.current_session

    def cancel(self) -> AuthorizationSession:
        if self.current_session is None or not self.current_session.is_active:
            raise ValueError("No active authorization session to cancel")
        self.current_session = self.current_session.complete(AuthorizationStatus.CANCELLED)
        return self.current_session


@pytest.fixture
def sample_apurbo_context() -> PersonContextDTO:
    return PersonContextDTO(
        person_id="p-apurbo-uuid",
        display_name="Apurbo",
        preferred_name="Apurbo",
        role="Lead Robotics Engineer",
        department="Robotech Valley R&D",
        office_room="Lab 301",
        working_hours="10 AM - 6 PM",
        profile_description="Apurbo is leading the development of the Eliva humanoid robot system.",
        qa_pairs=[
            QAPair(question="What is your favorite project?", answer="Eliva Robot"),
            QAPair(question="What is your emergency contact?", answer="01700000000"),
        ],
    )


@pytest.fixture
def sample_bob_context() -> PersonContextDTO:
    return PersonContextDTO(
        person_id="p-bob-uuid",
        display_name="Bob",
        preferred_name="Bob",
        role="System Analyst",
        department="Operations",
        office_room="Room 102",
        working_hours="9 AM - 5 PM",
        profile_description="Bob manages systems operations.",
        qa_pairs=[
            QAPair(question="What is your shift?", answer="Morning shift"),
        ],
    )


# -----------------------------------------------------------------------------
# 1 & 2: General & Company Knowledge
# -----------------------------------------------------------------------------


def test_general_question_without_csv_match_receives_generated_answer() -> None:
    """Requirement 1: General questions receive generated answer and do not force CSV refusal."""
    adapter = ElivaGeminiAdapter(generator_fn=lambda prompt, *_: f"Direct answer to '{prompt}'")
    # No CSV context
    response = adapter.generate_response(
        prompt="What is the distance to Mars?",
        filtered_context="",
        language="en",
    )
    assert response == "Direct answer to 'What is the distance to Mars?'"
    assert "I do not have that information yet" not in response


def test_company_question_uses_relevant_csv_context() -> None:
    """Requirement 2: Robotech Valley / company questions supply relevant CSV context."""
    person_svc = mock.MagicMock()
    ks = KnowledgeService(person_service=person_svc)

    # Query matching knowledge.csv entries
    context = ks.build_knowledge_context(
        query="Robotech Valley কি ধরনের প্রতিষ্ঠান?",
        language="bn",
    )
    assert context != ""
    assert "রোবোটেক ভ্যালি" in context or "Robotech Valley" in context


def test_gemini_system_instruction_allows_general_and_requires_company_transparency() -> None:
    """Requirement 1b: Instructions allow general queries and enforce transparency."""
    from dh360_hi.integrations.eliva.adapters import DEFAULT_ELIVA_SYSTEM_INSTRUCTION

    inst = DEFAULT_ELIVA_SYSTEM_INSTRUCTION
    assert "General Questions & World Knowledge" in inst
    assert "Never refuse general questions" in inst
    assert "Company-Specific Uncertainty" in inst
    assert "instead of inventing facts" in inst
    assert "Never use this disclaimer for general world questions" in inst
    assert "plain text complete sentences" in inst
    assert "Never use markdown formatting" in inst


# -----------------------------------------------------------------------------
# 3 & 4: Registered Person Context Retrieval & Authorization Gate
# -----------------------------------------------------------------------------


def test_registered_person_profile_and_qa_retrieved_during_valid_authorization(
    sample_apurbo_context: PersonContextDTO,
) -> None:
    """Requirement 3: Stored profile and Q&A pairs retrieved during valid authorization."""
    person_svc = mock.MagicMock()
    person_svc.get_person_context.return_value = sample_apurbo_context
    ks = KnowledgeService(person_service=person_svc)

    auth_sess = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id="p-apurbo-uuid",
        display_name="Apurbo",
    )

    # 1. Identity inquiry ("Who is Apurbo?")
    ctx_who = ks.get_privileged_context(
        "p-apurbo-uuid", auth_sess, language="en", query="Who is Apurbo?"
    )
    joined_who = "\n".join(ctx_who)
    assert "Person Name: Apurbo" in joined_who
    assert "Lead Robotics Engineer" in joined_who
    assert "Robotech Valley R&D" in joined_who
    assert "Apurbo is leading the development" in joined_who

    # 2. Self identity inquiry ("Who am I?")
    ctx_self = ks.get_privileged_context(
        "p-apurbo-uuid", auth_sess, language="en", query="Who am I?"
    )
    joined_self = "\n".join(ctx_self)
    assert "Person Name: Apurbo" in joined_self
    assert "Lead Robotics Engineer" in joined_self

    # 3. Specific role query ("What is my role?")
    ctx_role = ks.get_privileged_context(
        "p-apurbo-uuid", auth_sess, language="en", query="What is my role?"
    )
    joined_role = "\n".join(ctx_role)
    assert "Role: Lead Robotics Engineer" in joined_role

    # 4. Registered Q&A query ("What is your favorite project?")
    ctx_qa = ks.get_privileged_context(
        "p-apurbo-uuid", auth_sess, language="en", query="What is your favorite project?"
    )
    joined_qa = "\n".join(ctx_qa)
    assert "Eliva Robot" in joined_qa


def test_personal_information_blocked_without_authorization(
    sample_apurbo_context: PersonContextDTO,
) -> None:
    """Requirement 4: Privileged context remains completely blocked when unauthenticated."""
    person_svc = mock.MagicMock()
    person_svc.get_person_context.return_value = sample_apurbo_context
    ks = KnowledgeService(person_service=person_svc)

    # 1. None session
    ctx_none = ks.get_privileged_context(
        "p-apurbo-uuid", None, language="en", query="Who is Apurbo?"
    )
    assert ctx_none == []

    # 2. Inactive/unauthorized session
    inactive_sess = AuthorizationSession(
        status=AuthorizationStatus.ACTIVE,
    )
    ctx_inactive = ks.get_privileged_context(
        "p-apurbo-uuid", inactive_sess, language="en", query="Who am I?"
    )
    assert ctx_inactive == []

    # 3. Mismatched session (Bob's session cannot access Apurbo's context)
    mismatched_sess = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id="p-bob-uuid",
        display_name="Bob",
    )
    ctx_mismatched = ks.get_privileged_context(
        "p-apurbo-uuid", mismatched_sess, language="en", query="Who am I?"
    )
    assert ctx_mismatched == []


# -----------------------------------------------------------------------------
# 5: Authorized Session Survives Pauses and Multiple Turns
# -----------------------------------------------------------------------------


def test_authorized_session_survives_pauses_and_multiple_turns(
    sample_apurbo_context: PersonContextDTO,
) -> None:
    """Requirement 5: Session stays authorized across multiple queries and simulated pauses."""
    person_svc = mock.MagicMock()
    person_svc.get_person_context.return_value = sample_apurbo_context
    ks = KnowledgeService(person_service=person_svc)

    auth_sess = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id="p-apurbo-uuid",
        display_name="Apurbo",
    )
    auth_ctrl = MockAuthController(initial_session=auth_sess)
    spy_llm = SpyLLMProvider()

    coordinator = DialogueCoordinator(
        event_bus=EventBus(),
        knowledge_service=ks,
        authorization_controller=auth_ctrl,
        llm_provider=spy_llm,
    )

    t0 = 1000.0
    # Turn 1
    t1 = coordinator.handle_turn("Who am I?", language="en", now=t0)
    assert t1.role == "model"
    assert spy_llm.last_context is not None
    assert "Apurbo" in spy_llm.last_context

    # Simulate pause (e.g. 45 seconds later, which would have tripped a 30s timer)
    t2 = coordinator.handle_turn("What is my role?", language="en", now=t0 + 45.0)
    assert t2.role == "model"
    assert spy_llm.last_context is not None
    assert "Lead Robotics Engineer" in spy_llm.last_context

    # Turn 3 (another question 30 seconds later)
    t3 = coordinator.handle_turn("What is your favorite project?", language="en", now=t0 + 75.0)
    assert t3.role == "model"
    assert spy_llm.last_context is not None
    assert "Eliva Robot" in spy_llm.last_context


# -----------------------------------------------------------------------------
# 6, 7 & 8: Logout Revocation, Personal Blocking, and General Continuity
# -----------------------------------------------------------------------------


def test_logout_revokes_authorization_blocks_personal_and_keeps_general_working(
    sample_apurbo_context: PersonContextDTO,
) -> None:
    """Requirements 6, 7, 8: Logout revokes auth, blocks personal queries, keeps general."""
    person_svc = mock.MagicMock()
    person_svc.get_person_context.return_value = sample_apurbo_context
    ks = KnowledgeService(person_service=person_svc)

    auth_sess = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id="p-apurbo-uuid",
        display_name="Apurbo",
    )
    auth_ctrl = MockAuthController(initial_session=auth_sess)
    spy_llm = SpyLLMProvider()

    coordinator = DialogueCoordinator(
        event_bus=EventBus(),
        knowledge_service=ks,
        authorization_controller=auth_ctrl,
        llm_provider=spy_llm,
    )

    # 1. While authorized, personal question works
    coordinator.handle_turn("What is my role?", language="en")
    assert spy_llm.last_context is not None
    assert "Lead Robotics Engineer" in spy_llm.last_context

    # 2. Trigger logout: backend revokes auth session and coordinator resets
    auth_ctrl.cancel()
    coordinator.revoke_authorization()

    # Verify session state is deauthorized
    assert coordinator.session_state is not None
    assert coordinator.session_state.is_authorized is False
    assert coordinator.session_state.person_id is None

    # 3. Personal question after logout is blocked
    coordinator.handle_turn("What is my role?", language="en")
    assert spy_llm.last_context is not None
    assert "Lead Robotics Engineer" not in spy_llm.last_context

    # 4. General question after logout continues working
    turn_gen = coordinator.handle_turn("What is the capital of Bangladesh?", language="en")
    assert turn_gen.role == "model"
    assert turn_gen.text == "Generated response"


# -----------------------------------------------------------------------------
# 9 & 11: Identity Changes and Cross-Person Isolation
# -----------------------------------------------------------------------------


def test_identity_change_prevents_cross_person_leakage(
    sample_apurbo_context: PersonContextDTO,
    sample_bob_context: PersonContextDTO,
) -> None:
    """Requirements 9 & 11: Identity change purges prior history and isolates personal context."""

    def get_person_ctx(pid: str) -> PersonContextDTO | None:
        if pid == "p-apurbo-uuid":
            return sample_apurbo_context
        if pid == "p-bob-uuid":
            return sample_bob_context
        return None

    person_svc = mock.MagicMock()
    person_svc.get_person_context.side_effect = get_person_ctx
    ks = KnowledgeService(person_service=person_svc)

    auth_ctrl = MockAuthController()
    spy_llm = SpyLLMProvider()
    bus = EventBus()

    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=ks,
        authorization_controller=auth_ctrl,
        llm_provider=spy_llm,
    )

    # 1. Apurbo is authorized
    auth_ctrl.current_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id="p-apurbo-uuid",
        display_name="Apurbo",
    )
    coordinator.handle_turn("What is my role?", language="en")
    assert spy_llm.last_context is not None
    assert "Lead Robotics Engineer" in spy_llm.last_context
    assert "Bob" not in spy_llm.last_context

    # 2. Subject changes: Bob is now authorized
    auth_ctrl.current_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id="p-bob-uuid",
        display_name="Bob",
    )
    assert coordinator.session_state is not None
    bus.publish(
        PersonIdentifiedEvent(
            session_id=coordinator.session_state.session_id,
            person_id="p-bob-uuid",
            display_name="Bob",
        )
    )
    coordinator.handle_turn("What is my role?", language="en")
    assert spy_llm.last_context is not None
    assert "System Analyst" in spy_llm.last_context
    assert "Lead Robotics Engineer" not in spy_llm.last_context
    assert "Apurbo" not in spy_llm.last_context

    # 3. Bob asks about his shift
    coordinator.handle_turn("What is my shift?", language="en")
    assert spy_llm.last_context is not None
    assert "Morning shift" in spy_llm.last_context
    assert "Eliva Robot" not in spy_llm.last_context


# -----------------------------------------------------------------------------
# 10: In-Flight Response Invalidation on Logout / Revocation
# -----------------------------------------------------------------------------


def test_pending_response_suppressed_when_authorization_revoked_during_generation(
    sample_apurbo_context: PersonContextDTO,
) -> None:
    """Requirement 10: If auth revoked mid-generation, privileged answer is suppressed."""
    person_svc = mock.MagicMock()
    person_svc.get_person_context.return_value = sample_apurbo_context
    ks = KnowledgeService(person_service=person_svc)

    auth_sess = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id="p-apurbo-uuid",
        display_name="Apurbo",
    )
    auth_ctrl = MockAuthController(initial_session=auth_sess)

    # LLM generator that simulates cancellation during generation
    def generator_that_cancels(*_args: Any, **_kwargs: Any) -> str:
        auth_ctrl.cancel()  # Authorization cancelled while LLM is generating!
        return "Secret confidential response"

    coordinator = DialogueCoordinator(
        event_bus=EventBus(),
        knowledge_service=ks,
        authorization_controller=auth_ctrl,
        llm_provider=ElivaGeminiAdapter(generator_fn=generator_that_cancels),
    )

    turn = coordinator.handle_turn("What is my role?", language="en")
    # Response was suppressed because authorization was revoked mid-turn
    assert "Secret confidential response" not in turn.text
    assert turn.metadata.get("error") == "AUTHORIZATION_REVOKED"


# -----------------------------------------------------------------------------
# API Contract: DELETE /authorization/sessions/current and POST /authorization/logout
# -----------------------------------------------------------------------------


def test_api_logout_endpoints_revoke_authorization() -> None:
    """Verify DELETE /authorization/sessions/current and POST /authorization/logout revoke auth."""
    auth_sess = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id="p-apurbo-uuid",
        display_name="Apurbo",
    )
    mock_ctrl = mock.MagicMock()
    mock_ctrl.get_current.return_value = auth_sess
    mock_ctrl.cancel.return_value = auth_sess.complete(AuthorizationStatus.CANCELLED)

    mock_coord = mock.MagicMock()

    app = create_app(authorization_controller=mock_ctrl, dialogue_coordinator=mock_coord)
    app.state.operator_auth = ApiConfig(
        operator_token="test-token",
        operator_scopes=("authorization:manage",),
    )

    token = "test-token"
    with TestClient(app) as client:
        # 1. DELETE /authorization/sessions/current
        resp_del = client.delete(
            "/authorization/sessions/current",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp_del.status_code == 200
        assert resp_del.json()["status"] == "cancelled"
        mock_ctrl.cancel.assert_called_once()
        mock_coord.revoke_authorization.assert_called_once()

        # 2. POST /authorization/logout
        mock_ctrl.cancel.reset_mock()
        mock_coord.revoke_authorization.reset_mock()
        resp_post = client.post(
            "/authorization/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp_post.status_code == 200
        assert resp_post.json()["status"] == "cancelled"
        mock_ctrl.cancel.assert_called_once()
        mock_coord.revoke_authorization.assert_called_once()
