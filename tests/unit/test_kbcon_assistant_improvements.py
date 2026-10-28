"""Unit tests for KBCON assistant personal info accuracy, language selection, and context tagging."""

from __future__ import annotations

import unittest.mock as mock
from typing import Any

import pytest

from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.backend.person_service import PersonContextDTO
from dh360_hi.contracts.authorization import (
    AuthorizationSession,
    AuthorizationStatus,
)
from dh360_hi.contracts.dialogue import DialogueTurn, LLMProvider, SupportedLanguage
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.integrations.eliva.adapters import (
    ElivaGeminiAdapter,
    build_kbcon_system_instruction,
)


class RecordingLLMProvider(LLMProvider):
    """Test LLM provider recording prompt, context, and language."""

    def __init__(self, response_text: str = "Default generated answer") -> None:
        self.response_text = response_text
        self.last_prompt: str | None = None
        self.last_context: str | None = None
        self.last_language: str | None = None

    def generate_response(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> str:
        self.last_prompt = prompt
        self.last_context = filtered_context
        self.last_language = language
        return self.response_text


@pytest.fixture
def apurbo_context() -> PersonContextDTO:
    return PersonContextDTO(
        person_id="p-apurbo-1",
        display_name="Apurbo",
        preferred_name="Apurbo",
        role="Cardiologist",
        department="Surgery",
        office_room="Room 201, Floor 2",
        working_hours="Satarday to Thursday, 10 to 6",
        profile_description="Clinical lead with robotics knowledge",
        qa_pairs=[],
    )


@pytest.fixture
def arif_context_missing_fields() -> PersonContextDTO:
    return PersonContextDTO(
        person_id="p-arif-2",
        display_name="Arif",
        preferred_name=None,
        role=None,
        department=None,
        office_room=None,
        working_hours=None,
        profile_description=None,
        qa_pairs=[],
    )


def test_structured_profile_variations_and_inflections(apurbo_context: PersonContextDTO) -> None:
    """Verify natural language variations in Bengali and English for all supported fields."""
    ks = KnowledgeService()

    # Department variations
    assert ks.resolve_structured_profile_inquiry("আমার ডিপার্টমেন্ট কোনটা?", apurbo_context, "bn") == "আপনার বিভাগ হলো Surgery।"
    assert ks.resolve_structured_profile_inquiry("আমি কোন ডিপার্টমেন্টে?", apurbo_context, "bn") == "আপনার বিভাগ হলো Surgery।"
    assert ks.resolve_structured_profile_inquiry("আমি কোন বিভাগে?", apurbo_context, "bn") == "আপনার বিভাগ হলো Surgery।"
    assert ks.resolve_structured_profile_inquiry("আমার বিভাগ কি?", apurbo_context, "bn") == "আপনার বিভাগ হলো Surgery।"
    assert ks.resolve_structured_profile_inquiry("what is my department?", apurbo_context, "en") == "Your department is Surgery."
    assert ks.resolve_structured_profile_inquiry("which department do i belong to?", apurbo_context, "en") == "Your department is Surgery."

    # Office Room and Floor variations
    assert ks.resolve_structured_profile_inquiry("আমার অফিস কোথায়?", apurbo_context, "bn") == "আপনার অফিস রুম হলো Room 201, Floor 2।"
    assert ks.resolve_structured_profile_inquiry("আমার রুম কত?", apurbo_context, "bn") == "আপনার অফিস রুম হলো Room 201, Floor 2।"
    assert ks.resolve_structured_profile_inquiry("আমার রুম নম্বর কত?", apurbo_context, "bn") == "আপনার অফিস রুম হলো Room 201, Floor 2।"
    assert ks.resolve_structured_profile_inquiry("আমার অফিস কোন ফ্লোরে?", apurbo_context, "bn") == "আপনার অফিস রুম হলো Room 201, Floor 2।"
    assert ks.resolve_structured_profile_inquiry("what floor is my office on?", apurbo_context, "en") == "Your office room is Room 201, Floor 2."
    assert ks.resolve_structured_profile_inquiry("where is my office?", apurbo_context, "en") == "Your office room is Room 201, Floor 2."

    # Role and Designation variations
    assert ks.resolve_structured_profile_inquiry("আমার পদবি কি?", apurbo_context, "bn") == "আপনার পদবি হলো Cardiologist।"
    assert ks.resolve_structured_profile_inquiry("আমার ডেজিগনেশন কি?", apurbo_context, "bn") == "আপনার পদবি হলো Cardiologist।"
    assert ks.resolve_structured_profile_inquiry("আমার পদ কি?", apurbo_context, "bn") == "আপনার পদবি হলো Cardiologist।"
    assert ks.resolve_structured_profile_inquiry("what is my role?", apurbo_context, "en") == "Your role is Cardiologist."
    assert ks.resolve_structured_profile_inquiry("what is my position?", apurbo_context, "en") == "Your role is Cardiologist."


def test_missing_profile_fields_clearly_stated(arif_context_missing_fields: PersonContextDTO) -> None:
    """Absent fields must explicitly indicate they are not recorded in both languages."""
    ks = KnowledgeService()

    assert ks.resolve_structured_profile_inquiry("আমার পদবি কি?", arif_context_missing_fields, "bn") == "আপনার প্রোফাইলে কোনো পদবি উল্লেখ করা নেই।"
    assert ks.resolve_structured_profile_inquiry("what is my role?", arif_context_missing_fields, "en") == "No role is recorded in your profile."
    assert ks.resolve_structured_profile_inquiry("আমার বিভাগ কি?", arif_context_missing_fields, "bn") == "আপনার প্রোফাইলে কোনো বিভাগ উল্লেখ করা নেই।"
    assert ks.resolve_structured_profile_inquiry("what is my department?", arif_context_missing_fields, "en") == "No department is recorded in your profile."
    assert ks.resolve_structured_profile_inquiry("আমার অফিস রুম কত?", arif_context_missing_fields, "bn") == "আপনার প্রোফাইলে কোনো অফিস রুম উল্লেখ করা নেই।"
    assert ks.resolve_structured_profile_inquiry("where is my office?", arif_context_missing_fields, "en") == "No office room is recorded in your profile."


def test_general_questions_not_hijacked_by_profile_resolution(apurbo_context: PersonContextDTO) -> None:
    """General knowledge questions must return None from profile resolution and pass to general dialogue."""
    ks = KnowledgeService()

    assert ks.resolve_structured_profile_inquiry("what is the role of mitochondria?", apurbo_context, "en") is None
    assert ks.resolve_structured_profile_inquiry("Who is Shakib Al Hasan?", apurbo_context, "en") is None
    assert ks.resolve_structured_profile_inquiry("শাকিব আল হাসান কে?", apurbo_context, "bn") is None
    assert ks.resolve_structured_profile_inquiry("where is the central library?", apurbo_context, "en") is None


def test_personal_record_badge_only_for_actual_personal_data(apurbo_context: PersonContextDTO) -> None:
    """Turn metadata is_privileged must be True ONLY when personal record was actually retrieved and used."""
    person_svc = mock.MagicMock()
    person_svc.get_person_context.return_value = apurbo_context
    ks = KnowledgeService(person_service=person_svc)

    auth_sess = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id="p-apurbo-1",
        display_name="Apurbo",
    )
    auth_ctrl = mock.MagicMock()
    auth_ctrl.get_current.return_value = auth_sess

    coordinator = DialogueCoordinator(
        event_bus=EventBus(),
        knowledge_service=ks,
        authorization_controller=auth_ctrl,
        llm_provider=RecordingLLMProvider(response_text="General answer"),
    )

    # 1. Personal inquiry -> is_privileged is True
    t_personal = coordinator.handle_turn("আমার ডিপার্টমেন্ট কোনটা?", language="bn")
    assert t_personal.role == "model"
    assert "Surgery" in t_personal.text
    assert t_personal.metadata.get("is_privileged") is True

    # 2. General inquiry while authorized -> is_privileged is FALSE
    t_general = coordinator.handle_turn("Who is Shakib Al Hasan?", language="en")
    assert t_general.role == "model"
    assert t_general.metadata.get("is_privileged") is False

    # 3. KBCON identity query while authorized -> is_privileged is FALSE
    t_identity = coordinator.handle_turn("Who are you?", language="en")
    assert t_identity.role == "model"
    assert "KBCON" in t_identity.text
    assert t_identity.metadata.get("is_privileged") is False


def test_language_selection_authoritative_and_explicit_override(apurbo_context: PersonContextDTO) -> None:
    """Language selection must be authoritative, with explicit prompt requests having precedence."""
    person_svc = mock.MagicMock()
    person_svc.get_person_context.return_value = apurbo_context
    ks = KnowledgeService(person_service=person_svc)

    auth_sess = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id="p-apurbo-1",
        display_name="Apurbo",
    )
    auth_ctrl = mock.MagicMock()
    auth_ctrl.get_current.return_value = auth_sess

    coordinator = DialogueCoordinator(
        event_bus=EventBus(),
        knowledge_service=ks,
        authorization_controller=auth_ctrl,
        llm_provider=RecordingLLMProvider(response_text="English answer"),
    )

    # English selected: answers in English even if input is in Bangla
    t_en = coordinator.handle_turn("তুমি কে?", language="en")
    assert t_en.language == "en"
    assert "I'm KBCON" in t_en.text

    # Bangla selected: answers in Bangla even if input is in English
    t_bn = coordinator.handle_turn("Who are you?", language="bn")
    assert t_bn.language == "bn"
    assert "আমি কেবিসিওএন" in t_bn.text

    # Explicit user override in prompt: selector is Bangla, but user asks "please answer in English"
    t_override = coordinator.handle_turn("তুমি কে? please answer in English", language="bn")
    assert t_override.language == "en"
    assert "I'm KBCON" in t_override.text


def test_system_prompt_instruction_respects_target_language() -> None:
    """Verify build_kbcon_system_instruction generates targeted language rules."""
    en_prompt = build_kbcon_system_instruction(institution_type="university", target_language="en")
    assert "ENGLISH as the authoritative response language" in en_prompt
    assert "clear, natural English" in en_prompt

    bn_prompt = build_kbcon_system_instruction(institution_type="university", target_language="bn")
    assert "BANGLA as the authoritative response language" in bn_prompt
    assert "clear Bangla script" in bn_prompt
