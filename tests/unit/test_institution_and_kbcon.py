"""Unit tests for KBCON identity, institution mode configuration, and accurate profile handling."""

from __future__ import annotations

import sqlite3
import secrets
from unittest import mock
import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.api.security import OperatorScope
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.backend.person_service import PersonContextDTO, PersonService, QAPair
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.contracts.authorization import AuthorizationSession, AuthorizationStatus
from dh360_hi.integrations.eliva.adapters import (
    build_kbcon_system_instruction,
)


@pytest.fixture
def memory_db_manager() -> DatabaseManager:
    """In-memory DatabaseManager initialized with all migrations including v4."""
    return DatabaseManager(":memory:", auto_init=True)


@pytest.fixture
def person_service(memory_db_manager: DatabaseManager) -> PersonService:
    repo = SQLitePersonRepository(memory_db_manager)
    return PersonService(repo)


def test_schema_v4_migration_and_settings_persistence(memory_db_manager: DatabaseManager) -> None:
    """Verify system_settings table exists and persists values across connection queries."""
    repo = SQLitePersonRepository(memory_db_manager)
    assert memory_db_manager.get_schema_version() == 4

    # Initial state is default
    assert repo.get_setting("institution_type") is None
    assert repo.get_setting("institution_type", "university") == "university"

    # Set and update settings
    repo.set_setting("institution_type", "clinic")
    repo.set_setting("institution_name", "City Care Hospital")

    assert repo.get_setting("institution_type") == "clinic"
    assert repo.get_setting("institution_name") == "City Care Hospital"

    # Overwrite setting
    repo.set_setting("institution_type", "university")
    assert repo.get_setting("institution_type") == "university"


def test_institution_filtering_university_vs_clinic(person_service: PersonService) -> None:
    """Verify strict domain isolation between university and clinic facts with zero cross-contamination."""
    # 1. University Mode (Default)
    ks = KnowledgeService(person_service=person_service, institution_type="university")
    assert ks.institution_type == "university"

    # In university mode: Akib, library, admissions should match
    akib_matches = ks.get_public_facts("Who is Akib?", language="en")
    assert any("Akib" in m for m in akib_matches)

    library_matches = ks.get_public_facts("Where is the central library?", language="en")
    assert any("library" in m.lower() for m in library_matches)

    # In university mode: Clinic facts MUST NOT be returned!
    clinic_matches = ks.get_public_facts("What are the hospital visiting hours?", language="en")
    assert not any("visiting hours" in m.lower() for m in clinic_matches)

    # 2. Switch to Clinic Mode
    ks.set_institution("clinic", "Metropolitan Clinic")
    assert ks.institution_type == "clinic"
    assert ks.institution_name == "Metropolitan Clinic"

    # Persistent setting updated
    assert person_service.get_setting("institution_type") == "clinic"

    # In clinic mode: Visiting hours, OPD, pharmacy match
    clinic_matches = ks.get_public_facts("What are the hospital visiting hours?", language="en")
    assert any("visiting hours" in m.lower() for m in clinic_matches)

    pharmacy_matches = ks.get_public_facts("Where is the hospital pharmacy located?", language="en")
    assert any("pharmacy" in m.lower() for m in pharmacy_matches)

    # In clinic mode: University facts MUST NOT be returned!
    akib_matches_clinic = ks.get_public_facts("Who is Akib?", language="en")
    assert not any("Akib" in m for m in akib_matches_clinic)

    # General facts (KBCON capabilities) are accessible in both modes
    general_matches = ks.get_public_facts("What can the robot KBCON do?", language="en")
    assert any("KBCON" in m for m in general_matches)


def test_kbcon_identity_and_purpose_resolution(person_service: PersonService) -> None:
    """Verify deterministic KBCON identity and purpose responses in English and Bangla across modes."""
    ks = KnowledgeService(person_service=person_service, institution_type="university")

    # English University Identity & Purpose
    id_en = ks.resolve_direct_kbcon_inquiry("Who are you?", language="en")
    assert id_en is not None
    assert "KBCON" in id_en
    assert "university" in id_en.lower()
    assert "Eliva" not in id_en

    purpose_en = ks.resolve_direct_kbcon_inquiry("What is your purpose?", language="en")
    assert purpose_en is not None
    assert "university" in purpose_en.lower()

    # Bangla University Identity & Purpose
    id_bn = ks.resolve_direct_kbcon_inquiry("তুমি কে?", language="bn")
    assert id_bn is not None
    assert "কেবিসিওএন" in id_bn or "KBCON" in id_bn
    assert "বিশ্ববিদ্যালয়" in id_bn
    assert "এলিভা" not in id_bn

    purpose_bn = ks.resolve_direct_kbcon_inquiry("তোমার উদ্দেশ্য কি?", language="bn")
    assert purpose_bn is not None
    assert "বিশ্ববিদ্যালয়" in purpose_bn

    # Banglish query
    id_banglish = ks.resolve_direct_kbcon_inquiry("tumi ke?", language="bn")
    assert id_banglish is not None
    assert "কেবিসিওএন" in id_banglish

    # Switch to Clinic Mode
    ks.set_institution("clinic", "HealthCare Plus")

    # English Clinic Identity & Purpose
    id_clinic_en = ks.resolve_direct_kbcon_inquiry("Who are you?", language="en")
    assert id_clinic_en is not None
    assert "healthcare facility" in id_clinic_en.lower() or "HealthCare Plus" in id_clinic_en

    purpose_clinic_en = ks.resolve_direct_kbcon_inquiry("What do you do?", language="en")
    assert purpose_clinic_en is not None
    assert "outpatient" in purpose_clinic_en.lower() or "emergency" in purpose_clinic_en.lower()

    # Bangla Clinic Identity & Purpose
    id_clinic_bn = ks.resolve_direct_kbcon_inquiry("তুমি কে?", language="bn")
    assert id_clinic_bn is not None
    assert "স্বাস্থ্যসেবা" in id_clinic_bn


def test_structured_profile_resolution_populated_vs_missing_fields(
    person_service: PersonService,
) -> None:
    """Verify accurate factual profile retrieval for populated (Apurbo) and missing (Arif) profiles."""
    ks = KnowledgeService(person_service=person_service)

    # Context 1: Fully populated profile (Apurbo)
    apurbo_context = PersonContextDTO(
        person_id="apurbo-uuid",
        display_name="Apurbo",
        preferred_name="Biswas",
        role="Cardiologist",
        department="Surgery",
        office_room="Room 201, Floor 2",
        working_hours="Satarday to Thursday, 10 to 6",
        profile_description="Clinical lead with robotics knowledge",
        qa_pairs=[],
    )

    # Context 2: Minimal profile with missing optional fields (Arif)
    arif_context = PersonContextDTO(
        person_id="arif-uuid",
        display_name="Arif",
        preferred_name=None,
        role=None,
        department=None,
        office_room=None,
        working_hours=None,
        profile_description=None,
        qa_pairs=[],
    )

    # --- Test Apurbo (Populated Fields) ---
    # Role query
    role_res = ks.resolve_structured_profile_inquiry("আমার পদবি কি?", apurbo_context, language="bn")
    assert role_res == "আপনার পদবি হলো Cardiologist।"

    # Room query
    room_res = ks.resolve_structured_profile_inquiry("আমার রুম কত?", apurbo_context, language="bn")
    assert room_res == "আপনার অফিস রুম হলো Room 201, Floor 2।"

    # Schedule query
    sched_res = ks.resolve_structured_profile_inquiry("আমার কাজের সময় কি?", apurbo_context, language="bn")
    assert sched_res == "আপনার কাজের সময়সূচি হলো Satarday to Thursday, 10 to 6।"

    # Full profile query
    prof_res = ks.resolve_structured_profile_inquiry("আমার প্রোফাইলের তথ্য বলো", apurbo_context, language="bn")
    assert prof_res is not None
    assert "Cardiologist" in prof_res
    assert "Surgery" in prof_res
    assert "Room 201, Floor 2" in prof_res
    assert "Satarday to Thursday, 10 to 6" in prof_res

    # --- Test Arif (Missing Fields Explicit Handling) ---
    # Role query: Clearly indicates not recorded
    arif_role_res = ks.resolve_structured_profile_inquiry("আমার পদবি কি?", arif_context, language="bn")
    assert arif_role_res == "আপনার প্রোফাইলে কোনো পদবি উল্লেখ করা নেই।"

    # Room query: Clearly indicates not recorded
    arif_room_res = ks.resolve_structured_profile_inquiry("আমার রুম কত?", arif_context, language="bn")
    assert arif_room_res == "আপনার প্রোফাইলে কোনো অফিস রুম উল্লেখ করা নেই।"

    # Schedule query: Clearly indicates not recorded
    arif_sched_res = ks.resolve_structured_profile_inquiry("আমার কাজের সময় কি?", arif_context, language="bn")
    assert arif_sched_res == "আপনার প্রোফাইলে কোনো কাজের সময়সূচি উল্লেখ করা নেই।"

    # Full profile query: Explicitly shows unrecorded status for missing fields
    arif_prof_res = ks.resolve_structured_profile_inquiry("আমার প্রোফাইলের তথ্য বলো", arif_context, language="bn")
    assert arif_prof_res is not None
    assert "Arif" in arif_prof_res
    assert "প্রোফাইলে কোনো পদবি উল্লেখ নেই" in arif_prof_res
    assert "প্রোফাইলে কোনো অফিস রুম উল্লেখ নেই" in arif_prof_res


def test_privileged_context_includes_all_fields_on_identity_inquiry(
    person_service: PersonService,
) -> None:
    """Verify get_privileged_context includes office_room and working_hours during profile inquiries."""
    ks = KnowledgeService(person_service=person_service)

    # Register Apurbo in person service
    profile = person_service.create_person(
        display_name="Apurbo",
        preferred_name="Biswas",
        role="Cardiologist",
        department="Surgery",
        office_room="Room 201, Floor 2",
        working_hours="Satarday to Thursday, 10 to 6",
        profile_description="Clinical lead with robotics knowledge",
    )

    auth = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=profile.person_id,
        display_name="Apurbo",
    )

    # Query for profile info in Bangla
    context_facts = ks.get_privileged_context(
        requested_person_id=profile.person_id,
        auth_session=auth,
        language="bn",
        query="আমার প্রোফাইলের তথ্য বলো",
    )

    context_str = "\n".join(context_facts)
    assert "ব্যক্তির নাম: Biswas" in context_str
    assert "পদবি: Cardiologist" in context_str
    assert "বিভাগ: Surgery" in context_str
    assert "অফিস রুম: Room 201, Floor 2" in context_str
    assert "কর্মঘণ্টা: Satarday to Thursday, 10 to 6" in context_str


def test_system_prompt_generator_kbcon_rules() -> None:
    """Verify build_kbcon_system_instruction strictly builds KBCON identity and domain boundaries."""
    # University prompt
    univ_prompt = build_kbcon_system_instruction(institution_type="university", institution_name="BUBT")
    assert "You are KBCON" in univ_prompt
    assert "BUBT" in univ_prompt
    assert "Eliva" not in univ_prompt
    assert "Bangladesh Innovation Fair" not in univ_prompt
    assert "UNIVERSITY GUIDELINES" in univ_prompt

    # Clinic prompt
    clinic_prompt = build_kbcon_system_instruction(institution_type="clinic", institution_name="City Hospital")
    assert "You are KBCON" in clinic_prompt
    assert "City Hospital" in clinic_prompt
    assert "HEALTHCARE CLINIC GUIDELINES" in clinic_prompt
    assert "Never provide definitive medical diagnoses" in clinic_prompt


def test_api_settings_institution_endpoints(
    person_service: PersonService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify GET and PUT /settings/institution API endpoints with operator authentication."""
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv(
        "DH360_HI_API__OPERATOR_SCOPES",
        "settings:read,settings:manage",
    )
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)

    ks = KnowledgeService(person_service=person_service)
    app = create_app(knowledge_service=ks)

    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {token}"}

        # 1. GET initial settings
        get_res = client.get("/settings/institution", headers=headers)
        assert get_res.status_code == 200
        assert get_res.json()["institution_type"] == "university"

        # 2. PUT update settings to clinic
        put_res = client.put(
            "/settings/institution",
            json={"institution_type": "clinic", "institution_name": "KBCON General Hospital"},
            headers=headers,
        )
        assert put_res.status_code == 200
        data = put_res.json()
        assert data["institution_type"] == "clinic"
        assert data["institution_name"] == "KBCON General Hospital"
        assert ks.institution_type == "clinic"

        # 3. PUT invalid institution_type
        bad_res = client.put(
            "/settings/institution",
            json={"institution_type": "invalid_mode"},
            headers=headers,
        )
        assert bad_res.status_code == 422
