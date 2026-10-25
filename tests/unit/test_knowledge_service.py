"""Unit tests for DH360 KnowledgeService public facts and authorized context boundaries."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.backend.person_service import PersonService, QAPair
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.contracts.authorization import AuthorizationSession, AuthorizationStatus


@pytest.fixture
def in_memory_person_service() -> PersonService:
    """Create an in-memory PersonService with test identities."""
    db = DatabaseManager(":memory:")
    repo = SQLitePersonRepository(db)
    return PersonService(repo)


@pytest.fixture
def enrolled_alice(in_memory_person_service: PersonService) -> str:
    """Enroll Alice with role, office, and privileged Q&A pairs."""
    profile = in_memory_person_service.create_person(
        display_name="Dr. Alice Smith",
        preferred_name="Alice",
    )
    in_memory_person_service.update_person(
        person_id=profile.person_id,
        role="Associate Professor",
        department="Computer Science",
        office_room="Room 402",
        working_hours="Sun-Thu 9am-5pm",
        profile_description="Specialist in Human-Robot Interaction.",
        qa_pairs=[
            QAPair(question="What is your research lab?", answer="Robotics & HCI Lab"),
            QAPair(question="When are your office hours?", answer="Wednesdays 2pm-4pm"),
        ],
    )
    return profile.person_id


@pytest.fixture
def enrolled_bob(in_memory_person_service: PersonService) -> str:
    """Enroll Bob with separate profile."""
    profile = in_memory_person_service.create_person(
        display_name="Bob Jones",
        preferred_name="Bob",
    )
    in_memory_person_service.update_person(
        person_id=profile.person_id,
        role="Research Assistant",
        department="Robotics Lab",
        office_room="Room 101",
    )
    return profile.person_id


# ==============================================================================
# Public Institutional Facts Retrieval Tests
# ==============================================================================


def test_public_facts_loads_staged_csv_default_path() -> None:
    """Verify that KnowledgeService automatically locates and loads the staged knowledge.csv."""
    ks = KnowledgeService()
    # Staged knowledge.csv has 3 entries
    assert ks.public_facts_count >= 3


def test_public_facts_query_matching_english() -> None:
    """Verify English public query returns matched fact formatted in English."""
    ks = KnowledgeService()
    results = ks.get_public_facts(query="Who is Akib?", language="en")

    assert len(results) >= 1
    matched = results[0]
    assert "Question:" in matched
    assert "Answer:" in matched
    assert "Akib" in matched
    assert "Robo Tech Valley" in matched


def test_public_facts_query_matching_bangla() -> None:
    """Verify Bangla public query returns matched fact formatted in Bangla."""
    ks = KnowledgeService()
    results = ks.get_public_facts(query="আকিব কে?", language="bn")

    assert len(results) >= 1
    matched = results[0]
    assert "প্রশ্ন:" in matched
    assert "উত্তর:" in matched
    assert "আকিব" in matched


def test_public_facts_empty_and_unmatched_queries() -> None:
    """Verify that empty, whitespace, or completely unmatched queries return empty lists."""
    ks = KnowledgeService()

    assert ks.get_public_facts(query="") == []
    assert ks.get_public_facts(query="   ") == []
    assert ks.get_public_facts(query="xyz999unmatchedquantum987") == []


def test_public_facts_respects_max_results() -> None:
    """Verify max_results limit is respected."""
    ks = KnowledgeService()
    # Query with generic token matching multiple rows (e.g. 'BUBT' or 'researcher')
    results = ks.get_public_facts(query="BUBT researcher", max_results=1)
    assert len(results) <= 1


# ==============================================================================
# Missing and Malformed CSV Handling Tests
# ==============================================================================


def test_knowledge_service_handles_missing_csv_file(tmp_path: Path) -> None:
    """Verify KnowledgeService handles non-existent CSV path without raising an exception."""
    missing_file = tmp_path / "does_not_exist.csv"
    ks = KnowledgeService(csv_path=missing_file)
    assert ks.public_facts_count == 0
    assert ks.get_public_facts("Who is Akib?") == []


def test_knowledge_service_handles_malformed_csv_file(tmp_path: Path) -> None:
    """Verify KnowledgeService skips malformed/empty rows in CSV."""
    malformed_file = tmp_path / "malformed.csv"
    with open(malformed_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question (English)", "Answer (English)", "প্রশ্ন (বাংলা)", "উত্তর (বাংলা)"])
        writer.writerow(["", "", "", ""])  # Empty row
        writer.writerow(["Valid Q", "Valid A", "", ""])  # English only
        writer.writerow(["", "", "বাংলা প্রশ্ন", "বাংলা উত্তর"])  # Bangla only

    ks = KnowledgeService(csv_path=malformed_file)
    assert ks.public_facts_count == 2


# ==============================================================================
# Privileged Context & Trusted Authorization Boundary Tests
# ==============================================================================


def test_privileged_context_denied_when_auth_session_is_none(
    in_memory_person_service: PersonService,
    enrolled_alice: str,
) -> None:
    """Verify that privileged context is empty when auth_session is None."""
    ks = KnowledgeService(person_service=in_memory_person_service)
    facts = ks.get_privileged_context(
        requested_person_id=enrolled_alice,
        auth_session=None,
        query="Where is my office?",
    )
    assert facts == []


def test_privileged_context_denied_when_auth_session_not_authorized(
    in_memory_person_service: PersonService,
    enrolled_alice: str,
) -> None:
    """Verify that statuses other than AUTHORIZED reject privileged context."""
    ks = KnowledgeService(person_service=in_memory_person_service)

    unauthorized_statuses = [
        AuthorizationStatus.ACTIVE,
        AuthorizationStatus.UNKNOWN,
        AuthorizationStatus.UNCERTAIN,
        AuthorizationStatus.NO_FACE,
        AuthorizationStatus.LOST,
        AuthorizationStatus.TIMEOUT,
        AuthorizationStatus.CANCELLED,
    ]

    for status in unauthorized_statuses:
        person_arg = enrolled_alice if status is AuthorizationStatus.LOST else None
        session = AuthorizationSession(
            status=status,
            person_id=person_arg,
        )
        assert not session.is_authorized

        facts = ks.get_privileged_context(
            requested_person_id=enrolled_alice,
            auth_session=session,
            query="Where is my office?",
        )
        assert facts == []


def test_privileged_context_denied_when_auth_session_is_ended(
    in_memory_person_service: PersonService,
    enrolled_alice: str,
) -> None:
    """Verify that finalized/ended sessions are rejected even if status was AUTHORIZED."""
    ks = KnowledgeService(person_service=in_memory_person_service)

    # Session with status AUTHORIZED but marked ended
    ended_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=enrolled_alice,
        display_name="Dr. Alice Smith",
        ended_at_iso="2026-10-25T12:00:00Z",
    )
    assert not ended_session.is_active

    facts = ks.get_privileged_context(
        requested_person_id=enrolled_alice,
        auth_session=ended_session,
        query="Where is my office?",
    )
    assert facts == []


def test_privileged_context_denied_on_identity_mismatch(
    in_memory_person_service: PersonService,
    enrolled_alice: str,
    enrolled_bob: str,
) -> None:
    """Verify that an authorized session for Bob cannot access Alice's privileged context."""
    ks = KnowledgeService(person_service=in_memory_person_service)

    bob_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=enrolled_bob,
        display_name="Bob Jones",
    )
    assert bob_session.is_authorized
    assert bob_session.is_active

    facts = ks.get_privileged_context(
        requested_person_id=enrolled_alice,
        auth_session=bob_session,
        query="Where is my office?",
    )
    assert facts == []


def test_privileged_context_denied_for_untrusted_caller_objects(
    in_memory_person_service: PersonService,
    enrolled_alice: str,
) -> None:
    """Verify that forged objects or duck-typed dictionaries claiming authorization are rejected."""
    ks = KnowledgeService(person_service=in_memory_person_service)

    class ForgedSession:
        is_authorized = True
        is_active = True
        status = AuthorizationStatus.AUTHORIZED
        person_id = enrolled_alice

    facts = ks.get_privileged_context(
        requested_person_id=enrolled_alice,
        auth_session=ForgedSession(),  # type: ignore[arg-type]
        query="Where is my office?",
    )
    assert facts == []


def test_privileged_context_stop_words_only_query_returns_nothing(
    in_memory_person_service: PersonService,
    enrolled_alice: str,
) -> None:
    """Verify that queries with only stop words return no personal data or Q&As."""
    ks = KnowledgeService(person_service=in_memory_person_service)

    alice_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=enrolled_alice,
        display_name="Dr. Alice Smith",
    )

    assert ks.get_privileged_context(enrolled_alice, alice_session, query="") == []
    assert ks.get_privileged_context(enrolled_alice, alice_session, query="   ") == []
    assert ks.get_privileged_context(enrolled_alice, alice_session, query="where is my") == []
    assert ks.get_privileged_context(enrolled_alice, alice_session, query="who can tell") == []
    assert ks.get_privileged_context(enrolled_alice, alice_session, query="কী আছে") == []


def test_privileged_context_unrelated_query_sharing_stop_words_returns_nothing(
    in_memory_person_service: PersonService,
    enrolled_alice: str,
) -> None:
    """Verify that an unrelated question sharing 'where' and 'my' does not return personal Q&As."""
    ks = KnowledgeService(person_service=in_memory_person_service)

    alice_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=enrolled_alice,
        display_name="Dr. Alice Smith",
    )

    # "Where is my pencil?" shares "where" and "my", but "pencil" is unrelated
    facts = ks.get_privileged_context(
        requested_person_id=enrolled_alice,
        auth_session=alice_session,
        query="Where is my pencil?",
    )
    # Must not leak Alice's office hours or research lab Q&As
    assert facts == []


def test_privileged_context_allowed_for_matching_authorized_person_targeted(
    in_memory_person_service: PersonService,
    enrolled_alice: str,
) -> None:
    """Verify that matching authorized person retrieves targeted privileged details."""
    ks = KnowledgeService(person_service=in_memory_person_service)

    alice_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=enrolled_alice,
        display_name="Dr. Alice Smith",
    )

    # 1. Target office room
    facts_office = ks.get_privileged_context(
        requested_person_id=enrolled_alice,
        auth_session=alice_session,
        language="en",
        query="Where is my room?",
    )
    office_blob = "\n".join(facts_office)
    assert "Person Name: Alice" in office_blob
    assert "Office Room: Room 402" in office_blob
    # Does not dump unrelated working hours or Q&As
    assert "Wednesdays 2pm-4pm" not in office_blob
    assert "Sun-Thu 9am-5pm" not in office_blob

    # 2. Target specific Q&A
    facts_qa = ks.get_privileged_context(
        requested_person_id=enrolled_alice,
        auth_session=alice_session,
        language="en",
        query="What is your research lab?",
    )
    qa_blob = "\n".join(facts_qa)
    assert "Personal Q&A:" in qa_blob
    assert "Robotics & HCI Lab" in qa_blob

    # 3. Target Bangla self-identity
    facts_bn = ks.get_privileged_context(
        requested_person_id=enrolled_alice,
        auth_session=alice_session,
        language="bn",
        query="আমি কে? আমার পরিচয় বলো",
    )
    bn_blob = "\n".join(facts_bn)
    assert "ব্যক্তির নাম: Alice" in bn_blob
    assert "পদবি: Associate Professor" in bn_blob


# ==============================================================================
# Unified Context Construction (build_knowledge_context) Tests
# ==============================================================================


def test_build_knowledge_context_unauthorized_user_receives_only_public_facts(
    in_memory_person_service: PersonService,
    enrolled_alice: str,
) -> None:
    """Verify that an unauthorized query contains ONLY public institutional facts
    and excludes all personal context, office numbers, and private Q&A.
    """
    ks = KnowledgeService(person_service=in_memory_person_service)

    context = ks.build_knowledge_context(
        query="Who is Akib? Where is Alice's room?",
        language="en",
        auth_session=None,
        requested_person_id=enrolled_alice,
    )

    # Public fact is present
    assert "Institutional Facts:" in context
    assert "Akib" in context

    # Privileged data is completely absent
    assert "Authorized Person Context:" not in context
    assert "Room 402" not in context
    assert "Associate Professor" not in context
    assert "Robotics & HCI Lab" not in context


def test_build_knowledge_context_public_question_excludes_personal_details_even_when_authorized(
    in_memory_person_service: PersonService,
    enrolled_alice: str,
) -> None:
    """Verify that a purely public institutional question does NOT include personal profile details
    or personal Q&A even when the user is authorized.
    """
    ks = KnowledgeService(person_service=in_memory_person_service)

    alice_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=enrolled_alice,
        display_name="Dr. Alice Smith",
    )

    context = ks.build_knowledge_context(
        query="Who is Akib?",
        language="en",
        auth_session=alice_session,
        requested_person_id=enrolled_alice,
    )

    # Institutional fact is present
    assert "Institutional Facts:" in context
    assert "Akib" in context

    # Unrelated personal details are NOT appended
    assert "Authorized Person Context:" not in context
    assert "Room 402" not in context
    assert "Associate Professor" not in context
    assert "Robotics & HCI Lab" not in context


def test_build_knowledge_context_authorized_user_receives_both_when_query_is_relevant(
    in_memory_person_service: PersonService,
    enrolled_alice: str,
) -> None:
    """Verify that a validly authorized request includes both public facts and personal context
    when the query asks about both.
    """
    ks = KnowledgeService(person_service=in_memory_person_service)

    alice_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=enrolled_alice,
        display_name="Dr. Alice Smith",
    )

    context = ks.build_knowledge_context(
        query="Who is Akib? Where is my office?",
        language="en",
        auth_session=alice_session,
        requested_person_id=enrolled_alice,
    )

    # Both public and personal sections are present
    assert "Institutional Facts:" in context
    assert "Akib" in context
    assert "Authorized Person Context:" in context
    assert "Person Name: Alice" in context
    assert "Office Room: Room 402" in context
