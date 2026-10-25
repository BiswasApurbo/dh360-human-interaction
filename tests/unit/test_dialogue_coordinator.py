"""Unit tests for DialogueCoordinator runtime service."""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.backend.person_service import PersonService, QAPair
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.contracts.authorization import (
    AuthorizationSession,
    AuthorizationStatus,
)
from dh360_hi.contracts.dialogue import (
    DialogueTurn,
    SupportedLanguage,
    get_fallback_message,
)
from dh360_hi.contracts.errors import (
    LLMProviderError,
    ProviderNetworkError,
    ProviderTimeoutError,
    STTProviderError,
    TTSProviderError,
)
from dh360_hi.contracts.events import (
    PersonIdentifiedEvent,
    PersonLostIdentityEvent,
    SessionEndedEvent,
    SessionStartedEvent,
)
from dh360_hi.edge.event_bus import EventBus

# ==============================================================================
# Mock Providers & Test Fixtures
# ==============================================================================


class MockLLMProvider:
    """Mock LLMProvider recording calls and returning predictable text."""

    def __init__(self, response_text: str | None = None) -> None:
        self.default_response = response_text
        self.last_prompt: str | None = None
        self.last_context: str | None = None
        self.last_language: SupportedLanguage | None = None
        self.last_history: list[DialogueTurn] | None = None
        self.raise_error: Exception | None = None
        self.before_generate_hook: Callable[[], None] | None = None

    def generate_response(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> str:
        if self.before_generate_hook is not None:
            self.before_generate_hook()

        if self.raise_error is not None:
            raise self.raise_error

        self.last_prompt = prompt
        self.last_context = filtered_context
        self.last_language = language
        self.last_history = list(session_history) if session_history is not None else None

        if self.default_response is not None:
            return self.default_response
        return f"Response to '{prompt}' (Context length: {len(filtered_context)})"


class MockSTTProvider:
    """Mock STTProvider tracking hold/release states."""

    def __init__(self, text: str = "Test spoken utterance") -> None:
        self.text = text
        self.is_held: bool = False
        self.hold_count: int = 0
        self.release_count: int = 0
        self.last_language_hint: SupportedLanguage | None = None
        self.raise_error: Exception | None = None

    def listen_utterance(
        self,
        language_hint: SupportedLanguage = "bn",
        on_partial_text: Callable[[str], None] | None = None,
    ) -> str:
        if self.raise_error is not None:
            raise self.raise_error
        self.last_language_hint = language_hint
        if on_partial_text:
            on_partial_text("partial")
        return self.text

    def hold(self) -> None:
        self.is_held = True
        self.hold_count += 1

    def release(self) -> None:
        self.is_held = False
        self.release_count += 1


class MockTTSProvider:
    """Mock TTSProvider capturing spoken text and simulating errors."""

    def __init__(self) -> None:
        self.last_spoken_text: str | None = None
        self.last_language: SupportedLanguage | None = None
        self.is_playing_state: bool = False
        self.raise_error: Exception | None = None
        self.interrupt_count: int = 0

    def speak(
        self,
        text: str,
        language: SupportedLanguage = "bn",
        blocking: bool = True,
    ) -> None:
        _ = blocking
        if self.raise_error is not None:
            raise self.raise_error
        self.last_spoken_text = text
        self.last_language = language

    def interrupt(self) -> None:
        self.is_playing_state = False
        self.interrupt_count += 1

    def is_playing(self) -> bool:
        return self.is_playing_state


class MockAuthController:
    """Mock AuthorizationController for setting current session snapshot."""

    def __init__(self, current: AuthorizationSession | None = None) -> None:
        self.current_session = current

    def get_current(self) -> AuthorizationSession | None:
        return self.current_session


@pytest.fixture
def person_service() -> tuple[PersonService, str, str]:
    """In-memory PersonService with enrolled Alice and Bob."""
    db = DatabaseManager(":memory:")
    repo = SQLitePersonRepository(db)
    service = PersonService(repository=repo)

    # Enroll Alice
    alice_profile = service.create_person(
        display_name="Dr. Alice Smith",
        preferred_name="Alice",
    )
    service.update_person(
        person_id=alice_profile.person_id,
        role="Associate Professor",
        department="Computer Science",
        office_room="Room 402",
        working_hours="Sun-Thu 9am-5pm",
        profile_description="Specialist in AI and Robotics",
        qa_pairs=[
            QAPair(
                question="Where can I find your office?",
                answer="Room 402, Science Building 4th floor.",
            ),
            QAPair(
                question="When are your student consulting hours?",
                answer="Wednesdays 2pm-4pm.",
            ),
        ],
    )

    # Enroll Bob
    bob_profile = service.create_person(
        display_name="Dr. Bob Jones",
        preferred_name="Bob",
    )
    service.update_person(
        person_id=bob_profile.person_id,
        role="Assistant Professor",
        department="Electrical Engineering",
        office_room="Room 105",
    )

    return service, alice_profile.person_id, bob_profile.person_id


@pytest.fixture
def knowledge_service(person_service: tuple[PersonService, str, str]) -> KnowledgeService:
    """KnowledgeService loaded with staged CSV and test person repository."""
    service, _, _ = person_service
    return KnowledgeService(person_service=service)


@pytest.fixture
def alice_id(person_service: tuple[PersonService, str, str]) -> str:
    return person_service[1]


@pytest.fixture
def bob_id(person_service: tuple[PersonService, str, str]) -> str:
    return person_service[2]


# ==============================================================================
# Unit Tests
# ==============================================================================


def test_dialogue_coordinator_initialization(knowledge_service: KnowledgeService) -> None:
    """Verify DialogueCoordinator initializes with clean state and event subscriptions."""
    bus = EventBus()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        default_language="bn",
    )

    assert coordinator.session_state is None
    assert coordinator.active_session_id is None
    assert coordinator.default_language == "bn"
    assert coordinator.get_history() == []

    # Subscribed to 4 domain event types
    assert bus.subscriber_count(SessionStartedEvent) == 1
    assert bus.subscriber_count(PersonIdentifiedEvent) == 1
    assert bus.subscriber_count(PersonLostIdentityEvent) == 1
    assert bus.subscriber_count(SessionEndedEvent) == 1

    coordinator.close()
    assert bus.subscriber_count(SessionStartedEvent) == 0


def test_session_started_initializes_session_state(
    knowledge_service: KnowledgeService, alice_id: str
) -> None:
    """Verify SessionStartedEvent initializes an isolated DialogueSessionState."""
    bus = EventBus()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
    )

    bus.publish(
        SessionStartedEvent(
            session_id="session-1",
            track_id=1,
            person_id=alice_id,
            display_name="Alice",
        )
    )

    state = coordinator.session_state
    assert state is not None
    assert state.session_id == "session-1"
    assert state.person_id == alice_id
    assert state.display_name == "Alice"
    assert not state.is_authorized  # Not authorized until verified by AuthorizationController
    assert coordinator.get_history() == []

    coordinator.close()


def test_session_history_isolation_across_sessions(knowledge_service: KnowledgeService) -> None:
    """Verify dialog turns from one session never leak into a subsequent session."""
    bus = EventBus()
    llm = MockLLMProvider("Response 1")
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        llm_provider=llm,
    )

    # Start Session 1
    bus.publish(SessionStartedEvent(session_id="session-1"))
    coordinator.handle_turn("Question 1", language="en")
    coordinator.handle_turn("Question 2", language="en")

    history_1 = coordinator.get_history()
    assert len(history_1) == 4  # 2 user turns + 2 model turns

    # Start Session 2 with different session ID
    bus.publish(SessionStartedEvent(session_id="session-2"))
    history_2 = coordinator.get_history()
    assert len(history_2) == 0  # History of Session 1 purged

    coordinator.close()


def test_session_ended_clears_history(knowledge_service: KnowledgeService) -> None:
    """Verify SessionEndedEvent purges dialogue turns and clears session state."""
    bus = EventBus()
    llm = MockLLMProvider("Model answer")
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        llm_provider=llm,
    )

    bus.publish(SessionStartedEvent(session_id="session-1"))
    coordinator.handle_turn("Hello robot", language="en")
    assert len(coordinator.get_history()) == 2

    # End the session
    bus.publish(SessionEndedEvent(session_id="session-1"))
    assert coordinator.session_state is None
    assert coordinator.get_history() == []

    coordinator.close()


def test_face_identification_alone_never_grants_privileged_access(
    knowledge_service: KnowledgeService, alice_id: str
) -> None:
    """Verify that face identification events alone never grant authorization (Fix F-03)."""
    bus = EventBus()
    auth_controller = MockAuthController(current=None)
    llm = MockLLMProvider()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        authorization_controller=auth_controller,
        llm_provider=llm,
    )

    bus.publish(
        SessionStartedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )
    bus.publish(
        PersonIdentifiedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )

    # State tracks identity, but AuthorizationController has not authorized
    assert coordinator.session_state is not None
    assert coordinator.session_state.person_id == alice_id
    assert coordinator.session_state.is_authorized is False

    # Alice asks for her room
    coordinator.handle_turn("Where is my room?", language="en")
    assert llm.last_context is not None
    # Privileged data is completely withheld
    assert "Authorized Person Context:" not in llm.last_context
    assert "Room 402" not in llm.last_context

    coordinator.close()


def test_old_session_identity_event_cannot_overwrite_newer_session(
    knowledge_service: KnowledgeService, alice_id: str, bob_id: str
) -> None:
    """Verify delayed/stale PersonIdentifiedEvent from an old session is rejected (Fix F-04)."""
    bus = EventBus()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        llm_provider=MockLLMProvider("OK"),
    )

    # Session 1 starts and ends
    bus.publish(SessionStartedEvent(session_id="session-1", person_id=alice_id))
    bus.publish(SessionEndedEvent(session_id="session-1"))
    assert coordinator.session_state is None

    # Stale identity event from session-1 arrives after session ended
    bus.publish(PersonIdentifiedEvent(session_id="session-1", person_id=alice_id))
    assert coordinator.session_state is None  # Stale event rejected!

    # Session 2 starts with Bob
    bus.publish(SessionStartedEvent(session_id="session-2", person_id=bob_id))
    assert coordinator.session_state is not None
    assert coordinator.session_state.session_id == "session-2"
    assert coordinator.session_state.person_id == bob_id

    # Stale identity event from session-1 arrives while session-2 is active
    bus.publish(PersonIdentifiedEvent(session_id="session-1", person_id=alice_id))
    # Session 2 is completely unperturbed!
    assert coordinator.session_state.session_id == "session-2"
    assert coordinator.session_state.person_id == bob_id

    coordinator.close()


def test_identity_loss_immediate_downgrade(
    knowledge_service: KnowledgeService, alice_id: str
) -> None:
    """Verify PersonLostIdentityEvent immediately revokes privileged access and prunes context."""
    bus = EventBus()
    auth_controller = MockAuthController()
    llm = MockLLMProvider()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        authorization_controller=auth_controller,
        llm_provider=llm,
    )

    # 1. Start session with Alice authorized
    bus.publish(
        SessionStartedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )
    bus.publish(
        PersonIdentifiedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )
    alice_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=alice_id,
        display_name="Alice",
    )
    auth_controller.current_session = alice_session

    # Alice asks for room while authorized
    coordinator.handle_turn("Where is my room?", language="en")
    assert llm.last_context is not None
    assert "Office Room: Room 402" in llm.last_context

    # 2. Identity lost event published
    bus.publish(
        PersonLostIdentityEvent(
            session_id="session-1",
            previous_person_id=alice_id,
        )
    )

    assert coordinator.session_state is not None
    assert coordinator.session_state.is_authorized is False
    assert coordinator.session_state.is_in_grace_period is True

    # Controller snapshot also reflects identity loss
    auth_controller.current_session = AuthorizationSession(
        status=AuthorizationStatus.LOST,
        person_id=alice_id,
        display_name="Alice",
    )

    # Alice (or unknown) asks for room during identity loss
    coordinator.handle_turn("Where is my room?", language="en")
    assert llm.last_context is not None
    # PRIVILEGE PRUNED: Office Room 402 is NOT in the filtered context
    assert "Room 402" not in llm.last_context

    coordinator.close()


def test_reidentification_same_person_within_grace_restores_history(
    knowledge_service: KnowledgeService, alice_id: str
) -> None:
    """Verify re-identifying the same person within 60s retains conversation history."""
    bus = EventBus()
    auth_controller = MockAuthController(
        AuthorizationSession(
            status=AuthorizationStatus.AUTHORIZED,
            person_id=alice_id,
            display_name="Alice",
        )
    )
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        authorization_controller=auth_controller,
        llm_provider=MockLLMProvider("OK"),
    )

    bus.publish(
        SessionStartedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )
    bus.publish(
        PersonIdentifiedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )

    # 2 turns recorded at t=100
    coordinator.handle_turn("Turn 1", language="en", now=100.0)
    assert len(coordinator.get_history(now=100.0)) == 2

    # Identity lost at t=105
    auth_controller.current_session = AuthorizationSession(
        status=AuthorizationStatus.LOST,
        person_id=alice_id,
        display_name="Alice",
    )
    bus.publish(PersonLostIdentityEvent(session_id="session-1", previous_person_id=alice_id))
    assert coordinator.session_state is not None
    assert coordinator.session_state.is_authorized is False

    # Alice returns at t=120 (within 60s grace period)
    auth_controller.current_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=alice_id,
        display_name="Alice",
    )
    bus.publish(
        PersonIdentifiedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )

    # History was preserved!
    assert len(coordinator.get_history(now=120.0)) == 2
    assert coordinator.session_state.is_authorized is True

    coordinator.close()


def test_reidentification_different_person_purges_history(
    knowledge_service: KnowledgeService, alice_id: str, bob_id: str
) -> None:
    """Verify that a different person being identified purges the previous subject's history."""
    bus = EventBus()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        llm_provider=MockLLMProvider("OK"),
    )

    bus.publish(
        SessionStartedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )
    bus.publish(
        PersonIdentifiedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )

    coordinator.handle_turn("Alice's secret turn", language="en")
    assert len(coordinator.get_history()) == 2

    # Bob steps up in the same session
    bus.publish(
        PersonIdentifiedEvent(
            session_id="session-1",
            person_id=bob_id,
            display_name="Bob",
        )
    )

    # Alice's dialogue history is completely purged
    assert len(coordinator.get_history()) == 0
    assert coordinator.session_state is not None
    assert coordinator.session_state.person_id == bob_id

    coordinator.close()


def test_reidentification_after_grace_expired_purges_history(
    knowledge_service: KnowledgeService, alice_id: str
) -> None:
    """Verify that identity restoration after grace period expiration purges previous history."""
    bus = EventBus()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        llm_provider=MockLLMProvider("OK"),
        grace_period_seconds=60.0,
    )

    bus.publish(
        SessionStartedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )
    bus.publish(
        PersonIdentifiedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )

    coordinator.handle_turn("Turn 1", language="en", now=100.0)
    assert len(coordinator.get_history(now=100.0)) == 2

    # Identity lost at t=105
    bus.publish(PersonLostIdentityEvent(session_id="session-1", previous_person_id=alice_id))
    assert coordinator.session_state is not None
    coordinator.session_state.identity_lost_at = 105.0

    # Alice returns at t=170 (>60s grace period)
    coordinator.session_state.on_reauthorized(alice_id, now=170.0)

    # Expired history was purged
    assert len(coordinator.get_history(now=170.0)) == 0

    coordinator.close()


def test_authorization_lost_during_generation_suppresses_privileged_response_and_tts(
    knowledge_service: KnowledgeService, alice_id: str
) -> None:
    """Verify identity loss during LLM generation suppresses TTS and returns fallback (Fix F-01)."""
    bus = EventBus()
    auth_controller = MockAuthController(
        AuthorizationSession(
            status=AuthorizationStatus.AUTHORIZED,
            person_id=alice_id,
            display_name="Alice",
        )
    )
    llm = MockLLMProvider("Your office is Room 402, Science Building.")
    tts = MockTTSProvider()

    llm_entered = threading.Event()
    continue_llm = threading.Event()

    def pause_in_llm() -> None:
        llm_entered.set()
        assert continue_llm.wait(timeout=5.0)

    llm.before_generate_hook = pause_in_llm

    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        authorization_controller=auth_controller,
        llm_provider=llm,
        tts_provider=tts,
    )

    bus.publish(SessionStartedEvent(session_id="session-1", person_id=alice_id))
    bus.publish(PersonIdentifiedEvent(session_id="session-1", person_id=alice_id))

    turn_result: dict[str, DialogueTurn] = {}

    def run_turn() -> None:
        turn_result["turn"] = coordinator.handle_turn(
            "Where is my room?", language="en", speak_response=True
        )

    turn_thread = threading.Thread(target=run_turn)
    turn_thread.start()

    # Wait until LLM is generating with privileged context
    assert llm_entered.wait(timeout=5.0)

    # Simulate identity loss and auth revocation while LLM is generating
    auth_controller.current_session = AuthorizationSession(
        status=AuthorizationStatus.LOST,
        person_id=alice_id,
        display_name="Alice",
    )
    bus.publish(PersonLostIdentityEvent(session_id="session-1", previous_person_id=alice_id))

    # Allow LLM generation to complete
    continue_llm.set()
    turn_thread.join(timeout=5.0)

    turn = turn_result["turn"]
    # Privileged text was suppressed and replaced with unauthorized fallback!
    assert turn.text == get_fallback_message("unauthorized", "en")
    assert turn.metadata.get("error") == "AUTHORIZATION_REVOKED"
    # TTS was suppressed! Privileged room number was NEVER spoken
    assert tts.last_spoken_text is None

    # History contains the unauthorized fallback turn, NOT Room 402
    history = coordinator.get_history()
    assert len(history) == 2
    assert "Room 402" not in history[1].text
    assert history[1].text == get_fallback_message("unauthorized", "en")

    coordinator.close()


def test_session_ended_during_generation_prevents_history_mutation(
    knowledge_service: KnowledgeService, alice_id: str
) -> None:
    """Verify SessionEndedEvent during LLM generation prevents history mutation (Fix F-01)."""
    bus = EventBus()
    auth_controller = MockAuthController(
        AuthorizationSession(
            status=AuthorizationStatus.AUTHORIZED,
            person_id=alice_id,
            display_name="Alice",
        )
    )
    llm = MockLLMProvider("Response for session 1")
    tts = MockTTSProvider()

    llm_entered = threading.Event()
    continue_llm = threading.Event()

    def pause_in_llm() -> None:
        llm_entered.set()
        assert continue_llm.wait(timeout=5.0)

    llm.before_generate_hook = pause_in_llm

    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        authorization_controller=auth_controller,
        llm_provider=llm,
        tts_provider=tts,
    )

    bus.publish(SessionStartedEvent(session_id="session-1", person_id=alice_id))

    turn_result: dict[str, DialogueTurn] = {}

    def run_turn() -> None:
        turn_result["turn"] = coordinator.handle_turn(
            "Where is my room?", language="en", speak_response=True
        )

    turn_thread = threading.Thread(target=run_turn)
    turn_thread.start()

    assert llm_entered.wait(timeout=5.0)

    # Session terminates while generation in flight
    bus.publish(SessionEndedEvent(session_id="session-1"))

    continue_llm.set()
    turn_thread.join(timeout=5.0)

    turn = turn_result["turn"]
    assert turn.metadata.get("error") == "SESSION_TERMINATED"
    # No TTS spoken
    assert tts.last_spoken_text is None
    # Coordinator state is None and history is empty
    assert coordinator.session_state is None
    assert coordinator.get_history() == []

    coordinator.close()


def test_person_change_during_generation_aborts_commit_and_tts(
    knowledge_service: KnowledgeService, alice_id: str, bob_id: str
) -> None:
    """Verify switching persons during generation suppresses output and commit (Fix F-01)."""
    bus = EventBus()
    auth_controller = MockAuthController(
        AuthorizationSession(
            status=AuthorizationStatus.AUTHORIZED,
            person_id=alice_id,
            display_name="Alice",
        )
    )
    llm = MockLLMProvider("Alice's confidential profile response")
    tts = MockTTSProvider()

    llm_entered = threading.Event()
    continue_llm = threading.Event()

    def pause_in_llm() -> None:
        llm_entered.set()
        assert continue_llm.wait(timeout=5.0)

    llm.before_generate_hook = pause_in_llm

    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        authorization_controller=auth_controller,
        llm_provider=llm,
        tts_provider=tts,
    )

    bus.publish(SessionStartedEvent(session_id="session-1", person_id=alice_id))
    bus.publish(PersonIdentifiedEvent(session_id="session-1", person_id=alice_id))

    turn_result: dict[str, DialogueTurn] = {}

    def run_turn() -> None:
        turn_result["turn"] = coordinator.handle_turn(
            "Where is my room?", language="en", speak_response=True
        )

    turn_thread = threading.Thread(target=run_turn)
    turn_thread.start()

    assert llm_entered.wait(timeout=5.0)

    # Bob steps up and is identified mid-generation
    auth_controller.current_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=bob_id,
        display_name="Bob",
    )
    bus.publish(PersonIdentifiedEvent(session_id="session-1", person_id=bob_id))

    continue_llm.set()
    turn_thread.join(timeout=5.0)

    turn = turn_result["turn"]
    assert turn.text == get_fallback_message("unauthorized", "en")
    assert turn.metadata.get("error") == "IDENTITY_CHANGED"
    # TTS suppressed
    assert tts.last_spoken_text is None
    # Bob's history is empty; Alice's response was NOT saved to Bob's history
    assert coordinator.session_state is not None
    assert coordinator.session_state.person_id == bob_id
    assert coordinator.get_history() == []

    coordinator.close()


def test_concurrent_turns_serialized_no_interleaving(
    knowledge_service: KnowledgeService,
) -> None:
    """Verify concurrent turns are serialized and cannot interleave history (Fix F-02)."""
    bus = EventBus()

    order: list[str] = []

    class SlowLLMProvider(MockLLMProvider):
        def generate_response(
            self,
            prompt: str,
            filtered_context: str,
            language: SupportedLanguage,
            session_history: list[DialogueTurn] | None = None,
        ) -> str:
            _ = (filtered_context, language, session_history)
            order.append(f"llm_{prompt}")
            return f"Answer to {prompt}"

    llm = SlowLLMProvider()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        llm_provider=llm,
    )

    bus.publish(SessionStartedEvent(session_id="session-1"))

    results: dict[int, DialogueTurn] = {}

    def run_turn(turn_id: int, text: str) -> None:
        results[turn_id] = coordinator.handle_turn(text, language="en")

    t1 = threading.Thread(target=run_turn, args=(1, "Turn1"))
    t2 = threading.Thread(target=run_turn, args=(2, "Turn2"))

    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    history = coordinator.get_history()
    assert len(history) == 4

    # Turns alternate strictly: User -> Model -> User -> Model
    assert [t.role for t in history] == ["user", "model", "user", "model"]
    first_query = history[0].text
    first_response = history[1].text
    second_query = history[2].text
    second_response = history[3].text

    assert first_response == f"Answer to {first_query}"
    assert second_response == f"Answer to {second_query}"

    coordinator.close()


def test_public_turn_not_treated_as_privileged_when_unauthorized(
    knowledge_service: KnowledgeService,
) -> None:
    """Verify public-only turns succeed and speak via TTS without requiring authorization."""
    bus = EventBus()
    tts = MockTTSProvider()
    llm = MockLLMProvider("The university is open Sun-Thu.")
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        authorization_controller=None,
        llm_provider=llm,
        tts_provider=tts,
    )

    bus.publish(SessionStartedEvent(session_id="session-1"))

    turn = coordinator.handle_turn(
        "When is the university open?", language="en", speak_response=True
    )
    assert turn.role == "model"
    assert turn.text == "The university is open Sun-Thu."
    assert "error" not in turn.metadata
    assert tts.last_spoken_text == "The university is open Sun-Thu."

    history = coordinator.get_history()
    assert len(history) == 2

    coordinator.close()


def test_authorization_refresh_on_every_turn_downgrade_when_controller_times_out(
    knowledge_service: KnowledgeService, alice_id: str
) -> None:
    """Verify that if AuthorizationController times out, next turn is immediately downgraded."""
    bus = EventBus()
    auth_controller = MockAuthController()
    llm = MockLLMProvider()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        authorization_controller=auth_controller,
        llm_provider=llm,
    )

    bus.publish(
        SessionStartedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )

    # Active authorization
    auth_controller.current_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=alice_id,
        display_name="Alice",
    )

    coordinator.handle_turn("Where is my room?", language="en")
    assert llm.last_context is not None
    assert "Office Room: Room 402" in llm.last_context

    # Controller times out
    auth_controller.current_session = AuthorizationSession(
        status=AuthorizationStatus.TIMEOUT,
        ended_at_iso="2026-10-25T12:00:00Z",
    )

    # Next turn: fresh check sees timeout and downgrades to public-only
    coordinator.handle_turn("Where is my room?", language="en")
    assert coordinator.session_state is not None
    assert coordinator.session_state.is_authorized is False
    assert "Room 402" not in llm.last_context

    coordinator.close()


def test_authorization_refresh_downgrade_when_person_mismatched(
    knowledge_service: KnowledgeService, alice_id: str, bob_id: str
) -> None:
    """Verify that an authorization snapshot for a different person is rejected."""
    bus = EventBus()
    auth_controller = MockAuthController()
    llm = MockLLMProvider()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        authorization_controller=auth_controller,
        llm_provider=llm,
    )

    bus.publish(
        SessionStartedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )

    # Controller mistakenly reports Bob authorized while session is Alice
    auth_controller.current_session = AuthorizationSession(
        status=AuthorizationStatus.AUTHORIZED,
        person_id=bob_id,
        display_name="Bob",
    )

    coordinator.handle_turn("Where is my room?", language="en")
    # Mismatch rejected: no privileged context
    assert llm.last_context is not None
    assert "Room 402" not in llm.last_context
    assert "Room 105" not in llm.last_context

    coordinator.close()


def test_unauthorized_user_receives_only_public_context(
    knowledge_service: KnowledgeService,
) -> None:
    """Verify unidentified/unauthorized users receive public facts only."""
    bus = EventBus()
    llm = MockLLMProvider()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        llm_provider=llm,
    )

    # Direct query without authorization
    coordinator.handle_turn("When is the university open?", language="en")
    assert llm.last_context is not None
    # Public facts included
    assert "Institutional Facts:" in llm.last_context
    # No private faculty details
    assert "Authorized Person Context:" not in llm.last_context
    assert "Room 402" not in llm.last_context

    coordinator.close()


def test_authorized_user_receives_privileged_context_when_query_matches(
    knowledge_service: KnowledgeService, alice_id: str
) -> None:
    """Verify authorized users receive privileged profile context when query asks for it."""
    bus = EventBus()
    auth_controller = MockAuthController(
        AuthorizationSession(
            status=AuthorizationStatus.AUTHORIZED,
            person_id=alice_id,
            display_name="Dr. Alice Smith",
        )
    )
    llm = MockLLMProvider()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        authorization_controller=auth_controller,
        llm_provider=llm,
    )

    bus.publish(
        SessionStartedEvent(
            session_id="session-1",
            person_id=alice_id,
            display_name="Alice",
        )
    )

    coordinator.handle_turn("Where is my room?", language="en")
    assert llm.last_context is not None
    assert "Authorized Person Context:" in llm.last_context
    assert "Office Room: Room 402" in llm.last_context

    coordinator.close()


def test_stt_provider_typed_error_fallback_english_and_bangla(
    knowledge_service: KnowledgeService,
) -> None:
    """Verify typed STTProviderError is converted to safe localized fallback text."""
    bus = EventBus()
    stt = MockSTTProvider()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        stt_provider=stt,
    )

    # 1. STT failure in English
    stt.raise_error = STTProviderError("Low signal-to-noise ratio in microphone input")
    turn_en = coordinator.handle_turn(None, language="en")
    assert turn_en.role == "model"
    assert turn_en.text == get_fallback_message("stt_error", "en")
    assert turn_en.metadata["error"] == "STT_PROVIDER_ERROR"

    # 2. STT failure in Bangla
    turn_bn = coordinator.handle_turn(None, language="bn")
    assert turn_bn.text == get_fallback_message("stt_error", "bn")

    # 3. Network error during STT
    stt.raise_error = ProviderNetworkError("DNS failure to soniox.com", provider_name="soniox")
    turn_net = coordinator.handle_turn(None, language="en")
    assert turn_net.text == get_fallback_message("network_error", "en")
    assert turn_net.metadata["error"] == "PROVIDER_NETWORK_ERROR"

    coordinator.close()


def test_llm_provider_typed_error_fallback_english_and_bangla(
    knowledge_service: KnowledgeService,
) -> None:
    """Verify typed LLMProviderError is converted to safe localized fallback text."""
    bus = EventBus()
    llm = MockLLMProvider()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        llm_provider=llm,
    )

    # 1. LLM Error in English
    llm.raise_error = LLMProviderError("Google Gemini quota exhausted 429", provider_name="gemini")
    turn_en = coordinator.handle_turn("Hello", language="en")
    assert turn_en.text == get_fallback_message("llm_error", "en")
    assert turn_en.metadata["error"] == "LLM_PROVIDER_ERROR"

    # 2. LLM Error in Bangla
    turn_bn = coordinator.handle_turn("নমস্কার", language="bn")
    assert turn_bn.text == get_fallback_message("llm_error", "bn")

    # 3. Timeout Error in English
    llm.raise_error = ProviderTimeoutError(
        "Request timed out after 10.0s", provider_name="gemini", timeout_seconds=10.0
    )
    turn_to = coordinator.handle_turn("Hello again", language="en")
    assert turn_to.text == get_fallback_message("network_error", "en")
    assert turn_to.metadata["error"] == "PROVIDER_TIMEOUT"

    coordinator.close()


def test_tts_provider_error_preserves_text_response(knowledge_service: KnowledgeService) -> None:
    """Verify that when TTS synthesis fails, the generated text response is preserved."""
    bus = EventBus()
    llm = MockLLMProvider("The library is open from 8am to 8pm.")
    tts = MockTTSProvider()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        llm_provider=llm,
        tts_provider=tts,
    )

    # Configure TTS to fail
    tts.raise_error = TTSProviderError(
        "ALSA lib pcm.c: playback write error", provider_name="google_tts"
    )

    turn = coordinator.handle_turn("What are library hours?", language="en", speak_response=True)

    # TEXT RESPONSE IS PRESERVED
    assert turn.text == "The library is open from 8am to 8pm."
    assert turn.metadata.get("tts_error") is True
    assert turn.metadata.get("tts_error_code") == "TTS_PROVIDER_ERROR"

    # Text response is recorded in dialogue history
    history = coordinator.get_history()
    assert len(history) == 2
    assert history[1].text == "The library is open from 8am to 8pm."

    coordinator.close()


def test_stt_hold_and_release_lifecycle(knowledge_service: KnowledgeService) -> None:
    """Verify STT provider hold() is called before TTS speak() and release() after."""
    bus = EventBus()
    llm = MockLLMProvider("Audio playback response")
    stt = MockSTTProvider("User spoke")
    tts = MockTTSProvider()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        llm_provider=llm,
        stt_provider=stt,
        tts_provider=tts,
    )

    assert stt.hold_count == 0
    assert stt.release_count == 0

    coordinator.handle_turn("Say something", language="en", speak_response=True)

    assert stt.hold_count == 1
    assert stt.release_count == 1
    assert not stt.is_held  # Released cleanly

    # Even if TTS fails, STT release is guaranteed in finally block
    tts.raise_error = TTSProviderError("Audio device busy", provider_name="google_tts")
    coordinator.handle_turn("Say something again", language="en", speak_response=True)

    assert stt.hold_count == 2
    assert stt.release_count == 2
    assert not stt.is_held

    coordinator.close()


@pytest.mark.anyio
async def test_async_turn_execution_non_blocking(knowledge_service: KnowledgeService) -> None:
    """Verify handle_turn_async offloads execution and returns valid turn."""
    bus = EventBus()
    llm = MockLLMProvider("Async model response")
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
        llm_provider=llm,
    )

    turn = await coordinator.handle_turn_async("Async test query", language="en")
    assert turn.role == "model"
    assert turn.text == "Async model response"
    assert len(coordinator.get_history()) == 2

    coordinator.close()


def test_close_unsubscribes_from_event_bus(knowledge_service: KnowledgeService) -> None:
    """Verify close() unregisters all event handlers and is idempotent."""
    bus = EventBus()
    coordinator = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_service,
    )

    assert bus.subscriber_count(SessionStartedEvent) == 1
    assert bus.subscriber_count(SessionEndedEvent) == 1

    coordinator.close()
    assert bus.subscriber_count(SessionStartedEvent) == 0
    assert bus.subscriber_count(SessionEndedEvent) == 0

    # Idempotent close
    coordinator.close()
