"""Unit tests for InteractionCoordinator and application interaction contracts."""

from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock

from dh360_hi.application.coordinator import InteractionCoordinator
from dh360_hi.backend.models import (
    ConsentStatus,
    PersonProfile,
    PersonStatus,
)
from dh360_hi.backend.person_service import BasePersonService
from dh360_hi.contracts.events import (
    PersonEnteredEvent,
    PersonIdentifiedEvent,
    PersonLeftEvent,
    PersonLostIdentityEvent,
    SessionEndedEvent,
    SessionStartedEvent,
)
from dh360_hi.contracts.interaction import (
    InteractionContext,
    InteractionState,
)
from dh360_hi.edge.event_bus import EventBus

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _mk_person_profile(
    person_id: str = "p-123",
    display_name: str = "Alice Smith",
    preferred_name: str | None = "Alice",
) -> PersonProfile:
    return PersonProfile(
        person_id=person_id,
        display_name=display_name,
        preferred_name=preferred_name,
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.GRANTED,
        consent_timestamp="2026-10-01T12:00:00Z",
        notes=None,
        created_at="2026-10-01T12:00:00Z",
        updated_at="2026-10-01T12:00:00Z",
        embedding_count=2,
    )


# ---------------------------------------------------------------------------
# Initial State & Idempotency Tests
# ---------------------------------------------------------------------------


class TestCoordinatorInitialState:
    def test_initial_state_is_idle(self) -> None:
        bus = EventBus()
        coordinator = InteractionCoordinator(bus)

        ctx = coordinator.get_context()
        assert ctx.state == InteractionState.IDLE
        assert ctx.session_id is None
        assert ctx.person_id is None
        assert ctx.display_name is None
        assert ctx.person_profile is None
        assert ctx.started_at is None
        assert not ctx.identity_known
        assert coordinator.get_last_summary() is None

        coordinator.close()

    def test_close_is_idempotent(self) -> None:
        bus = EventBus()
        coordinator = InteractionCoordinator(bus)
        coordinator.close()
        coordinator.close()  # No error on second close


# ---------------------------------------------------------------------------
# Session Lifecycle & State Transitions Tests
# ---------------------------------------------------------------------------


class TestCoordinatorLifecycleTransitions:
    def test_person_entered_updates_preliminary_idle_context(self) -> None:
        bus = EventBus()
        coordinator = InteractionCoordinator(bus)

        bus.publish(
            PersonEnteredEvent(
                track_id=1,
                person_id="p-prelim",
                display_name="Preliminary User",
                frame_id=1,
                timestamp=10.0,
            )
        )

        ctx = coordinator.get_context()
        assert ctx.state == InteractionState.IDLE
        assert ctx.session_id is None
        assert ctx.person_id == "p-prelim"
        assert ctx.display_name == "Preliminary User"
        assert ctx.last_updated_at == 10.0

        coordinator.close()

    def test_events_without_active_session_ignored_safely(self) -> None:
        bus = EventBus()
        coordinator = InteractionCoordinator(bus)

        # Identified event without active session
        bus.publish(
            PersonIdentifiedEvent(
                session_id="s-ghost",
                track_id=1,
                person_id="p-001",
                display_name="Alice",
                frame_id=1,
                timestamp=10.0,
            )
        )
        assert coordinator.get_context().state == InteractionState.IDLE
        assert coordinator.get_context().session_id is None

        # Lost identity event without active session
        bus.publish(
            PersonLostIdentityEvent(
                session_id="s-ghost",
                track_id=1,
                previous_person_id="p-001",
                frame_id=2,
                timestamp=11.0,
            )
        )
        assert coordinator.get_context().state == InteractionState.IDLE

        coordinator.close()

    def test_unregistered_session_started_creates_active_context(self) -> None:
        bus = EventBus()
        coordinator = InteractionCoordinator(bus)

        bus.publish(
            SessionStartedEvent(
                session_id="s-001",
                track_id=1,
                person_id=None,
                display_name=None,
                frame_id=10,
                timestamp=100.0,
            )
        )

        ctx = coordinator.get_context()
        assert ctx.state == InteractionState.GREETING
        assert ctx.session_id == "s-001"
        assert ctx.person_id is None
        assert ctx.display_name is None
        assert ctx.person_profile is None
        assert not ctx.identity_known
        assert ctx.started_at == 100.0
        assert ctx.identity_transitions_count == 0

        coordinator.close()

    def test_identified_session_started_enriches_profile(self) -> None:
        bus = EventBus()
        profile = _mk_person_profile(person_id="p-001", preferred_name="Dr. Alice")
        service = MagicMock(spec=BasePersonService)
        service.get_person.return_value = profile

        coordinator = InteractionCoordinator(bus, person_service=service)

        bus.publish(
            SessionStartedEvent(
                session_id="s-002",
                track_id=1,
                person_id="p-001",
                display_name="Alice Smith",
                frame_id=10,
                timestamp=100.0,
            )
        )

        ctx = coordinator.get_context()
        assert ctx.state == InteractionState.GREETING
        assert ctx.session_id == "s-002"
        assert ctx.person_id == "p-001"
        assert ctx.display_name == "Dr. Alice"
        assert ctx.person_profile == profile
        assert ctx.identity_known
        assert ctx.identity_transitions_count == 1
        service.get_person.assert_called_once_with("p-001")

        coordinator.close()

    def test_mid_session_identification_updates_context(self) -> None:
        bus = EventBus()
        profile = _mk_person_profile(
            person_id="p-001",
            display_name="Bob Jones",
            preferred_name=None,
        )
        service = MagicMock(spec=BasePersonService)
        service.get_person.return_value = profile

        coordinator = InteractionCoordinator(bus, person_service=service)

        # 1. Start as unregistered
        bus.publish(
            SessionStartedEvent(
                session_id="s-003",
                track_id=1,
                person_id=None,
                display_name=None,
                frame_id=10,
                timestamp=100.0,
            )
        )
        assert not coordinator.get_context().identity_known

        # 2. Identified mid-session
        bus.publish(
            PersonIdentifiedEvent(
                session_id="s-003",
                track_id=1,
                person_id="p-001",
                display_name="Bob Jones",
                frame_id=20,
                timestamp=101.0,
            )
        )

        ctx = coordinator.get_context()
        assert ctx.state == InteractionState.ENGAGED
        assert ctx.session_id == "s-003"
        assert ctx.person_id == "p-001"
        assert ctx.display_name == "Bob Jones"
        assert ctx.person_profile == profile
        assert ctx.identity_known
        assert ctx.identity_transitions_count == 1

        coordinator.close()

    def test_profile_enrichment_failure_handled_gracefully(self) -> None:
        bus = EventBus()
        service = MagicMock(spec=BasePersonService)
        service.get_person.side_effect = RuntimeError("Database connection timeout")

        coordinator = InteractionCoordinator(bus, person_service=service)

        bus.publish(
            SessionStartedEvent(
                session_id="s-004",
                track_id=1,
                person_id="p-999",
                display_name="Fallback Name",
                frame_id=10,
                timestamp=100.0,
            )
        )

        ctx = coordinator.get_context()
        assert ctx.session_id == "s-004"
        assert ctx.person_id == "p-999"
        assert ctx.display_name == "Fallback Name"
        assert ctx.person_profile is None
        assert ctx.identity_known

        coordinator.close()

    def test_identity_loss_preserves_session_and_tracks_last_known(self) -> None:
        bus = EventBus()
        coordinator = InteractionCoordinator(bus)

        # Start identified session
        bus.publish(
            SessionStartedEvent(
                session_id="s-005",
                track_id=1,
                person_id="p-001",
                display_name="Alice",
                frame_id=10,
                timestamp=100.0,
            )
        )
        assert coordinator.get_context().identity_known

        # Identity degrades mid-session
        bus.publish(
            PersonLostIdentityEvent(
                session_id="s-005",
                track_id=1,
                previous_person_id="p-001",
                frame_id=30,
                timestamp=105.0,
            )
        )

        ctx = coordinator.get_context()
        assert ctx.session_id == "s-005"
        assert ctx.person_id is None
        assert not ctx.identity_known
        assert ctx.last_known_person_id == "p-001"
        assert ctx.state == InteractionState.GREETING
        assert ctx.identity_transitions_count == 2

        # Re-identified
        bus.publish(
            PersonIdentifiedEvent(
                session_id="s-005",
                track_id=1,
                person_id="p-001",
                display_name="Alice",
                frame_id=40,
                timestamp=108.0,
            )
        )

        ctx2 = coordinator.get_context()
        assert ctx2.person_id == "p-001"
        assert ctx2.identity_known
        assert ctx2.identity_transitions_count == 3

        coordinator.close()

    def test_person_left_transitions_to_farewell(self) -> None:
        bus = EventBus()
        coordinator = InteractionCoordinator(bus)

        bus.publish(
            SessionStartedEvent(
                session_id="s-006",
                track_id=1,
                person_id="p-001",
                display_name="Alice",
                frame_id=10,
                timestamp=100.0,
            )
        )

        bus.publish(
            PersonLeftEvent(
                track_id=1,
                person_id="p-001",
                display_name="Alice",
                frame_id=50,
                timestamp=110.0,
            )
        )

        ctx = coordinator.get_context()
        assert ctx.state == InteractionState.FAREWELL
        assert ctx.session_id == "s-006"

        coordinator.close()

    def test_session_ended_generates_summary_and_resets_context(self) -> None:
        bus = EventBus()
        profile = _mk_person_profile(person_id="p-001")
        service = MagicMock(spec=BasePersonService)
        service.get_person.return_value = profile

        coordinator = InteractionCoordinator(bus, person_service=service)

        bus.publish(
            SessionStartedEvent(
                session_id="s-007",
                track_id=1,
                person_id="p-001",
                display_name="Alice Smith",
                frame_id=10,
                timestamp=100.0,
            )
        )

        bus.publish(
            PersonLeftEvent(
                track_id=1,
                person_id="p-001",
                display_name="Alice Smith",
                frame_id=50,
                timestamp=115.0,
            )
        )

        bus.publish(
            SessionEndedEvent(
                session_id="s-007",
                track_id=1,
                final_person_id="p-001",
                final_display_name="Alice Smith",
                frame_id=51,
                timestamp=115.0,
            )
        )

        # Context reset
        ctx = coordinator.get_context()
        assert ctx.state == InteractionState.IDLE
        assert ctx.session_id is None
        assert ctx.person_id is None

        # Summary finalized
        summary = coordinator.get_last_summary()
        assert summary is not None
        assert summary.session_id == "s-007"
        assert summary.person_id == "p-001"
        assert summary.started_at == 100.0
        assert summary.ended_at == 115.0
        assert summary.duration_seconds == 15.0
        assert summary.final_state == InteractionState.FAREWELL
        assert summary.identity_was_known
        assert summary.person_profile == profile

        coordinator.close()

    def test_forget_deleted_person_detaches_live_context_and_summary(self) -> None:
        bus = EventBus()
        profile = _mk_person_profile(person_id="p-delete")
        person_service = MagicMock(spec=BasePersonService)
        person_service.get_person.return_value = profile
        coordinator = InteractionCoordinator(bus, person_service=person_service)

        bus.publish(
            SessionStartedEvent(
                session_id="s-delete",
                track_id=7,
                person_id="p-delete",
                display_name="Alice",
                timestamp=100.0,
            )
        )
        bus.publish(
            PersonIdentifiedEvent(
                session_id="s-delete",
                track_id=7,
                person_id="p-delete",
                display_name="Alice",
                timestamp=101.0,
            )
        )
        bus.publish(
            PersonLostIdentityEvent(
                session_id="s-delete",
                track_id=7,
                previous_person_id="p-delete",
                timestamp=102.0,
            )
        )

        assert coordinator.get_context().last_known_person_id == "p-delete"
        coordinator.forget_deleted_person("p-delete")
        assert coordinator.get_context().last_known_person_id is None

        bus.publish(
            SessionStartedEvent(
                session_id="s-summary",
                track_id=8,
                person_id="p-delete",
                display_name="Alice",
                timestamp=200.0,
            )
        )
        bus.publish(
            SessionEndedEvent(
                session_id="s-summary",
                track_id=8,
                final_person_id="p-delete",
                final_display_name="Alice Smith",
                timestamp=210.0,
            )
        )

        coordinator.forget_deleted_person("p-delete")
        summary = coordinator.get_last_summary()
        assert summary is not None
        assert summary.person_id is None
        assert summary.person_profile is None
        assert summary.display_name == "Alice"
        assert summary.identity_was_known is True
        coordinator.close()


# ---------------------------------------------------------------------------
# Multiple Sessions & Listeners Tests
# ---------------------------------------------------------------------------


class TestCoordinatorMultipleSessionsAndListeners:
    def test_multiple_sequential_sessions(self) -> None:
        bus = EventBus()
        coordinator = InteractionCoordinator(bus)

        # Session 1: Anonymous
        bus.publish(
            SessionStartedEvent(
                session_id="s-01",
                track_id=1,
                person_id=None,
                display_name=None,
                frame_id=1,
                timestamp=10.0,
            )
        )
        bus.publish(
            SessionEndedEvent(
                session_id="s-01",
                track_id=1,
                final_person_id=None,
                final_display_name=None,
                frame_id=5,
                timestamp=15.0,
            )
        )
        s1 = coordinator.get_last_summary()
        assert s1 is not None
        assert s1.session_id == "s-01"
        assert not s1.identity_was_known

        # Session 2: Identified
        bus.publish(
            SessionStartedEvent(
                session_id="s-02",
                track_id=2,
                person_id="p-02",
                display_name="Charlie",
                frame_id=10,
                timestamp=20.0,
            )
        )
        bus.publish(
            SessionEndedEvent(
                session_id="s-02",
                track_id=2,
                final_person_id="p-02",
                final_display_name="Charlie",
                frame_id=20,
                timestamp=30.0,
            )
        )
        s2 = coordinator.get_last_summary()
        assert s2 is not None
        assert s2.session_id == "s-02"
        assert s2.identity_was_known
        assert s2.duration_seconds == 10.0

        coordinator.close()

    def test_context_listeners_receive_updates_with_exception_isolation(self) -> None:
        bus = EventBus()
        coordinator = InteractionCoordinator(bus)

        received: list[InteractionContext] = []

        def failing_listener(_ctx: InteractionContext) -> None:
            raise RuntimeError("Failing listener intentionally explodes")

        coordinator.add_context_listener(failing_listener)
        coordinator.add_context_listener(received.append)

        bus.publish(
            SessionStartedEvent(
                session_id="s-99",
                track_id=1,
                person_id=None,
                display_name=None,
                frame_id=1,
                timestamp=1.0,
            )
        )

        assert len(received) == 1
        assert received[0].session_id == "s-99"

        coordinator.close()

    def test_remove_context_listener(self) -> None:
        bus = EventBus()
        coordinator = InteractionCoordinator(bus)

        received: list[InteractionContext] = []
        lid = coordinator.add_context_listener(received.append)
        assert coordinator.remove_context_listener(lid)
        assert not coordinator.remove_context_listener(lid)  # False on second remove

        bus.publish(
            SessionStartedEvent(
                session_id="s-100",
                track_id=1,
                person_id=None,
                display_name=None,
                frame_id=1,
                timestamp=1.0,
            )
        )
        assert len(received) == 0

        coordinator.close()

    def test_concurrent_context_reads(self) -> None:
        bus = EventBus()
        coordinator = InteractionCoordinator(bus)

        bus.publish(
            SessionStartedEvent(
                session_id="s-conc",
                track_id=1,
                person_id="p-1",
                display_name="Alice",
                frame_id=1,
                timestamp=1.0,
            )
        )

        def read_context() -> InteractionContext:
            return coordinator.get_context()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(read_context) for _ in range(50)]
            results = [f.result() for f in futures]

        assert len(results) == 50
        for ctx in results:
            assert ctx.session_id == "s-conc"
            assert ctx.identity_known

        coordinator.close()
