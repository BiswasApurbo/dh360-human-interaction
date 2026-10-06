"""Unit tests for transport-neutral external integration events and domain event mapper."""

from __future__ import annotations

import dataclasses

import pytest

from dh360_hi.contracts.dto import InteractionContextDTO, InteractionSummaryDTO
from dh360_hi.contracts.events import (
    PersonEnteredEvent,
    PersonIdentifiedEvent,
    PersonLeftEvent,
    PersonLostIdentityEvent,
    SessionEndedEvent,
    SessionStartedEvent,
)
from dh360_hi.contracts.integration import (
    AuthorizationUpdatedIntegrationEvent,
    BaseIntegrationEvent,
    InteractionContextUpdatedIntegrationEvent,
    PersonIdentifiedIntegrationEvent,
    PersonLostIdentityIntegrationEvent,
    SessionEndedIntegrationEvent,
    SessionStartedIntegrationEvent,
    map_domain_event_to_integration,
)


class TestIntegrationEvents:
    def test_base_integration_event(self) -> None:
        """Test BaseIntegrationEvent default fields and immutability."""
        ev = BaseIntegrationEvent()
        assert ev.event_id is not None
        assert ev.event_type == "base"
        assert ev.timestamp_iso is not None

        d = ev.to_dict()
        assert d["event_type"] == "base"
        assert "track_id" not in d
        assert "frame_id" not in d

        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.event_type = "modified"  # type: ignore[misc]

    def test_session_started_integration_event(self) -> None:
        """Test SessionStartedIntegrationEvent."""
        ev = SessionStartedIntegrationEvent(
            session_id="sess-100",
            person_id="p-100",
            display_name="Alice",
        )
        assert ev.event_type == "session_started"
        assert ev.session_id == "sess-100"
        assert ev.person_id == "p-100"
        assert ev.display_name == "Alice"
        assert not hasattr(ev, "track_id")
        assert not hasattr(ev, "frame_id")

    def test_person_identified_integration_event(self) -> None:
        """Test PersonIdentifiedIntegrationEvent."""
        ev = PersonIdentifiedIntegrationEvent(
            session_id="sess-100",
            person_id="p-100",
            display_name="Alice",
        )
        assert ev.event_type == "person_identified"
        assert ev.session_id == "sess-100"
        assert ev.person_id == "p-100"

    def test_person_lost_identity_integration_event(self) -> None:
        """Test PersonLostIdentityIntegrationEvent."""
        ev = PersonLostIdentityIntegrationEvent(
            session_id="sess-100",
            previous_person_id="p-100",
        )
        assert ev.event_type == "person_lost_identity"
        assert ev.session_id == "sess-100"
        assert ev.previous_person_id == "p-100"

    def test_session_ended_integration_event(self) -> None:
        """Test SessionEndedIntegrationEvent with nested summary DTO."""
        summary_dto = InteractionSummaryDTO(
            session_id="sess-100",
            person_id="p-100",
            display_name="Alice",
            started_at_iso="2026-10-01T12:00:00+00:00",
            ended_at_iso="2026-10-01T12:00:30+00:00",
            duration_seconds=30.0,
            final_state="farewell",
            identity_was_known=True,
            identity_transitions_count=1,
        )
        ev = SessionEndedIntegrationEvent(
            session_id="sess-100",
            final_person_id="p-100",
            final_display_name="Alice",
            summary=summary_dto,
        )
        assert ev.event_type == "session_ended"
        assert ev.summary is not None
        assert ev.summary.duration_seconds == 30.0
        d = ev.to_dict()
        assert d["summary"]["duration_seconds"] == 30.0

    def test_context_updated_integration_event(self) -> None:
        """Test InteractionContextUpdatedIntegrationEvent."""
        ctx_dto = InteractionContextDTO(session_id="sess-100", state="engaged")
        ev = InteractionContextUpdatedIntegrationEvent(context=ctx_dto)
        assert ev.event_type == "context_updated"
        assert ev.context.session_id == "sess-100"
        d = ev.to_dict()
        assert d["context"]["state"] == "engaged"

    def test_authorization_updated_integration_event_defaults_and_serialization(self) -> None:
        """Test AuthorizationUpdatedIntegrationEvent fields and serialization."""
        ev = AuthorizationUpdatedIntegrationEvent(
            authorization_id="auth-100",
            status="authorized",
            person_id="p-100",
            display_name="Alice",
            started_at="2026-10-06T10:00:00+00:00",
            ended_at=None,
            updated_at="2026-10-06T10:00:01+00:00",
            is_active=True,
            is_completed=False,
            is_authorized=True,
        )
        assert ev.event_type == "authorization_updated"
        assert ev.authorization_id == "auth-100"
        assert ev.status == "authorized"
        assert ev.person_id == "p-100"
        assert ev.display_name == "Alice"
        assert ev.started_at == "2026-10-06T10:00:00+00:00"
        assert ev.ended_at is None
        assert ev.updated_at == "2026-10-06T10:00:01+00:00"
        assert ev.is_active is True
        assert ev.is_completed is False
        assert ev.is_authorized is True

        # Check immutability
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.status = "lost"  # type: ignore[misc]

        # Serialization checks
        d = ev.to_dict()
        assert d["event_type"] == "authorization_updated"
        assert d["authorization_id"] == "auth-100"
        assert d["status"] == "authorized"
        assert d["person_id"] == "p-100"
        assert d["display_name"] == "Alice"
        assert d["started_at"] == "2026-10-06T10:00:00+00:00"
        assert d["ended_at"] is None
        assert d["updated_at"] == "2026-10-06T10:00:01+00:00"
        assert d["is_active"] is True
        assert d["is_completed"] is False
        assert d["is_authorized"] is True

        # Privacy invariants: no internal vision/perception fields
        forbidden_fields = [
            "track_id",
            "frame_id",
            "embeddings",
            "embedding",
            "bbox",
            "recognition",
            "RecognitionResult",
            "TwoRateFrameResult",
        ]
        for field in forbidden_fields:
            assert field not in d
            assert not hasattr(ev, field)


class TestDomainEventToIntegrationMapper:
    def test_map_session_started(self) -> None:
        """Test mapping SessionStartedEvent to SessionStartedIntegrationEvent."""
        domain_ev = SessionStartedEvent(
            event_id="ev-1",
            frame_id=42,
            timestamp=1700000000.0,
            session_id="sess-1",
            track_id=7,
            person_id="p-1",
            display_name="Bob",
        )
        integ_ev = map_domain_event_to_integration(domain_ev)
        assert isinstance(integ_ev, SessionStartedIntegrationEvent)
        assert integ_ev.event_id == "ev-1"
        assert integ_ev.session_id == "sess-1"
        assert integ_ev.person_id == "p-1"
        assert integ_ev.display_name == "Bob"
        assert integ_ev.timestamp_iso is not None
        # Verify internal details are stripped
        assert not hasattr(integ_ev, "track_id")
        assert not hasattr(integ_ev, "frame_id")

    def test_map_person_identified(self) -> None:
        """Test mapping PersonIdentifiedEvent to PersonIdentifiedIntegrationEvent."""
        domain_ev = PersonIdentifiedEvent(
            event_id="ev-2",
            frame_id=43,
            timestamp=1700000001.0,
            session_id="sess-1",
            track_id=7,
            person_id="p-1",
            display_name="Bob",
        )
        integ_ev = map_domain_event_to_integration(domain_ev)
        assert isinstance(integ_ev, PersonIdentifiedIntegrationEvent)
        assert integ_ev.person_id == "p-1"
        assert not hasattr(integ_ev, "track_id")

    def test_map_person_lost_identity(self) -> None:
        """Test mapping PersonLostIdentityEvent to PersonLostIdentityIntegrationEvent."""
        domain_ev = PersonLostIdentityEvent(
            event_id="ev-3",
            frame_id=44,
            timestamp=1700000002.0,
            session_id="sess-1",
            track_id=7,
            previous_person_id="p-1",
        )
        integ_ev = map_domain_event_to_integration(domain_ev)
        assert isinstance(integ_ev, PersonLostIdentityIntegrationEvent)
        assert integ_ev.previous_person_id == "p-1"
        assert not hasattr(integ_ev, "track_id")

    def test_map_session_ended(self) -> None:
        """Test mapping SessionEndedEvent to SessionEndedIntegrationEvent."""
        domain_ev = SessionEndedEvent(
            event_id="ev-4",
            frame_id=45,
            timestamp=1700000005.0,
            session_id="sess-1",
            track_id=7,
            final_person_id="p-1",
            final_display_name="Bob",
        )
        integ_ev = map_domain_event_to_integration(domain_ev)
        assert isinstance(integ_ev, SessionEndedIntegrationEvent)
        assert integ_ev.session_id == "sess-1"
        assert integ_ev.final_person_id == "p-1"
        assert not hasattr(integ_ev, "track_id")

    def test_internal_presence_events_return_none(self) -> None:
        """Verify internal perception events are not leaked to external integration."""
        entered = PersonEnteredEvent(track_id=1, person_id="p-1")
        left = PersonLeftEvent(track_id=1, person_id="p-1")
        assert map_domain_event_to_integration(entered) is None
        assert map_domain_event_to_integration(left) is None
