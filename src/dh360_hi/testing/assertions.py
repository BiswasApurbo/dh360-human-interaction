"""Assertion and validation utilities for deterministic scenario replay."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dh360_hi.contracts.dto import InteractionSessionDTO
from dh360_hi.contracts.integration import (
    BaseIntegrationEvent,
    PersonIdentifiedIntegrationEvent,
    PersonLostIdentityIntegrationEvent,
    SessionEndedIntegrationEvent,
    SessionStartedIntegrationEvent,
)
from dh360_hi.testing.contracts import ExpectedIntegrationEvent

if TYPE_CHECKING:
    pass


def assert_scenario_events(
    collected_events: list[BaseIntegrationEvent],
    expected_events: list[ExpectedIntegrationEvent],
    allow_context_updates: bool = True,
) -> None:
    """Assert that collected integration events strictly match the expected event sequence.

    Args:
        collected_events: List of actual BaseIntegrationEvent instances emitted during replay.
        expected_events: List of ExpectedIntegrationEvent specifications to match against.
        allow_context_updates: If True (default), ignores 'context_updated' events during matching.

    Raises:
        AssertionError: If event count, sequence, types, or field values do not match expectations.
    """
    events_to_check = (
        [e for e in collected_events if e.event_type != "context_updated"]
        if allow_context_updates
        else list(collected_events)
    )

    actual_types = [e.event_type for e in events_to_check]
    expected_types = [e.event_type for e in expected_events]

    if len(events_to_check) != len(expected_events):
        msg = (
            f"Event count mismatch: expected {len(expected_events)} events {expected_types}, "
            f"but got {len(events_to_check)} events {actual_types}"
        )
        raise AssertionError(msg)

    for i, (actual, expected) in enumerate(zip(events_to_check, expected_events, strict=True)):
        # 1. Check Event Type
        if actual.event_type != expected.event_type:
            msg = (
                f"Event #{i} type mismatch: expected '{expected.event_type}', "
                f"got '{actual.event_type}' (full sequence: {actual_types})"
            )
            raise AssertionError(msg)

        # 2. Check Person ID
        if expected.person_id is not None:
            actual_person_id: str | None = None
            if isinstance(
                actual,
                (SessionStartedIntegrationEvent, PersonIdentifiedIntegrationEvent),
            ):
                actual_person_id = actual.person_id
            elif isinstance(actual, SessionEndedIntegrationEvent):
                actual_person_id = actual.final_person_id

            if actual_person_id != expected.person_id:
                msg = (
                    f"Event #{i} ({actual.event_type}) person_id mismatch: "
                    f"expected '{expected.person_id}', got '{actual_person_id}'"
                )
                raise AssertionError(msg)

        # 3. Check Display Name
        if expected.display_name is not None:
            actual_display_name: str | None = None
            if isinstance(
                actual,
                (SessionStartedIntegrationEvent, PersonIdentifiedIntegrationEvent),
            ):
                actual_display_name = actual.display_name
            elif isinstance(actual, SessionEndedIntegrationEvent):
                actual_display_name = actual.final_display_name

            if actual_display_name != expected.display_name:
                msg = (
                    f"Event #{i} ({actual.event_type}) display_name mismatch: "
                    f"expected '{expected.display_name}', got '{actual_display_name}'"
                )
                raise AssertionError(msg)

        # 4. Check Previous Person ID (for PersonLostIdentity)
        if (
            expected.previous_person_id is not None
            and isinstance(actual, PersonLostIdentityIntegrationEvent)
            and actual.previous_person_id != expected.previous_person_id
        ):
            msg = (
                f"Event #{i} previous_person_id mismatch: "
                f"expected '{expected.previous_person_id}', "
                f"got '{actual.previous_person_id}'"
            )
            raise AssertionError(msg)

        # 5. Check Minimum Duration for SessionEnded
        if expected.min_duration_seconds is not None and isinstance(
            actual, SessionEndedIntegrationEvent
        ):
            if actual.summary is None:
                msg = f"Event #{i} (session_ended) missing summary DTO"
                raise AssertionError(msg)
            if actual.summary.duration_seconds < expected.min_duration_seconds:
                msg = (
                    f"Event #{i} (session_ended) duration too short: "
                    f"expected >= {expected.min_duration_seconds}s, "
                    f"got {actual.summary.duration_seconds}s"
                )
                raise AssertionError(msg)


def assert_persisted_session_matches(
    session: InteractionSessionDTO,
    expected_person_id: str | None = None,
    min_duration: float | None = None,
) -> None:
    """Assert properties of a persisted historical interaction session record."""
    if expected_person_id is not None and session.person_id != expected_person_id:
        msg = (
            f"Session person_id mismatch: expected '{expected_person_id}', "
            f"got '{session.person_id}'"
        )
        raise AssertionError(msg)

    if min_duration is not None and session.duration_seconds < min_duration:
        msg = (
            f"Session duration_seconds mismatch: expected >= {min_duration}s, "
            f"got {session.duration_seconds}s"
        )
        raise AssertionError(msg)
