"""Immutable typed domain events for the session and presence lifecycle.

All events are frozen dataclasses that cross the vision → session boundary.
Vision internals (embeddings, images, raw tracks) are NOT exposed here.
``person_id=None`` is valid and represents an unregistered but present person.

Event ordering contract
-----------------------
New interaction::

    PersonEnteredEvent → SessionStartedEvent [→ PersonIdentifiedEvent]

Person departure::

    PersonLeftEvent → SessionEndedEvent

Identity change mid-session::

    PersonLostIdentityEvent → PersonIdentifiedEvent
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseEvent:
    """Common fields shared by all domain events."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    frame_id: int = 0
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Presence events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonEnteredEvent(BaseEvent):
    """Emitted once when a person's presence is confirmed after entry debounce.

    Args:
        track_id:  Vision tracker track identifier.
        person_id: Registered person UUID, or ``None`` for unregistered.
        display_name: Human-readable name, or ``None`` if unknown.
    """

    track_id: int = 0
    person_id: str | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class PersonLeftEvent(BaseEvent):
    """Emitted once when a person's departure is confirmed after leave debounce.

    Args:
        track_id:  Vision tracker track identifier.
        person_id: Registered person UUID at time of departure, or ``None``.
        display_name: Human-readable name at time of departure, or ``None``.
    """

    track_id: int = 0
    person_id: str | None = None
    display_name: str | None = None


# ---------------------------------------------------------------------------
# Session events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionStartedEvent(BaseEvent):
    """Emitted exactly once when a new primary interaction session begins.

    A session may start with ``person_id=None`` (person present but not yet
    or never identified).

    Args:
        session_id:   Opaque UUID identifying this session instance.
        track_id:     Vision tracker track identifier.
        person_id:    Registered person UUID, or ``None``.
        display_name: Human-readable name, or ``None``.
    """

    session_id: str = ""
    track_id: int = 0
    person_id: str | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class SessionEndedEvent(BaseEvent):
    """Emitted exactly once when the primary interaction session ends.

    Args:
        session_id:        UUID of the session that ended.
        track_id:          Vision tracker track identifier.
        final_person_id:   Person UUID at session end, or ``None``.
        final_display_name: Display name at session end, or ``None``.
    """

    session_id: str = ""
    track_id: int = 0
    final_person_id: str | None = None
    final_display_name: str | None = None


# ---------------------------------------------------------------------------
# Identity change events (within a session)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonIdentifiedEvent(BaseEvent):
    """Emitted when the active session target transitions to IDENTIFIED state.

    Not emitted again if the same person is already identified in this session
    (deduplication enforced by SessionManager).

    Args:
        session_id:   UUID of the currently active session.
        track_id:     Vision tracker track identifier.
        person_id:    Confirmed registered person UUID (never ``None``).
        display_name: Human-readable name, or ``None``.
    """

    session_id: str = ""
    track_id: int = 0
    person_id: str = ""
    display_name: str | None = None


@dataclass(frozen=True)
class PersonLostIdentityEvent(BaseEvent):
    """Emitted when the active session target loses a previously stable identity.

    Typically precedes a ``PersonIdentifiedEvent`` for a new identity.

    Args:
        session_id:         UUID of the currently active session.
        track_id:           Vision tracker track identifier.
        previous_person_id: The person UUID that was lost, or ``None`` if the
                            target was UNREGISTERED before becoming ambiguous.
    """

    session_id: str = ""
    track_id: int = 0
    previous_person_id: str | None = None
