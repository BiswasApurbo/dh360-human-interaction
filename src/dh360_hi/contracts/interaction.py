"""Domain contracts and data models for application-level human interaction state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dh360_hi.backend.models import PersonProfile


class InteractionState(StrEnum):
    """Deterministic application lifecycle states for an active human interaction session."""

    IDLE = "idle"
    GREETING = "greeting"
    ENGAGED = "engaged"
    FAREWELL = "farewell"


@dataclass(frozen=True)
class InteractionContext:
    """Immutable snapshot of current application-level interaction context.

    Attributes:
        session_id: Active session UUID identifier, or None if in IDLE state.
        person_id: Identified person UUID identifier, or None if unregistered/anonymous.
        display_name: Human-readable display name of the subject, if known.
        person_profile: Public profile record if subject is registered and resolved.
        state: Current interaction lifecycle state.
        started_at: Timestamp when the active session began, or None if IDLE.
        last_updated_at: Timestamp when this context snapshot was last updated.
        identity_known: Whether a confirmed identity is currently active.
        last_known_person_id: Previous person_id if identity degraded mid-session.
        identity_transitions_count: Number of identity state changes within this session.
    """

    session_id: str | None = None
    person_id: str | None = None
    display_name: str | None = None
    person_profile: PersonProfile | None = None
    state: InteractionState = InteractionState.IDLE
    started_at: float | None = None
    last_updated_at: float = 0.0
    identity_known: bool = False
    last_known_person_id: str | None = None
    identity_transitions_count: int = 0


@dataclass(frozen=True)
class InteractionSummary:
    """Immutable historical summary record of a completed interaction session.

    Attributes:
        session_id: Unique session UUID identifier.
        person_id: Final person UUID identifier at session end, if identified.
        display_name: Final display name of the subject at session end, if known.
        person_profile: Public profile record if subject was registered and resolved.
        started_at: Timestamp when the session began.
        ended_at: Timestamp when the session terminated.
        duration_seconds: Total duration of the interaction session in seconds.
        final_state: Final interaction state prior to termination (e.g. FAREWELL / ENGAGED).
        identity_was_known: Whether the subject was identified at any point during the session.
        identity_transitions_count: Total identity transitions recorded during the session.
    """

    session_id: str
    person_id: str | None
    display_name: str | None
    person_profile: PersonProfile | None
    started_at: float
    ended_at: float
    duration_seconds: float
    final_state: InteractionState
    identity_was_known: bool
    identity_transitions_count: int
