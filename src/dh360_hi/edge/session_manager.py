"""Session lifecycle manager: converts raw perception results into typed domain events.

Architecture
------------
``SessionManager`` sits directly above the two-rate vision pipeline.  It
consumes :class:`~dh360_hi.vision.two_rate_pipeline.TwoRateFrameResult` and
produces an ordered ``list[BaseEvent]`` that represents the authoritative
semantic view of the current primary interaction session.

Design decisions
----------------
Dual output
    ``process_frame_result()`` **returns** ``list[BaseEvent]`` AND optionally
    **publishes** each event to an :class:`~dh360_hi.edge.event_bus.EventBus`
    if one was provided at construction.  This keeps ``SessionManager``
    independently unit-testable (inspect the return value) while still wiring
    into the bus in production.

Primary session only
    ``SessionManager`` manages exactly **one** primary session at a time.
    Background persons that are not the interaction target do NOT receive
    sessions, ``PersonEnteredEvent``, ``PersonIdentifiedEvent``, or
    ``PersonLeftEvent``.  Only the person selected as the primary interaction
    target drives domain events.

Target switching semantics
    * If the primary target changes while the active target is still present,
      no fake ``PersonLeftEvent`` is emitted, the session is NOT ended, and
      no second session is created. The current session is retained.
    * A session ends only when a real ``PERSON_LEFT`` transition is emitted by
      the presence debouncer for the currently active session's track.

Identity tracking within a session
    * ``PersonIdentifiedEvent`` is emitted at most **once per stable identity**
      per session (deduplicated by ``_last_emitted_person_id``).
    * When a previously identified person loses their stable identity (state
      transitions away from IDENTIFIED), ``PersonLostIdentityEvent`` is
      emitted without terminating the session, resetting the deduplication
      guard so that re-identification later emits a fresh
      ``PersonIdentifiedEvent``.

Event ordering guarantees (per-frame)
    New interaction::

        PersonEnteredEvent → SessionStartedEvent [→ PersonIdentifiedEvent]

    Departure::

        PersonLeftEvent → SessionEndedEvent

    Identity change mid-session::

        PersonLostIdentityEvent → PersonIdentifiedEvent
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from dh360_hi.contracts.events import (
    BaseEvent,
    PersonEnteredEvent,
    PersonIdentifiedEvent,
    PersonLeftEvent,
    PersonLostIdentityEvent,
    SessionEndedEvent,
    SessionStartedEvent,
)
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.vision.identity_state import IdentityState
from dh360_hi.vision.presence_debouncer import PresenceEvent
from dh360_hi.vision.two_rate_pipeline import TwoRateFrameResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionConfig:
    """Tuning parameters for session lifecycle decisions.

    Attributes:
        require_confirmed_presence: If ``True`` (default), a session starts
            only after the presence debouncer has confirmed presence for the
            primary track (via ``PERSON_ENTERED``). If ``False``, a session starts
            as soon as the target selector nominates a primary track — useful
            for tests with very tight frame budgets.
    """

    require_confirmed_presence: bool = True


# ---------------------------------------------------------------------------
# Internal session state
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """Mutable state for the single active primary interaction session.

    Attributes:
        session_id:             Opaque UUID for this session instance.
        track_id:               Vision tracker track for the primary person.
        person_id:              Registered person UUID, or ``None`` for
                                unregistered.
        display_name:           Human-readable name, or ``None``.
        start_frame_id:         Frame at which the session started.
        start_timestamp:        Epoch seconds at session start.
        is_identified:          ``True`` once ``PersonIdentifiedEvent`` has
                                been emitted for a stable identity this session.
        last_emitted_person_id: The ``person_id`` of the most recently emitted
                                ``PersonIdentifiedEvent``, or ``None``.  Used
                                to suppress duplicate identity events.
    """

    session_id: str
    track_id: int
    person_id: str | None
    display_name: str | None
    start_frame_id: int
    start_timestamp: float
    is_identified: bool = False
    last_emitted_person_id: str | None = None


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------


class SessionManager:
    """Converts per-frame perception results into ordered semantic domain events.

    Args:
        event_bus: Optional :class:`~dh360_hi.edge.event_bus.EventBus`. When
                   provided, every event returned by
                   :meth:`process_frame_result` is also published to the bus.
        config:    Optional :class:`SessionConfig` for tuning lifecycle
                   thresholds.

    Example::

        bus = EventBus()
        manager = SessionManager(event_bus=bus)
        events = manager.process_frame_result(frame_result)
        # events is a list[BaseEvent] in guaranteed order
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        config: SessionConfig | None = None,
    ) -> None:
        self._bus = event_bus
        self._config = config or SessionConfig()

        # Active primary session; None when no interaction is in progress
        self._active_session: SessionState | None = None

        # Set of track IDs currently confirmed present by presence debouncer
        self._present_tracks: set[int] = set()

        # Per-frame observers receive the raw pipeline result after normal processing.
        self._observers: list[Callable[[TwoRateFrameResult], None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active_session(self) -> SessionState | None:
        """The currently active primary session, or ``None``."""
        return self._active_session

    @property
    def config(self) -> SessionConfig:
        """Session lifecycle configuration."""
        return self._config

    def reset(self) -> None:
        """Reset all internal session state (e.g. between test runs)."""
        self._active_session = None
        self._present_tracks.clear()

    def add_observer(self, callback: Callable[[TwoRateFrameResult], None]) -> None:
        """Register a post-processing observer for raw frame results."""
        if callback not in self._observers:
            self._observers.append(callback)

    def remove_observer(self, callback: Callable[[TwoRateFrameResult], None]) -> None:
        """Unregister a frame-result observer if present."""
        try:
            self._observers.remove(callback)
        except ValueError:
            return

    def invalidate_person_identity(
        self,
        person_id: str,
        *,
        deleted: bool = False,
        timestamp: float | None = None,
    ) -> PersonLostIdentityEvent | None:
        """Reconcile an operator mutation with the active session without ending it.

        Deletion detaches the person ID so a later session summary cannot violate
        the person foreign key. Reset/update preserve the historical association.
        """
        session = self._active_session
        if session is None:
            return None

        was_emitted = session.last_emitted_person_id == person_id
        if session.person_id != person_id and not was_emitted:
            return None

        event = None
        if was_emitted:
            event = PersonLostIdentityEvent(
                timestamp=timestamp if timestamp is not None else time.time(),
                session_id=session.session_id,
                track_id=session.track_id,
                previous_person_id=person_id,
            )
            session.is_identified = False
            session.last_emitted_person_id = None

        if deleted:
            session.person_id = None

        if event is not None and self._bus is not None:
            self._bus.publish(event)
        return event

    def process_frame_result(self, result: TwoRateFrameResult) -> list[BaseEvent]:
        """Process a single pipeline frame result and return ordered domain events.

        All returned events are also published to the ``EventBus`` if one was
        provided at construction time.

        Args:
            result: A :class:`~dh360_hi.vision.two_rate_pipeline.TwoRateFrameResult`
                    from the two-rate perception pipeline.

        Returns:
            An ordered ``list[BaseEvent]``. Within a single call the ordering
            guarantees hold (see module docstring). The list may be empty.
        """
        events: list[BaseEvent] = []

        # 1. Update internal presence tracking from presence transitions
        for pt in result.presence_transitions:
            if pt.event == PresenceEvent.PERSON_ENTERED:
                self._present_tracks.add(pt.track_id)
            elif pt.event == PresenceEvent.PERSON_LEFT:
                self._present_tracks.discard(pt.track_id)

        # 2. Handle departure of the ACTIVE session's track
        if self._active_session is not None:
            active_track = self._active_session.track_id
            active_left = any(
                pt.event == PresenceEvent.PERSON_LEFT and pt.track_id == active_track
                for pt in result.presence_transitions
            )

            if active_left:
                session = self._active_session
                events.append(
                    PersonLeftEvent(
                        frame_id=result.frame_id,
                        timestamp=result.timestamp,
                        track_id=session.track_id,
                        person_id=session.person_id,
                        display_name=session.display_name,
                    )
                )
                events.append(
                    SessionEndedEvent(
                        frame_id=result.frame_id,
                        timestamp=result.timestamp,
                        session_id=session.session_id,
                        track_id=session.track_id,
                        final_person_id=session.person_id,
                        final_display_name=session.display_name,
                    )
                )
                self._active_session = None

        # 3. Session start for primary target
        primary_track_id = result.target_result.primary_track_id

        if self._active_session is None and primary_track_id is not None:
            can_start = (not self._config.require_confirmed_presence) or (
                primary_track_id in self._present_tracks
            )
            if can_start:
                id_state = result.identity_states.get(primary_track_id)
                person_id = id_state.current_person_id if id_state else None
                display_name = id_state.current_display_name if id_state else None

                session_id = str(uuid.uuid4())
                self._active_session = SessionState(
                    session_id=session_id,
                    track_id=primary_track_id,
                    person_id=person_id,
                    display_name=display_name,
                    start_frame_id=result.frame_id,
                    start_timestamp=result.timestamp,
                )

                # Order: PersonEnteredEvent -> SessionStartedEvent -> [PersonIdentifiedEvent]
                events.append(
                    PersonEnteredEvent(
                        frame_id=result.frame_id,
                        timestamp=result.timestamp,
                        track_id=primary_track_id,
                        person_id=person_id,
                        display_name=display_name,
                    )
                )
                events.append(
                    SessionStartedEvent(
                        frame_id=result.frame_id,
                        timestamp=result.timestamp,
                        session_id=session_id,
                        track_id=primary_track_id,
                        person_id=person_id,
                        display_name=display_name,
                    )
                )

                if (
                    id_state is not None
                    and id_state.state == IdentityState.IDENTIFIED
                    and id_state.current_person_id is not None
                ):
                    events.append(
                        PersonIdentifiedEvent(
                            frame_id=result.frame_id,
                            timestamp=result.timestamp,
                            session_id=session_id,
                            track_id=primary_track_id,
                            person_id=id_state.current_person_id,
                            display_name=id_state.current_display_name,
                        )
                    )
                    self._active_session.is_identified = True
                    self._active_session.last_emitted_person_id = id_state.current_person_id

        # 4. Mid-session identity change detection (for active session)
        elif self._active_session is not None:
            session = self._active_session
            id_state = result.identity_states.get(session.track_id)

            if id_state is not None:
                # Update the session's cached person_id/display_name
                if id_state.current_person_id is not None:
                    session.person_id = id_state.current_person_id
                    session.display_name = id_state.current_display_name

                current_is_identified = id_state.state == IdentityState.IDENTIFIED
                current_person_id = id_state.current_person_id

                if current_is_identified and current_person_id is not None:
                    if current_person_id != session.last_emitted_person_id:
                        # New or changed identity — emit LostIdentity first if previously identified
                        if session.last_emitted_person_id is not None:
                            events.append(
                                PersonLostIdentityEvent(
                                    frame_id=result.frame_id,
                                    timestamp=result.timestamp,
                                    session_id=session.session_id,
                                    track_id=session.track_id,
                                    previous_person_id=session.last_emitted_person_id,
                                )
                            )
                        events.append(
                            PersonIdentifiedEvent(
                                frame_id=result.frame_id,
                                timestamp=result.timestamp,
                                session_id=session.session_id,
                                track_id=session.track_id,
                                person_id=current_person_id,
                                display_name=id_state.current_display_name,
                            )
                        )
                        session.is_identified = True
                        session.last_emitted_person_id = current_person_id

                elif not current_is_identified and session.last_emitted_person_id is not None:
                    # Was identified, now lost stable identity
                    events.append(
                        PersonLostIdentityEvent(
                            frame_id=result.frame_id,
                            timestamp=result.timestamp,
                            session_id=session.session_id,
                            track_id=session.track_id,
                            previous_person_id=session.last_emitted_person_id,
                        )
                    )
                    session.is_identified = False
                    session.last_emitted_person_id = None

        # 5. Publish all events to the bus (if wired)
        if self._bus is not None:
            for event in events:
                self._bus.publish(event)

        # 6. Notify raw frame-result observers after normal processing.
        for observer in list(self._observers):
            try:
                observer(result)
            except Exception:
                logger.exception("SessionManager observer raised an unhandled exception")

        return events
