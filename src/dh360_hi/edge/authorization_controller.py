"""Authorization Controller for edge interaction.

This module implements the edge-side AuthorizationController which observes
`SessionManager` frame results and produces an `AuthorizationSession`
that can be consumed by the API layer.

Key invariants:
  * Registers exactly one observer callback with `SessionManager` on construction.
  * Unregisters the observer when `close()` is called (idempotent).
  * Never calls `process_frame_result`, `runtime.step_once`, DB, or network I/O.
  * Uses a monotonic deadline to enforce a configurable timeout.
  * Only a single active authorization session may exist at a time.
  * `AuthorizationSession` is a frozen/immutable dataclass; state evolution
    uses its `transition()` and `complete()` methods without mutating fields.
  * Privacy: Exposes only transport-neutral fields in `AuthorizationSession`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from dh360_hi.contracts.authorization import (
    AuthorizationSession,
    AuthorizationStatus,
    derive_authorization_status,
)
from dh360_hi.vision.identity_state import IdentityState

if TYPE_CHECKING:
    from dh360_hi.edge.session_manager import SessionManager
    from dh360_hi.vision.two_rate_pipeline import TwoRateFrameResult

_logger = logging.getLogger(__name__)


class AuthorizationController:
    """Controller that derives an :class:`AuthorizationSession` from frame results."""

    def __init__(self, session_manager: SessionManager, timeout_seconds: float = 30.0) -> None:
        self._session_manager = session_manager
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._session: AuthorizationSession | None = None
        self._deadline: float | None = None
        self._closed: bool = False

        # Register observer callback immediately
        self._session_manager.add_observer(self._on_frame)
        _logger.debug("AuthorizationController registered observer with SessionManager")

    def start(self) -> AuthorizationSession:
        """Start a new ``ACTIVE`` authorization session.

        Returns the newly created :class:`AuthorizationSession`.
        Raises ``ValueError`` if a session is already active.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("AuthorizationController is closed")
            if self._session is not None and self._session.is_active:
                raise ValueError("An active authorization session already exists")

            new_session = AuthorizationSession()
            self._session = new_session
            self._deadline = time.monotonic() + self._timeout_seconds
            _logger.debug(
                "AuthorizationController started new session %s with deadline %s",
                new_session.authorization_id,
                self._deadline,
            )
            return new_session

    def get_current(self) -> AuthorizationSession | None:
        """Return the latest session snapshot (may be ``None``)."""
        with self._lock:
            if (
                self._session is not None
                and self._session.is_active
                and self._deadline is not None
                and time.monotonic() >= self._deadline
            ):
                try:
                    self._session = self._session.complete(AuthorizationStatus.TIMEOUT)
                    _logger.debug("AuthorizationController session timed out")
                except Exception as exc:
                    _logger.exception("Failed to complete session on timeout: %s", exc)
            return self._session

    def close(self) -> None:
        """Unregister the observer and mark the controller closed.

        This method is idempotent.
        """
        with self._lock:
            if not self._closed:
                try:
                    self._session_manager.remove_observer(self._on_frame)
                    _logger.debug("AuthorizationController observer unregistered")
                except Exception:
                    _logger.exception("Error unregistering observer in AuthorizationController")
                self._closed = True

    def _on_frame(self, result: TwoRateFrameResult) -> None:
        """Observer invoked by ``SessionManager`` after processing each frame."""
        try:
            with self._lock:
                if self._session is None or not self._session.is_active:
                    return

                # Check timeout on each frame observation
                if self._deadline is not None and time.monotonic() >= self._deadline:
                    self._session = self._session.complete(AuthorizationStatus.TIMEOUT)
                    return

                target_result = result.target_result
                primary_track_id = target_result.primary_track_id if target_result else None

                track_state = None
                if primary_track_id is not None:
                    track_state = result.identity_states.get(primary_track_id)
                identity_state = track_state.state if track_state is not None else None

                previously_authorized = self._session.is_authorized
                new_status = derive_authorization_status(
                    identity_state,
                    target_result=target_result,
                    previously_authorized=previously_authorized,
                )

                if previously_authorized:
                    if new_status is AuthorizationStatus.LOST:
                        # LOST is a terminal outcome following authorization loss
                        self._session = self._session.complete(AuthorizationStatus.LOST)
                    # If still AUTHORIZED, session stays running as AUTHORIZED
                    return

                # Pre-authorization evolution:
                if new_status is AuthorizationStatus.AUTHORIZED:
                    if (
                        track_state is not None
                        and track_state.state is IdentityState.IDENTIFIED
                        and track_state.current_person_id
                    ):
                        self._session = self._session.transition(
                            AuthorizationStatus.AUTHORIZED,
                            person_id=track_state.current_person_id,
                            display_name=track_state.current_display_name,
                        )
                elif (
                    new_status
                    in (
                        AuthorizationStatus.UNKNOWN,
                        AuthorizationStatus.UNCERTAIN,
                        AuthorizationStatus.NO_FACE,
                    )
                    and self._session.status is not new_status
                ):
                    self._session = self._session.transition(new_status)
                elif (
                    new_status is AuthorizationStatus.ACTIVE
                    and self._session.status is not AuthorizationStatus.ACTIVE
                ):
                    self._session = self._session.transition(AuthorizationStatus.ACTIVE)

        except Exception as exc:
            # Observer errors must not propagate to SessionManager
            _logger.exception("AuthorizationController observer error: %s", exc)
