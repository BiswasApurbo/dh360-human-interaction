"""Application dialogue coordinator and runtime orchestrator.

Coordinates spoken and text-based dialogue turns with real DH360 edge session
lifecycle, fresh authorization verification, localized knowledge retrieval,
and speech/language model providers.

Security and Architecture Invariants:
1. Lifecycle-driven session boundaries:
   - Dialogue session history is strictly bound to an active interaction session.
   - Clears session history on session termination (`SessionEndedEvent`) and
     identity changes (different person identified).
   - Rejects stale/mismatched session events from old or inactive sessions (Fix F-04).
2. AuthorizationController as single source of truth:
   - Face identification (`PersonIdentifiedEvent`) alone NEVER grants authorization (Fix F-03).
   - Privileged access is granted strictly when AuthorizationController confirms an
     active, authorized session matching the identified person.
3. Post-generation authorization revalidation:
   - Re-verifies session validity and authorization freshness immediately before TTS audio
     output and before committing dialogue history (Fix F-01).
   - If authorization is revoked or identity lost while LLM is inferring, privileged output
     and audio playback are immediately suppressed and replaced with safe fallback text.
4. Serialized turn execution and safe concurrency:
   - Turn execution is serialized per coordinator to prevent history interleaving and out-of-order
     audio playback, while allowing event-bus lifecycle events to process without delay (Fix F-02).
   - In-flight turns cannot append history to a terminated or replaced session.
5. Delegation to KnowledgeService:
   - Relies strictly on `KnowledgeService.build_knowledge_context` for public and
     privileged context assembly; never duplicates knowledge filtering or DB queries.
6. Safe provider error handling:
   - Translates typed provider errors (STT, LLM, TTS, Network) to safe localized
     Bangla or English fallback messages.
   - Preserves text response when TTS audio synthesis or playback fails.
7. Non-blocking async execution:
   - Provides async execution wrappers (`handle_turn_async`) offloading synchronous
     provider execution to worker threads to prevent blocking the event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

from dh360_hi.contracts.dialogue import (
    VALID_LANGUAGES,
    DialogueSessionState,
    DialogueTurn,
    MockLLMProvider,
    SupportedLanguage,
    get_fallback_message,
)
from dh360_hi.contracts.errors import (
    LLMProviderError,
    ProviderError,
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

if TYPE_CHECKING:
    from dh360_hi.backend.knowledge_service import KnowledgeService
    from dh360_hi.contracts.authorization import AuthorizationSession
    from dh360_hi.contracts.dialogue import LLMProvider, STTProvider, TTSProvider
    from dh360_hi.edge.authorization_controller import AuthorizationController
    from dh360_hi.edge.event_bus import EventBus

logger = logging.getLogger(__name__)


class DialogueCoordinator:
    """Coordinates conversational dialogue sessions with DH360 runtime and lifecycle events."""

    def __init__(
        self,
        event_bus: EventBus,
        knowledge_service: KnowledgeService,
        authorization_controller: AuthorizationController | None = None,
        llm_provider: LLMProvider | None = None,
        stt_provider: STTProvider | None = None,
        tts_provider: TTSProvider | None = None,
        default_language: SupportedLanguage = "bn",
        grace_period_seconds: float = 60.0,
    ) -> None:
        self._event_bus = event_bus
        self._knowledge_service = knowledge_service
        self._auth_controller = authorization_controller
        self._llm_provider = llm_provider
        self._stt_provider = stt_provider
        self._tts_provider = tts_provider
        self._default_language: SupportedLanguage = default_language
        self._grace_period_seconds = grace_period_seconds

        # Lock hierarchy:
        # 1. _turn_lock: serializes turn execution pipeline (STT -> LLM -> TTS -> commit)
        # 2. _state_lock: protects internal session state, lifecycle transitions, and subscriptions
        # Lifecycle events acquire only _state_lock, ensuring they are never blocked by LLM or TTS.
        self._state_lock = threading.RLock()
        self._turn_lock = threading.Lock()
        self._lock = self._state_lock  # Backward compatibility alias

        self._session_state: DialogueSessionState | None = None
        self._subscription_ids: list[int] = []

        self._subscribe_events()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_state(self) -> DialogueSessionState | None:
        """Current dialogue session state snapshot, if active."""
        with self._state_lock:
            return self._session_state

    @property
    def active_session_id(self) -> str | None:
        """ID of the currently active session, or None."""
        with self._state_lock:
            return self._session_state.session_id if self._session_state is not None else None

    @property
    def default_language(self) -> SupportedLanguage:
        """Default response language."""
        return self._default_language

    def get_history(self, now: float | None = None) -> list[DialogueTurn]:
        """Retrieve active dialogue turns enforcing grace-period expiry."""
        with self._state_lock:
            if self._session_state is None:
                return []
            return self._session_state.get_turns(now=now)

    # ------------------------------------------------------------------
    # Authorization Helpers
    # ------------------------------------------------------------------

    def _is_person_authorized(self, person_id: str | None) -> bool:
        """Check whether AuthorizationController actively authorizes the specified person.

        Treats AuthorizationController as the sole source of authorization truth.
        Face identification alone never confers authorization (Fix F-03).
        """
        if person_id is None or self._auth_controller is None:
            return False
        auth = self._auth_controller.get_current()
        return bool(
            auth is not None
            and auth.is_active
            and auth.is_authorized
            and auth.person_id == person_id
        )

    # ------------------------------------------------------------------
    # Lifecycle and EventBus Subscriptions
    # ------------------------------------------------------------------

    def _subscribe_events(self) -> None:
        """Subscribe to edge session and identity domain events."""
        self._subscription_ids.append(
            self._event_bus.subscribe(SessionStartedEvent, self._on_session_started)
        )
        self._subscription_ids.append(
            self._event_bus.subscribe(PersonIdentifiedEvent, self._on_person_identified)
        )
        self._subscription_ids.append(
            self._event_bus.subscribe(PersonLostIdentityEvent, self._on_person_lost_identity)
        )
        self._subscription_ids.append(
            self._event_bus.subscribe(SessionEndedEvent, self._on_session_ended)
        )

    def close(self) -> None:
        """Unsubscribe from EventBus and release active session state. Idempotent."""
        with self._state_lock:
            for sub_id in self._subscription_ids:
                try:
                    self._event_bus.unsubscribe(sub_id)
                except Exception:
                    logger.exception("DialogueCoordinator: error unsubscribing %d", sub_id)
            self._subscription_ids.clear()
            if self._session_state is not None:
                self._session_state.reset()
                self._session_state = None

        if self._tts_provider is not None:
            with contextlib.suppress(Exception):
                self._tts_provider.interrupt()

    def _on_session_started(self, event: SessionStartedEvent) -> None:
        """Handle authoritative session start event."""
        with self._state_lock:
            # Fix F-03: AuthorizationController is sole source of authorization truth
            is_valid_auth = self._is_person_authorized(event.person_id)

            # New interaction session: purge any lingering turns from past sessions
            self._session_state = DialogueSessionState(
                session_id=event.session_id,
                person_id=event.person_id,
                display_name=event.display_name,
                is_authorized=is_valid_auth,
                grace_period_seconds=self._grace_period_seconds,
            )
            logger.debug(
                "DialogueCoordinator: initialized session %s (person: %s, authorized: %s)",
                event.session_id,
                event.person_id,
                is_valid_auth,
            )

    def _on_person_identified(self, event: PersonIdentifiedEvent) -> None:
        """Handle identity confirmation or transition during an active session."""
        different_person = False
        with self._state_lock:
            # Fix F-04: Reject stale or inactive session events.
            # PersonIdentifiedEvent cannot create a session alone or overwrite a different session.
            if self._session_state is None or self._session_state.session_id != event.session_id:
                logger.warning(
                    "DialogueCoordinator: rejecting PersonIdentifiedEvent for inactive or "
                    "mismatched session %s (current: %s)",
                    event.session_id,
                    self._session_state.session_id if self._session_state else None,
                )
                return

            # Fix F-03: Face identification alone never grants authorization.
            is_valid_auth = self._is_person_authorized(event.person_id)

            if self._session_state.person_id == event.person_id:
                # Same person re-identified: attempt grace-period history restoration
                restored = self._session_state.on_reauthorized(
                    person_id=event.person_id,
                    display_name=event.display_name,
                )
                # Overwrite is_authorized to reflect AuthorizationController truth, NOT event alone!
                self._session_state.is_authorized = is_valid_auth
                logger.debug(
                    "DialogueCoordinator: same person %s re-identified "
                    "(history restored: %s, authorized: %s)",
                    event.person_id,
                    restored,
                    is_valid_auth,
                )
            else:
                different_person = self._session_state.person_id is not None
                # Different person identified: purge prior person's history immediately
                logger.info(
                    "DialogueCoordinator: subject changed from %s to %s; purging history",
                    self._session_state.person_id,
                    event.person_id,
                )
                self._session_state.reset(
                    person_id=event.person_id,
                    display_name=event.display_name,
                    is_authorized=is_valid_auth,
                )

        if different_person and self._tts_provider is not None:
            with contextlib.suppress(Exception):
                self._tts_provider.interrupt()

    def _on_person_lost_identity(self, event: PersonLostIdentityEvent) -> None:
        """Handle identity loss: immediately downgrade privilege and start grace period."""
        with self._state_lock:
            if self._session_state is None or self._session_state.session_id != event.session_id:
                return
            logger.info(
                "DialogueCoordinator: identity lost for session %s; downgrading privilege",
                event.session_id,
            )
            self._session_state.on_identity_lost()

        # Immediately abort active TTS playback on identity loss
        if self._tts_provider is not None:
            with contextlib.suppress(Exception):
                self._tts_provider.interrupt()

    def _on_session_ended(self, event: SessionEndedEvent) -> None:
        """Handle session termination: purge history and reset session state."""
        with self._state_lock:
            if (
                self._session_state is not None
                and self._session_state.session_id == event.session_id
            ):
                logger.debug(
                    "DialogueCoordinator: session %s ended; purging dialogue history",
                    event.session_id,
                )
                self._session_state.reset()
                self._session_state = None

        # Immediately abort active TTS playback on session termination
        if self._tts_provider is not None:
            with contextlib.suppress(Exception):
                self._tts_provider.interrupt()

    def interrupt(self) -> None:
        """Abort active speech synthesis playback and microphone listening immediately."""
        if self._tts_provider is not None:
            with contextlib.suppress(Exception):
                self._tts_provider.interrupt()
        if self._stt_provider is not None:
            with contextlib.suppress(Exception):
                if hasattr(self._stt_provider, "interrupt"):
                    self._stt_provider.interrupt()
                self._stt_provider.release()

    def revoke_authorization(self) -> None:
        """Explicitly revoke authorization, clear private history, and abort audio."""
        with self._state_lock:
            if self._session_state is not None:
                self._session_state.reset(person_id=None, display_name=None, is_authorized=False)
                logger.info("DialogueCoordinator: authorization revoked; private history purged")
        self.interrupt()

    # ------------------------------------------------------------------
    # Turn Execution Pipeline
    # ------------------------------------------------------------------

    def handle_turn(
        self,
        user_text: str | None = None,
        language: SupportedLanguage | None = None,
        speak_response: bool = True,
        blocking_tts: bool = False,
        now: float | None = None,
    ) -> DialogueTurn:
        """Execute a full dialogue exchange turn synchronously.

        Pipeline:
        1. Capture utterance via STT (if user_text is None) or use direct text.
        2. Evaluate fresh authorization snapshot via AuthorizationController under lock.
        3. Build pre-filtered knowledge context via KnowledgeService.
        4. Generate response via LLMProvider with isolated session history.
        5. Revalidate session, subject, and authorization status before TTS and commit (Fix F-01).
        6. Synthesize speech via TTSProvider (preserving text response on failure).
        7. Safely commit model turn to history under lock if session unchanged (Fix F-02).
        """
        target_lang = language or self._default_language
        if target_lang not in VALID_LANGUAGES:
            target_lang = "bn"

        # Fix F-02: Serialize turn execution to prevent concurrent interleaving
        with self._turn_lock:
            # 1. Obtain input text (STT transcription or direct text)
            prompt_text = (user_text or "").strip() if user_text is not None else None

            if prompt_text is None:
                if self._stt_provider is None:
                    fallback = get_fallback_message("stt_error", target_lang)
                    return DialogueTurn(
                        role="model",
                        text=fallback,
                        language=target_lang,
                        metadata={"error": "STT_UNAVAILABLE"},
                    )

                try:
                    transcription = self._stt_provider.listen_utterance(language_hint=target_lang)
                    prompt_text = (transcription or "").strip()
                except ProviderNetworkError as exc:
                    fallback = get_fallback_message("network_error", target_lang)
                    return DialogueTurn(
                        role="model",
                        text=fallback,
                        language=target_lang,
                        metadata={"error": exc.code},
                    )
                except (STTProviderError, ProviderError) as exc:
                    fallback = get_fallback_message("stt_error", target_lang)
                    return DialogueTurn(
                        role="model",
                        text=fallback,
                        language=target_lang,
                        metadata={"error": exc.code},
                    )
                except Exception as exc:
                    logger.exception("DialogueCoordinator: unhandled error in STT: %s", exc)
                    fallback = get_fallback_message("stt_error", target_lang)
                    return DialogueTurn(
                        role="model",
                        text=fallback,
                        language=target_lang,
                        metadata={"error": "STT_ERROR"},
                    )

            if not prompt_text:
                fallback = get_fallback_message("stt_error", target_lang)
                return DialogueTurn(
                    role="model",
                    text=fallback,
                    language=target_lang,
                    metadata={"error": "EMPTY_UTTERANCE"},
                )

            # 2. Check fresh authorization snapshot and obtain isolated session history
            active_auth: AuthorizationSession | None = None
            requested_person_id: str | None = None

            with self._state_lock:
                if self._session_state is None:
                    # Transient unauthenticated session for direct interactions
                    self._session_state = DialogueSessionState(
                        session_id=str(uuid.uuid4()),
                        person_id=None,
                        display_name=None,
                        is_authorized=False,
                        grace_period_seconds=self._grace_period_seconds,
                    )

                originating_session_id = self._session_state.session_id
                originating_person_id = self._session_state.person_id

                # Enforce grace-period expiry prior to state evaluation
                self._session_state.get_turns(now=now)

                # Retrieve fresh authorization snapshot on EVERY turn (Fix F-03)
                auth_snapshot = (
                    self._auth_controller.get_current()
                    if self._auth_controller is not None
                    else None
                )

                # Verify if authorization snapshot is active and authorized
                if (
                    auth_snapshot is not None
                    and auth_snapshot.is_active
                    and auth_snapshot.is_authorized
                    and auth_snapshot.person_id is not None
                ):
                    if originating_person_id is None:
                        # Direct/transient dialogue session adopts the newly authorized individual
                        self._session_state.person_id = auth_snapshot.person_id
                        self._session_state.display_name = auth_snapshot.display_name
                        self._session_state.is_authorized = True
                        originating_person_id = auth_snapshot.person_id
                        is_valid_auth = True
                    elif originating_person_id == auth_snapshot.person_id:
                        self._session_state.is_authorized = True
                        is_valid_auth = True
                    else:
                        # Identity mismatch: session belongs to originating_person_id,
                        # but controller authorized auth_snapshot.person_id.
                        # Reject authorization to prevent cross-person leakage!
                        logger.warning(
                            "DialogueCoordinator: authorization person mismatch "
                            "(session: %s, controller: %s); rejecting authorization",
                            originating_person_id,
                            auth_snapshot.person_id,
                        )
                        self._session_state.is_authorized = False
                        is_valid_auth = False
                else:
                    self._session_state.is_authorized = False
                    is_valid_auth = False

                was_privileged = is_valid_auth

                if is_valid_auth:
                    active_auth = auth_snapshot
                    requested_person_id = originating_person_id

                # Retrieve defensive copy of isolated dialogue history before adding this turn
                history_snapshot = self._session_state.get_turns(now=now)

                # Record user turn in dialogue history
                user_turn = DialogueTurn(
                    role="user",
                    text=prompt_text,
                    language=target_lang,
                    timestamp=now if now is not None else time.time(),
                )
                self._session_state.add_turn(user_turn, now=now)

            # 3. Assemble knowledge context (outside state lock)
            knowledge_context = self._knowledge_service.build_knowledge_context(
                query=prompt_text,
                language=target_lang,
                auth_session=active_auth,
                requested_person_id=requested_person_id,
            )

            # 4. Generate response via LLMProvider (outside state lock)
            error_code: str | None = None
            response_text: str

            if self._llm_provider is None:
                response_text = get_fallback_message("llm_error", target_lang)
                error_code = "LLM_UNAVAILABLE"
            else:
                try:
                    try:
                        response_text = self._llm_provider.generate_response(
                            prompt=prompt_text,
                            filtered_context=knowledge_context,
                            language=target_lang,
                            session_history=history_snapshot,
                        )
                    except TypeError as type_err:
                        if "session_history" in str(type_err):
                            response_text = self._llm_provider.generate_response(
                                prompt=prompt_text,
                                filtered_context=knowledge_context,
                                language=target_lang,
                            )
                        else:
                            raise
                except (ProviderNetworkError, ProviderTimeoutError) as exc:
                    response_text = get_fallback_message("network_error", target_lang)
                    error_code = exc.code
                except (LLMProviderError, ProviderError) as exc:
                    response_text = get_fallback_message("llm_error", target_lang)
                    error_code = exc.code
                except Exception as exc:
                    logger.exception("DialogueCoordinator: unhandled error in LLM: %s", exc)
                    response_text = get_fallback_message("llm_error", target_lang)
                    error_code = "LLM_ERROR"

            # 5. Revalidate after LLM generation before TTS / committing (Fix F-01 & Fix F-02)
            with self._state_lock:
                # Check 1: Did the session terminate or get replaced?
                session_alive = (
                    self._session_state is not None
                    and self._session_state.session_id == originating_session_id
                )
                if not session_alive:
                    logger.warning(
                        "DialogueCoordinator: session %s ended or replaced during turn execution",
                        originating_session_id,
                    )
                    return DialogueTurn(
                        role="model",
                        text=get_fallback_message("llm_error", target_lang),
                        language=target_lang,
                        timestamp=now if now is not None else time.time(),
                        metadata={"error": "SESSION_TERMINATED"},
                    )

                # Check 2: Did the identified subject change during generation?
                person_unchanged = self._session_state.person_id == originating_person_id
                if not person_unchanged:
                    logger.warning(
                        "DialogueCoordinator: subject changed from %s to %s during turn execution",
                        originating_person_id,
                        self._session_state.person_id,
                    )
                    return DialogueTurn(
                        role="model",
                        text=get_fallback_message("unauthorized", target_lang),
                        language=target_lang,
                        timestamp=now if now is not None else time.time(),
                        metadata={"error": "IDENTITY_CHANGED"},
                    )

                # Check 3: If turn received privileged context, revalidate authorization freshness
                if was_privileged:
                    fresh_auth = (
                        self._auth_controller.get_current()
                        if self._auth_controller is not None
                        else None
                    )
                    still_authorized = (
                        fresh_auth is not None
                        and fresh_auth.is_active
                        and fresh_auth.is_authorized
                        and fresh_auth.person_id is not None
                        and fresh_auth.person_id == originating_person_id
                        and self._session_state.is_authorized
                    )
                    if not still_authorized:
                        logger.warning(
                            "DialogueCoordinator: authorization revoked or lost for %s "
                            "during generation; suppressing privileged response",
                            originating_person_id,
                        )
                        self._session_state.is_authorized = False
                        response_text = get_fallback_message("unauthorized", target_lang)
                        error_code = "AUTHORIZATION_REVOKED"
                        speak_response = False

                should_speak = (
                    speak_response and (self._tts_provider is not None) and (error_code is None)
                )

            # 6. Playback audio via TTSProvider (outside state lock)
            metadata: dict[str, Any] = {}
            if prompt_text:
                metadata["user_text"] = prompt_text
            if user_text is None:
                metadata["is_voice"] = True
            if isinstance(self._llm_provider, MockLLMProvider):
                metadata["mock"] = True
            if error_code is not None:
                metadata["error"] = error_code
            metadata["is_privileged"] = bool(was_privileged and error_code is None)
            metadata["speak_response"] = bool(should_speak)

            if should_speak and self._tts_provider is not None:
                if self._stt_provider is not None:
                    with contextlib.suppress(Exception):
                        self._stt_provider.hold()

                try:
                    self._tts_provider.speak(response_text, language=target_lang, blocking=blocking_tts)
                except (TTSProviderError, ProviderError) as exc:
                    logger.warning("DialogueCoordinator: TTS playback failed: %s", exc)
                    metadata["tts_error"] = True
                    metadata["tts_error_code"] = exc.code
                except Exception as exc:
                    logger.exception("DialogueCoordinator: unhandled error in TTS: %s", exc)
                    metadata["tts_error"] = True
                finally:
                    if self._stt_provider is not None:
                        with contextlib.suppress(Exception):
                            self._stt_provider.release()

            # 7. Record and return model turn (inside state lock)
            model_turn = DialogueTurn(
                role="model",
                text=response_text,
                language=target_lang,
                timestamp=now if now is not None else time.time(),
                metadata=metadata,
            )

            with self._state_lock:
                if (
                    self._session_state is not None
                    and self._session_state.session_id == originating_session_id
                    and self._session_state.person_id == originating_person_id
                ):
                    self._session_state.add_turn(model_turn, now=now)

            return model_turn

    async def handle_turn_async(
        self,
        user_text: str | None = None,
        language: SupportedLanguage | None = None,
        speak_response: bool = True,
        blocking_tts: bool = False,
        now: float | None = None,
    ) -> DialogueTurn:
        """Asynchronously execute a dialogue turn, offloading synchronous providers to a thread.

        Prevents blocking the FastAPI asynchronous event loop when invoking
        synchronous speech or LLM APIs.
        """
        return await asyncio.to_thread(
            self.handle_turn,
            user_text=user_text,
            language=language,
            speak_response=speak_response,
            blocking_tts=blocking_tts,
            now=now,
        )
