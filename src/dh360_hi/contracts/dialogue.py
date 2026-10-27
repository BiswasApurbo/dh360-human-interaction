"""Contracts and protocol definitions for conversational speech, dialogue, and language models."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

SupportedLanguage = Literal["bn", "en"]
VALID_LANGUAGES: tuple[SupportedLanguage, ...] = ("bn", "en")
DialogueRole = Literal["user", "model", "system"]
VALID_ROLES: tuple[DialogueRole, ...] = ("user", "model", "system")


@dataclass(frozen=True)
class DialogueTurn:
    """Immutable single exchange turn within a dialogue session.

    Attributes:
        role: Dialogue participant role ('user', 'model', or 'system').
        text: Plaintext content of the utterance or response.
        timestamp: Epoch seconds when the turn occurred.
        language: Language associated with this turn ('bn' or 'en').
        metadata: Optional auxiliary non-sensitive metadata dictionary.
    """

    role: DialogueRole
    text: str
    timestamp: float = field(default_factory=time.time)
    language: SupportedLanguage = "bn"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"Invalid dialogue role '{self.role}'; must be one of {VALID_ROLES}")
        if self.language not in VALID_LANGUAGES:
            raise ValueError(
                f"Invalid language '{self.language}'; must be one of {VALID_LANGUAGES}"
            )
        if not isinstance(self.text, str):
            raise TypeError("Dialogue text must be a string")

    def to_dict(self) -> dict[str, Any]:
        """Convert dialogue turn to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class DialogueSessionState:
    """In-memory dialogue turns, active person identity, and grace period manager.

    Invariants:
    1. Privilege Downgrade: As soon as identity is lost or invalidated, `is_authorized`
       is immediately set to False. Any ongoing or incoming turn receives only public context.
    2. Grace Period: A 60-second in-memory grace period retains dialogue turns for potential
       recovery if the same person looks away and returns, but NEVER retains privileged access.
    3. Fresh Authorization Required: History is restored only upon reliable re-identification
       and fresh authorization of the SAME person within the grace period.
    4. Enforced Expiry: Grace-period expiry is actively evaluated on every history access,
       retrieval, and mutation. Expired history is purged immediately.
    5. Monotonic Time: All elapsed-time and timeout calculations strictly use a monotonic clock.
    """

    def __init__(
        self,
        session_id: str,
        person_id: str | None = None,
        display_name: str | None = None,
        is_authorized: bool = False,
        turns: list[DialogueTurn] | None = None,
        identity_lost_at: float | None = None,
        grace_period_seconds: float = 60.0,
    ) -> None:
        self.session_id = session_id
        self.person_id = person_id
        self.display_name = display_name
        self.is_authorized = is_authorized
        self.identity_lost_at = identity_lost_at
        self.grace_period_seconds = grace_period_seconds
        self._turns: list[DialogueTurn] = list(turns) if turns is not None else []

    def _enforce_expiry(self, now: float | None = None) -> bool:
        """Purge turns if the grace period has elapsed.

        Returns True if history was purged due to expiry, False otherwise.
        """
        if self.identity_lost_at is None:
            return False
        current_time = now if now is not None else time.monotonic()
        if (current_time - self.identity_lost_at) >= self.grace_period_seconds:
            self._turns.clear()
            self.identity_lost_at = None
            return True
        return False

    @property
    def turns(self) -> list[DialogueTurn]:
        """Controlled view of active dialogue turns, enforcing grace-period expiry."""
        self._enforce_expiry()
        return list(self._turns)

    @property
    def is_in_grace_period(self) -> bool:
        """Return True if identity was lost and grace period has not yet expired."""
        if self.identity_lost_at is None:
            return False
        return (time.monotonic() - self.identity_lost_at) < self.grace_period_seconds

    @property
    def is_grace_expired(self) -> bool:
        """Return True if grace period has elapsed."""
        if self.identity_lost_at is None:
            return False
        return (time.monotonic() - self.identity_lost_at) >= self.grace_period_seconds

    def check_grace_expired(self, now: float | None = None) -> bool:
        """Evaluate grace-period expiry at a specified monotonic timestamp."""
        if self.identity_lost_at is None:
            return False
        current_time = now if now is not None else time.monotonic()
        return (current_time - self.identity_lost_at) >= self.grace_period_seconds

    def get_turns(self, now: float | None = None) -> list[DialogueTurn]:
        """Retrieve active dialogue turns, enforcing grace-period expiry.

        Args:
            now: Optional monotonic timestamp for deterministic evaluation.

        Returns:
            Defensive copy of active dialogue turns. If grace expired, returns empty list.
        """
        self._enforce_expiry(now=now)
        return list(self._turns)

    def add_turn(self, turn: DialogueTurn, now: float | None = None) -> None:
        """Append a dialogue turn, enforcing grace-period expiry prior to addition.

        Args:
            turn: Dialogue turn to record.
            now: Optional monotonic timestamp for deterministic evaluation.
        """
        self._enforce_expiry(now=now)
        self._turns.append(turn)

    def on_identity_lost(self, now: float | None = None) -> None:
        """Immediately revoke authorization and start the monotonic grace period timer.

        CRITICAL SECURITY INVARIANT:
        `is_authorized` is immediately set to False. Subsequent knowledge retrieval
        must prune all privileged facts, offering only public facts during grace.
        """
        self.is_authorized = False
        if self.identity_lost_at is None:
            self.identity_lost_at = now if now is not None else time.monotonic()

    def on_reauthorized(
        self,
        person_id: str,
        display_name: str | None = None,
        now: float | None = None,
    ) -> bool:
        """Restore conversation history if the same person is re-identified within grace.

        Args:
            person_id: Newly confirmed person ID from face recognition.
            display_name: Optional human-readable display name.
            now: Optional monotonic timestamp (useful for deterministic testing).

        Returns:
            True if previous conversation history was safely restored.
            False if grace period expired or identity mismatched, triggering a purge.
        """
        current_time = now if now is not None else time.monotonic()
        was_in_grace = False
        if self.identity_lost_at is not None:
            was_in_grace = (current_time - self.identity_lost_at) < self.grace_period_seconds

        if self.person_id == person_id and was_in_grace:
            self.is_authorized = True
            self.identity_lost_at = None
            if display_name:
                self.display_name = display_name
            return True

        # Either a new subject arrived or grace period expired: purge old history cleanly
        self.reset(person_id=person_id, display_name=display_name, is_authorized=True)
        return False

    def reset(
        self,
        person_id: str | None = None,
        display_name: str | None = None,
        is_authorized: bool = False,
    ) -> None:
        """Purge all dialogue turns and reset authorization status."""
        self.person_id = person_id
        self.display_name = display_name
        self.is_authorized = is_authorized
        self._turns.clear()
        self.identity_lost_at = None


# ==============================================================================
# Localized Fallback Text Mapping
# ==============================================================================

FALLBACK_MESSAGES: dict[str, dict[SupportedLanguage, str]] = {
    "stt_error": {
        "bn": "আমি আপনার কথাটি স্পষ্টভাবে শুনতে পাইনি। অনুগ্রহ করে আবার বলুন।",
        "en": "I could not hear you clearly. Please say that again.",
    },
    "llm_error": {
        "bn": "আমি বর্তমানে তথ্য প্রক্রিয়াকরণ করতে পারছি না। অনুগ্রহ করে কিছুক্ষণ পর চেষ্টা করুন।",
        "en": "I am currently unable to process your request. Please try again shortly.",
    },
    "tts_error": {
        "bn": "অডিও আউটপুটে সমস্যা হয়েছে, তবে স্ক্রিনে উত্তরটি দেখানো হচ্ছে।",
        "en": "There was an issue with audio playback, but the response is displayed on screen.",
    },
    "network_error": {
        "bn": "নেটওয়ার্ক সংযোগে সমস্যা হচ্ছে। অনুগ্রহ করে সংযোগটি পরীক্ষা করুন।",
        "en": "There is a network connection issue. Please check your connection.",
    },
    "unauthorized": {
        "bn": "এই তথ্যটি সুরক্ষিত। এটি জানার জন্য আপনাকে অনুমোদিত হতে হবে।",
        "en": "This information is restricted. You must be authorized to access it.",
    },
}


def get_fallback_message(error_key: str, language: SupportedLanguage = "bn") -> str:
    """Retrieve the dialogue-coordinator selected localized fallback text.

    Providers must not embed localized strings or swallow exceptions. The coordinator
    maps typed provider exceptions to user-facing fallback text in the selected language.
    """
    messages = FALLBACK_MESSAGES.get(error_key, FALLBACK_MESSAGES["llm_error"])
    return messages.get(language, messages["bn"])


# ==============================================================================
# Provider Protocols
# ==============================================================================


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for generative conversational models (e.g. Gemini).

    The LLM provider generates conversational responses using pre-filtered knowledge
    context supplied by DH360. The provider does NOT evaluate authorization or read
    raw databases.
    """

    def generate_response(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> str:
        """Generate a single dialogue response.

        Args:
            prompt: User utterance or query.
            filtered_context: Knowledge facts already filtered and cleared by DH360.
            language: Target response language ('bn' for Bangla, 'en' for English).
            session_history: Isolated turn history scoped only to the current person's
                             active session. Never uses shared global history.

        Returns:
            Plaintext response text formatted for natural speech output.
        """
        ...


@runtime_checkable
class STTProvider(Protocol):
    """Protocol for speech-to-text transcription engines (e.g. Soniox)."""

    def listen_utterance(
        self,
        language_hint: SupportedLanguage = "bn",
        on_partial_text: Callable[[str], None] | None = None,
    ) -> str:
        """Capture and transcribe a single spoken utterance.

        Args:
            language_hint: Target language hint for acoustic models ('bn' or 'en').
            on_partial_text: Optional live callback for partial transcriptions.

        Returns:
            Transcribed utterance text.
        """
        ...

    def hold(self) -> None:
        """Mute or hold speech input processing (e.g. while robot is speaking)."""
        ...

    def release(self) -> None:
        """Resume speech input processing."""
        ...


@runtime_checkable
class TTSProvider(Protocol):
    """Protocol for text-to-speech audio synthesis and playback (e.g. Google Cloud Chirp3)."""

    def speak(
        self,
        text: str,
        language: SupportedLanguage = "bn",
        blocking: bool = True,
    ) -> None:
        """Synthesize and play audio for the provided text.

        Args:
            text: Text content to synthesize.
            language: Target voice language code ('bn' or 'en').
            blocking: Whether to block until audio playback completes.
        """
        ...

    def interrupt(self) -> None:
        """Immediately abort active playback and discard queued audio buffers."""
        ...

    def is_playing(self) -> bool:
        """Return True if audio playback is currently in progress."""
        ...


class MockLLMProvider:
    """Explicit offline mock LLM provider for development and testing.

    Always formats responses with an explicit `[MOCK]` prefix so that callers
    and frontend interfaces can never mistake mock output for live AI generation.
    """

    def generate_response(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> str:
        _ = session_history
        prompt_clean = prompt.strip()
        prefix = "[MOCK]"
        if language == "en":
            if filtered_context:
                first_fact = filtered_context.splitlines()[0]
                return f"{prefix} Knowledge: {first_fact} | Response: {prompt_clean}"
            return f"{prefix} Response to: {prompt_clean}"
        else:
            if filtered_context:
                first_fact = filtered_context.splitlines()[0]
                return f"{prefix} তথ্য: {first_fact} | উত্তর: {prompt_clean}"
            return f"{prefix} উত্তর: {prompt_clean}"
