"""Unit tests for conversational dialogue, speech, and language model contracts."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest

from dh360_hi.contracts.dialogue import (
    FALLBACK_MESSAGES,
    VALID_LANGUAGES,
    VALID_ROLES,
    DialogueSessionState,
    DialogueTurn,
    LLMProvider,
    STTProvider,
    SupportedLanguage,
    TTSProvider,
    get_fallback_message,
)
from dh360_hi.contracts.errors import (
    SAFE_DIAGNOSTIC_KEYS,
    SAFE_PROVIDER_MESSAGES,
    ApplicationError,
    HardwareUnavailableError,
    LLMProviderError,
    ProviderError,
    ProviderNetworkError,
    ProviderTimeoutError,
    STTProviderError,
    TTSProviderError,
)

# ==============================================================================
# Mock Implementations for Protocol Conformance Verification
# ==============================================================================


class ConcreteLLMProvider:
    """Mock implementation adhering to the LLMProvider protocol."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None
        self.last_filtered_context: str | None = None
        self.last_language: SupportedLanguage | None = None
        self.last_history: list[DialogueTurn] | None = None

    def generate_response(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> str:
        self.last_prompt = prompt
        self.last_filtered_context = filtered_context
        self.last_language = language
        self.last_history = list(session_history) if session_history is not None else None

        if language == "bn":
            return f"বাংলা উত্তর: {prompt}"
        return f"English response: {prompt}"


class IncompleteLLMProvider:
    """Provider missing required methods to verify runtime protocol checks."""

    pass


class ConcreteSTTProvider:
    """Mock implementation adhering to the STTProvider protocol."""

    def __init__(self) -> None:
        self.is_held: bool = False
        self.last_language_hint: SupportedLanguage | None = None

    def listen_utterance(
        self,
        language_hint: SupportedLanguage = "bn",
        on_partial_text: Callable[[str], None] | None = None,
    ) -> str:
        self.last_language_hint = language_hint
        if on_partial_text is not None:
            on_partial_text("আংশিক")
        return "পরীক্ষামূলক বাক্য" if language_hint == "bn" else "test utterance"

    def hold(self) -> None:
        self.is_held = True

    def release(self) -> None:
        self.is_held = False


class IncompleteSTTProvider:
    """Provider missing hold/release methods."""

    def listen_utterance(
        self,
        language_hint: SupportedLanguage = "bn",
        on_partial_text: Callable[[str], None] | None = None,
    ) -> str:
        _ = (language_hint, on_partial_text)
        return "mock"


class ConcreteTTSProvider:
    """Mock implementation adhering to the TTSProvider protocol."""

    def __init__(self) -> None:
        self.playing: bool = False
        self.last_spoken_text: str | None = None
        self.last_language: SupportedLanguage | None = None

    def speak(
        self,
        text: str,
        language: SupportedLanguage = "bn",
        blocking: bool = True,
    ) -> None:
        self.last_spoken_text = text
        self.last_language = language
        self.playing = not blocking

    def interrupt(self) -> None:
        self.playing = False

    def is_playing(self) -> bool:
        return self.playing


class IncompleteTTSProvider:
    """Provider missing interrupt method."""

    def speak(
        self,
        text: str,
        language: SupportedLanguage = "bn",
        blocking: bool = True,
    ) -> None:
        _ = (text, language, blocking)

    def is_playing(self) -> bool:
        return False


# ==============================================================================
# DialogueTurn Model Tests
# ==============================================================================


def test_dialogue_turn_creation_and_defaults() -> None:
    """Verify DialogueTurn initializes with valid defaults and is immutable."""
    turn = DialogueTurn(role="user", text="হ্যালো রোবট")

    assert turn.role == "user"
    assert turn.text == "হ্যালো রোবট"
    assert turn.language == "bn"
    assert turn.metadata == {}
    assert isinstance(turn.timestamp, float)
    assert turn.timestamp <= time.time()

    # Immutable verification (dataclass frozen=True)
    with pytest.raises(FrozenInstanceError):
        turn.text = "modified"  # type: ignore[misc]


def test_dialogue_turn_to_dict_serialization() -> None:
    """Verify serialization to dictionary payload."""
    turn = DialogueTurn(
        role="model",
        text="Hello human",
        timestamp=1700000000.0,
        language="en",
        metadata={"tokens": 12},
    )
    data = turn.to_dict()

    assert data == {
        "role": "model",
        "text": "Hello human",
        "timestamp": 1700000000.0,
        "language": "en",
        "metadata": {"tokens": 12},
    }


def test_dialogue_turn_validates_role() -> None:
    """Verify that invalid roles raise ValueError."""
    for role in VALID_ROLES:
        turn = DialogueTurn(role=role, text="Valid text")
        assert turn.role == role

    with pytest.raises(ValueError, match="Invalid dialogue role 'operator'"):
        DialogueTurn(role="operator", text="Invalid role")  # type: ignore[arg-type]


def test_dialogue_turn_validates_language() -> None:
    """Verify that invalid language codes raise ValueError."""
    for lang in VALID_LANGUAGES:
        turn = DialogueTurn(role="user", text="Text", language=lang)
        assert turn.language == lang

    with pytest.raises(ValueError, match="Invalid language 'es'"):
        DialogueTurn(role="user", text="Hola", language="es")  # type: ignore[arg-type]


def test_dialogue_turn_validates_text_type() -> None:
    """Verify that non-string text raises TypeError."""
    with pytest.raises(TypeError, match="Dialogue text must be a string"):
        DialogueTurn(role="user", text=123)  # type: ignore[arg-type]


# ==============================================================================
# Protocol Conformance Tests
# ==============================================================================


def test_llm_provider_protocol_conformance() -> None:
    """Verify runtime checkability and conformance for LLMProvider."""
    llm = ConcreteLLMProvider()
    assert isinstance(llm, LLMProvider)

    incomplete = IncompleteLLMProvider()
    assert not isinstance(incomplete, LLMProvider)


def test_stt_provider_protocol_conformance() -> None:
    """Verify runtime checkability and conformance for STTProvider."""
    stt = ConcreteSTTProvider()
    assert isinstance(stt, STTProvider)

    incomplete = IncompleteSTTProvider()
    assert not isinstance(incomplete, STTProvider)


def test_tts_provider_protocol_conformance() -> None:
    """Verify runtime checkability and conformance for TTSProvider."""
    tts = ConcreteTTSProvider()
    assert isinstance(tts, TTSProvider)

    incomplete = IncompleteTTSProvider()
    assert not isinstance(incomplete, TTSProvider)


# ==============================================================================
# Isolated Session History & Flow Tests
# ==============================================================================


def test_llm_receives_filtered_context_and_language() -> None:
    """Verify LLMProvider receives strictly pre-filtered context and language targets."""
    llm = ConcreteLLMProvider()

    # Turn in Bangla
    response_bn = llm.generate_response(
        prompt="ডিন স্যারের রুম কোথায়?",
        filtered_context="Fact: ডিন স্যারের রুম ৪০২।",
        language="bn",
        session_history=[],
    )
    assert "বাংলা উত্তর:" in response_bn
    assert llm.last_language == "bn"
    assert llm.last_filtered_context == "Fact: ডিন স্যারের রুম ৪০২।"

    # Turn in English
    response_en = llm.generate_response(
        prompt="Where is the dean's room?",
        filtered_context="Fact: Dean room is 402.",
        language="en",
        session_history=[],
    )
    assert "English response:" in response_en
    assert llm.last_language == "en"
    assert llm.last_filtered_context == "Fact: Dean room is 402."


def test_session_history_isolation_between_subjects() -> None:
    """Verify dialogue turns remain isolated to their respective sessions and never bleed."""
    llm = ConcreteLLMProvider()

    session_store: dict[str, list[DialogueTurn]] = {
        "session-user-alice": [],
        "session-user-bob": [],
    }

    # Alice speaks in session A
    turn1_alice = DialogueTurn(role="user", text="What is my prescription?", language="en")
    session_store["session-user-alice"].append(turn1_alice)

    resp_alice = llm.generate_response(
        prompt=turn1_alice.text,
        filtered_context="Alice Prescription: Amoxicillin",
        language="en",
        session_history=session_store["session-user-alice"],
    )
    turn2_alice = DialogueTurn(role="model", text=resp_alice, language="en")
    session_store["session-user-alice"].append(turn2_alice)

    # Bob speaks in session B
    turn1_bob = DialogueTurn(role="user", text="What is my prescription?", language="en")
    session_store["session-user-bob"].append(turn1_bob)

    resp_bob = llm.generate_response(
        prompt=turn1_bob.text,
        filtered_context="Bob Prescription: Ibuprofen",
        language="en",
        session_history=session_store["session-user-bob"],
    )
    turn2_bob = DialogueTurn(role="model", text=resp_bob, language="en")
    session_store["session-user-bob"].append(turn2_bob)

    # Verify history isolation: Alice's history has only Alice's turns
    assert len(session_store["session-user-alice"]) == 2
    alice_texts = [t.text for t in session_store["session-user-alice"]]
    assert turn1_alice.text in alice_texts
    assert resp_alice in alice_texts

    # Verify Bob's history has only Bob's turns
    assert len(session_store["session-user-bob"]) == 2
    bob_texts = [t.text for t in session_store["session-user-bob"]]
    assert turn1_bob.text in bob_texts
    assert resp_bob in bob_texts

    # The last call to the LLM (for Bob) only received Bob's history
    assert llm.last_history == [turn1_bob]
    assert llm.last_filtered_context == "Bob Prescription: Ibuprofen"


def test_stt_language_hints_and_hold_state() -> None:
    """Verify STTProvider speech listening, partial callbacks, and hold/release states."""
    stt = ConcreteSTTProvider()

    # Initial state
    assert not stt.is_held

    # Test hold / release
    stt.hold()
    assert stt.is_held
    stt.release()
    assert not stt.is_held

    # Test listening with callback in Bangla
    partial_results: list[str] = []
    result_bn = stt.listen_utterance(
        language_hint="bn",
        on_partial_text=lambda t: partial_results.append(t),
    )
    assert result_bn == "পরীক্ষামূলক বাক্য"
    assert stt.last_language_hint == "bn"
    assert partial_results == ["আংশিক"]

    # Test listening in English
    result_en = stt.listen_utterance(language_hint="en")
    assert result_en == "test utterance"
    assert stt.last_language_hint == "en"


def test_tts_language_selection_and_interrupt() -> None:
    """Verify TTSProvider text synthesis, language passing, and interrupt handling."""
    tts = ConcreteTTSProvider()

    # Speak blocking in Bangla
    tts.speak("শুভ সকাল", language="bn", blocking=True)
    assert tts.last_spoken_text == "শুভ সকাল"
    assert tts.last_language == "bn"
    assert not tts.is_playing()

    # Speak non-blocking in English
    tts.speak("Good morning", language="en", blocking=False)
    assert tts.last_spoken_text == "Good morning"
    assert tts.last_language == "en"
    assert tts.is_playing()

    # Interrupt playback
    tts.interrupt()
    assert not tts.is_playing()


# ==============================================================================
# Provider Error Taxonomy Tests
# ==============================================================================


def test_provider_error_hierarchy() -> None:
    """Verify that all dialogue provider errors inherit from ProviderError and ApplicationError."""
    base_err = ProviderError("base failure", provider_name="test_provider")
    assert isinstance(base_err, ApplicationError)
    assert isinstance(base_err, Exception)

    subclasses = [
        LLMProviderError("llm failed"),
        STTProviderError("stt failed"),
        TTSProviderError("tts failed"),
        ProviderTimeoutError("timeout", provider_name="gemini", timeout_seconds=5.0),
        ProviderNetworkError("net err", provider_name="soniox"),
        HardwareUnavailableError("mic missing", device_type="mic", provider_name="soniox"),
    ]

    for err in subclasses:
        assert isinstance(err, ProviderError)
        assert isinstance(err, ApplicationError)


def test_provider_error_attributes_and_serialization() -> None:
    """Verify error codes, provider identifiers, and dictionary serializations."""
    # LLM Provider Error
    llm_err = LLMProviderError("Prompt rejected", provider_name="gemini", details={"tokens": 0})
    assert llm_err.code == "LLM_PROVIDER_ERROR"
    assert llm_err.message == "Prompt rejected"
    assert llm_err.provider_name == "gemini"
    assert llm_err.details["provider"] == "gemini"
    assert llm_err.details["tokens"] == 0

    llm_dict = llm_err.to_dict()
    assert llm_dict["code"] == "LLM_PROVIDER_ERROR"
    # Serialized message must be canonical safe message, never raw third-party prompt/error
    assert llm_dict["message"] == "Language model provider failed to generate response."
    assert llm_dict["details"]["provider"] == "gemini"
    assert llm_dict["details"]["tokens"] == 0

    # STT Provider Error
    stt_err = STTProviderError("Buffer overflow", provider_name="soniox")
    assert stt_err.code == "STT_PROVIDER_ERROR"
    assert stt_err.provider_name == "soniox"
    assert stt_err.to_dict()["message"] == "Speech-to-text recognition provider failed."

    # TTS Provider Error
    tts_err = TTSProviderError("Synthesis pipe closed", provider_name="gtts")
    assert tts_err.code == "TTS_PROVIDER_ERROR"
    assert tts_err.provider_name == "gtts"
    assert tts_err.to_dict()["message"] == "Text-to-speech synthesis provider failed."

    # Provider Timeout Error
    timeout_err = ProviderTimeoutError(
        "Request timed out",
        provider_name="gemini",
        timeout_seconds=10.5,
    )
    assert timeout_err.code == "PROVIDER_TIMEOUT"
    assert timeout_err.timeout_seconds == 10.5
    assert timeout_err.details["timeout_seconds"] == 10.5
    assert timeout_err.details["provider"] == "gemini"
    assert timeout_err.to_dict()["message"] == "Provider operation timed out."

    # Provider Network Error
    net_err = ProviderNetworkError("DNS lookup failed", provider_name="soniox")
    assert net_err.code == "PROVIDER_NETWORK_ERROR"
    assert net_err.details["provider"] == "soniox"
    assert net_err.to_dict()["message"] == "Provider network communication failed."

    # Hardware Unavailable Error
    hw_err = HardwareUnavailableError(
        "Microphone device /dev/snd not detected",
        device_type="microphone",
        provider_name="soniox",
    )
    assert hw_err.code == "HARDWARE_UNAVAILABLE"
    assert hw_err.device_type == "microphone"
    assert hw_err.details["device_type"] == "microphone"
    assert hw_err.details["provider"] == "soniox"
    assert hw_err.to_dict()["message"] == "Required audio hardware device is unavailable."


def test_provider_errors_never_leak_credentials() -> None:
    """Verify that provider error dictionaries avoid credential and secret leakage."""
    safe_details = {"status_code": 401, "reason": "unauthorized"}
    err = ProviderError("Auth failed", provider_name="gemini", details=safe_details)
    payload = err.to_dict()

    assert "api_key" not in payload["details"]
    assert "private_key" not in payload["details"]
    assert "token" not in payload["details"]


def test_provider_errors_scrub_injected_secrets_and_confidential_prompts() -> None:
    """Verify that serialized provider errors never expose raw third-party messages,
    fake API keys, bearer tokens, private keys, confidential prompts, or sensitive details.
    """
    fake_key = "AIzaSyFakeSecretKey9876543210abcdefg"
    fake_bearer = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummyPayload.signature"
    fake_private_key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0..."
    confidential_prompt = "What is patient Alice's confidential medical prescription?"
    sensitive_transcript = "My secret passcode is 9876"

    # Inject into raw error message and details
    raw_msg = (
        f"Gemini API error 403: {fake_bearer} using key {fake_key} on prompt: {confidential_prompt}"
    )
    err = LLMProviderError(
        message=raw_msg,
        provider_name="gemini",
        details={
            "api_key": fake_key,
            "authorization": fake_bearer,
            "private_key": fake_private_key,
            "prompt": confidential_prompt,
            "transcript": sensitive_transcript,
            "device_type": fake_private_key,  # Injected into an allowlisted field name
            "status_code": 403,
            "is_retryable": False,
            "tokens": 120,
        },
    )

    serialized = err.to_dict()

    # 1. Message must be safe canonical text
    assert serialized["message"] == "Language model provider failed to generate response."
    assert fake_key not in serialized["message"]
    assert fake_bearer not in serialized["message"]
    assert confidential_prompt not in serialized["message"]

    # 2. Details allowlist enforcement
    details = serialized["details"]
    assert "api_key" not in details
    assert "authorization" not in details
    assert "private_key" not in details
    assert "prompt" not in details
    assert "transcript" not in details

    # 3. String value safety checking (injected private key into device_type was stripped)
    assert "device_type" not in details

    # 4. Safe diagnostic scalars were preserved
    assert details["provider"] == "gemini"
    assert details["status_code"] == 403
    assert details["is_retryable"] is False
    assert details["tokens"] == 120

    # 5. Full string serialization inspection
    serialized_str = str(serialized)
    assert fake_key not in serialized_str
    assert fake_bearer not in serialized_str
    assert "BEGIN RSA PRIVATE KEY" not in serialized_str
    assert confidential_prompt not in serialized_str
    assert sensitive_transcript not in serialized_str


def test_provider_errors_allowlist_diagnostic_fields() -> None:
    """Verify that SAFE_DIAGNOSTIC_KEYS allows non-sensitive fields and drops unknown ones."""
    assert "status_code" in SAFE_DIAGNOSTIC_KEYS
    assert "LLM_PROVIDER_ERROR" in SAFE_PROVIDER_MESSAGES
    err = ProviderError(
        message="Hardware sound failure",
        provider_name="soniox",
        details={
            "device_type": "microphone",
            "timeout_seconds": 15.0,
            "audio_sample_rate": 16000,
            "audio_channels": 1,
            "arbitrary_untrusted_field": "some_data",
        },
    )
    payload = err.to_dict()["details"]

    assert payload["provider"] == "soniox"
    assert payload["device_type"] == "microphone"
    assert payload["timeout_seconds"] == 15.0
    assert payload["audio_sample_rate"] == 16000
    assert payload["audio_channels"] == 1
    assert "arbitrary_untrusted_field" not in payload


# ==============================================================================
# DialogueSessionState & Grace Period Tests
# ==============================================================================


def test_dialogue_session_state_initialization() -> None:
    """Verify initial state of DialogueSessionState."""
    session = DialogueSessionState(session_id="sess-001")

    assert session.session_id == "sess-001"
    assert session.person_id is None
    assert session.display_name is None
    assert not session.is_authorized
    assert session.turns == []
    assert session.get_turns() == []
    assert session.identity_lost_at is None
    assert not session.is_in_grace_period
    assert not session.is_grace_expired


def test_dialogue_session_state_identity_loss_immediate_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify immediate downgrade to unprivileged mode upon identity loss."""
    session = DialogueSessionState(
        session_id="sess-002",
        person_id="p-001",
        display_name="Alice",
        is_authorized=True,
    )
    session.add_turn(DialogueTurn(role="user", text="Confidential query"))

    # Initial verified state
    assert session.is_authorized

    # Trigger identity loss at simulated monotonic time t0
    t0 = 1000.0
    session.on_identity_lost(now=t0)

    # CRITICAL INVARIANT: is_authorized MUST immediately drop to False
    assert not session.is_authorized
    assert session.identity_lost_at == t0

    # In grace period at t0 + 30 seconds
    monkeypatch.setattr(time, "monotonic", lambda: t0 + 30.0)
    assert session.is_in_grace_period
    assert not session.is_grace_expired

    # While in grace period, is_authorized remains False to prevent privileged leakage
    assert not session.is_authorized


def test_dialogue_session_state_reauthorization_same_person() -> None:
    """Verify history restoration when the same person is reauthorized within grace period."""
    session = DialogueSessionState(
        session_id="sess-003",
        person_id="p-001",
        display_name="Alice",
        is_authorized=True,
    )
    turn = DialogueTurn(role="user", text="What is my appointment?")
    session.add_turn(turn)

    # Identity lost at t0
    t0 = 1000.0
    session.on_identity_lost(now=t0)
    assert not session.is_authorized

    # Re-identify Alice at t0 + 20s (well within 60s grace)
    reauthorized = session.on_reauthorized(
        person_id="p-001",
        display_name="Alice Updated",
        now=t0 + 20.0,
    )

    assert reauthorized is True
    assert session.is_authorized is True
    assert session.identity_lost_at is None
    assert session.display_name == "Alice Updated"
    # History preserved
    assert len(session.turns) == 1
    assert session.turns[0].text == "What is my appointment?"


def test_dialogue_session_state_reauthorization_different_person() -> None:
    """Verify history is immediately purged if a different person arrives."""
    session = DialogueSessionState(
        session_id="sess-004",
        person_id="p-001",
        display_name="Alice",
        is_authorized=True,
    )
    session.add_turn(DialogueTurn(role="user", text="Alice's secret question"))

    t0 = 1000.0
    session.on_identity_lost(now=t0)

    # Different person Bob identified at t0 + 10s
    reauthorized = session.on_reauthorized(
        person_id="p-002",
        display_name="Bob",
        now=t0 + 10.0,
    )

    assert reauthorized is False
    # History must be purged cleanly
    assert len(session.turns) == 0
    # New state is established for Bob
    assert session.person_id == "p-002"
    assert session.display_name == "Bob"
    assert session.is_authorized is True
    assert session.identity_lost_at is None


def test_dialogue_session_state_grace_period_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that grace period expiry purges history on subsequent re-identification."""
    session = DialogueSessionState(
        session_id="sess-005",
        person_id="p-001",
        display_name="Alice",
        is_authorized=True,
    )
    session.add_turn(DialogueTurn(role="user", text="Hello"))

    t0 = 1000.0
    session.on_identity_lost(now=t0)

    # Fast forward past 60.0s grace period (t0 + 65.0s)
    monkeypatch.setattr(time, "monotonic", lambda: t0 + 65.0)
    assert not session.is_in_grace_period
    assert session.is_grace_expired

    # Attempt reauthorization of same person after expiry
    reauthorized = session.on_reauthorized(
        person_id="p-001",
        display_name="Alice",
        now=t0 + 65.0,
    )

    assert reauthorized is False
    # History must be purged
    assert len(session.turns) == 0
    assert session.is_authorized is True


def test_dialogue_session_state_reset() -> None:
    """Verify manual reset clears turns and resets authorization status."""
    session = DialogueSessionState(
        session_id="sess-006",
        person_id="p-001",
        display_name="Alice",
        is_authorized=True,
    )
    session.add_turn(DialogueTurn(role="user", text="Hello"))
    session.identity_lost_at = 1000.0

    session.reset()

    assert session.person_id is None
    assert session.display_name is None
    assert not session.is_authorized
    assert len(session.turns) == 0
    assert session.identity_lost_at is None


def test_dialogue_session_state_history_access_and_mutation_after_expiry() -> None:
    """Verify that accessing (get_turns) or mutating (add_turn) history after grace expiry
    automatically purges expired turns without requiring on_reauthorized.
    """
    session = DialogueSessionState(
        session_id="sess-expiry-check",
        person_id="p-alice",
        display_name="Alice",
        is_authorized=True,
    )
    session.add_turn(DialogueTurn(role="user", text="Confidential appointment"))
    assert len(session.get_turns()) == 1

    # Identity lost at monotonic t0 = 2000.0
    t0 = 2000.0
    session.on_identity_lost(now=t0)
    assert not session.is_authorized

    # 1. Access before expiry (t0 + 30.0s): turns still available in grace
    active_turns = session.get_turns(now=t0 + 30.0)
    assert len(active_turns) == 1
    assert active_turns[0].text == "Confidential appointment"

    # 2. Add turn before expiry (t0 + 40.0s): successfully added
    session.add_turn(DialogueTurn(role="user", text="Another question in grace"), now=t0 + 40.0)
    assert len(session.get_turns(now=t0 + 40.0)) == 2

    # 3. Access after expiry (t0 + 60.1s): MUST purge turns immediately
    expired_turns = session.get_turns(now=t0 + 60.1)
    assert expired_turns == []
    # Subsequent access is also empty
    assert session.turns == []
    assert session.identity_lost_at is None

    # 4. Mutation after expiry (t0 + 70.0s): adding a turn starts a fresh history
    new_turn = DialogueTurn(role="user", text="New unprivileged request")
    session.add_turn(new_turn, now=t0 + 70.0)
    current_turns = session.get_turns(now=t0 + 70.0)
    assert len(current_turns) == 1
    assert current_turns[0].text == "New unprivileged request"
    # Old confidential turns never leaked
    assert "Confidential appointment" not in [t.text for t in current_turns]
    # Retained history does not confer authorization
    assert not session.is_authorized


def test_dialogue_session_state_encapsulation_defensive_copy() -> None:
    """Verify that callers cannot tamper with internal dialogue history via returned lists."""
    session = DialogueSessionState(session_id="sess-tamper")
    original_turn = DialogueTurn(role="user", text="Original")
    session.add_turn(original_turn)

    # 1. Modifying returned list from get_turns() must not affect internal state
    turns_copy = session.get_turns()
    turns_copy.clear()
    assert len(session.get_turns()) == 1

    # 2. Modifying returned list from .turns property must not affect internal state
    property_turns = session.turns
    property_turns.append(DialogueTurn(role="user", text="Injected turn"))
    assert len(session.get_turns()) == 1
    assert session.get_turns()[0].text == "Original"


def test_dialogue_session_state_monotonic_clock_immune_to_wall_clock_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that grace period calculations rely on monotonic clock and ignore wall-clock steps."""
    session = DialogueSessionState(session_id="sess-monotonic")
    session.add_turn(DialogueTurn(role="user", text="Turn 1"))

    # Fix monotonic clock at 5000.0
    mono_time = 5000.0
    monkeypatch.setattr(time, "monotonic", lambda: mono_time)

    # Simulate wall-clock stepping forward or backward by hours (e.g. NTP sync)
    wall_time = 1700000000.0
    monkeypatch.setattr(time, "time", lambda: wall_time)

    session.on_identity_lost()
    assert session.identity_lost_at == 5000.0

    # Step wall-clock back 1 hour (NTP adjustment) - monotonic time remains at +10s
    wall_time -= 3600.0
    mono_time += 10.0
    assert session.is_in_grace_period
    assert not session.is_grace_expired
    assert len(session.get_turns()) == 1

    # Step wall-clock forward 10 hours - monotonic time advances past grace period (+65s)
    wall_time += 36000.0
    mono_time = 5065.0
    assert not session.is_in_grace_period
    assert session.is_grace_expired
    # Access enforces expiry based on monotonic clock
    assert session.get_turns() == []


# ==============================================================================
# Localized Fallback Text Mapping Tests
# ==============================================================================


def test_fallback_messages_structure_and_completeness() -> None:
    """Verify that all standard fallback keys exist and contain both Bangla and English text."""
    expected_keys = {"stt_error", "llm_error", "tts_error", "network_error", "unauthorized"}

    for key in expected_keys:
        assert key in FALLBACK_MESSAGES
        assert "bn" in FALLBACK_MESSAGES[key]
        assert "en" in FALLBACK_MESSAGES[key]
        assert len(FALLBACK_MESSAGES[key]["bn"]) > 0
        assert len(FALLBACK_MESSAGES[key]["en"]) > 0


def test_get_fallback_message_routing() -> None:
    """Verify get_fallback_message returns appropriate localized message."""
    # STT Error
    bn_stt = get_fallback_message("stt_error", language="bn")
    en_stt = get_fallback_message("stt_error", language="en")
    assert "শুনতে পাইনি" in bn_stt
    assert "hear you clearly" in en_stt

    # LLM Error
    bn_llm = get_fallback_message("llm_error", language="bn")
    en_llm = get_fallback_message("llm_error", language="en")
    assert "তথ্য প্রক্রিয়াকরণ" in bn_llm
    assert "unable to process" in en_llm

    # TTS Error (preserves text fallback hint)
    bn_tts = get_fallback_message("tts_error", language="bn")
    en_tts = get_fallback_message("tts_error", language="en")
    assert "স্ক্রিনে উত্তরটি দেখানো হচ্ছে" in bn_tts
    assert "response is displayed on screen" in en_tts

    # Network Error
    bn_net = get_fallback_message("network_error", language="bn")
    en_net = get_fallback_message("network_error", language="en")
    assert "নেটওয়ার্ক সংযোগে" in bn_net
    assert "network connection issue" in en_net

    # Unauthorized Error
    bn_unauth = get_fallback_message("unauthorized", language="bn")
    en_unauth = get_fallback_message("unauthorized", language="en")
    assert "সুরক্ষিত" in bn_unauth
    assert "restricted" in en_unauth


def test_get_fallback_message_defaults() -> None:
    """Verify fallback behavior for unknown error keys and default language."""
    # Default language is 'bn'
    default_msg = get_fallback_message("llm_error")
    assert default_msg == FALLBACK_MESSAGES["llm_error"]["bn"]

    # Unknown key defaults to llm_error
    unknown_msg = get_fallback_message("unknown_error_key", language="en")
    assert unknown_msg == FALLBACK_MESSAGES["llm_error"]["en"]
