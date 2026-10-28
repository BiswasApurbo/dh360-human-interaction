"""Unit tests for Eliva provider adapters (Gemini, Soniox, Google Cloud TTS)."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.contracts.dialogue import (
    DialogueTurn,
    LLMProvider,
    STTProvider,
    SupportedLanguage,
    TTSProvider,
)
from dh360_hi.contracts.errors import (
    HardwareUnavailableError,
    LLMProviderError,
    ProviderNetworkError,
    ProviderTimeoutError,
    STTProviderError,
    TTSProviderError,
)
from dh360_hi.contracts.events import SessionStartedEvent
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.integrations.eliva.adapters import (
    ElivaGeminiAdapter,
    ElivaGoogleTTSAdapter,
    ElivaSonioxAdapter,
)

# ==============================================================================
# 1. Contract Conformance & Safe Instantiation Tests
# ==============================================================================


def test_adapters_conform_to_protocols() -> None:
    """Verify all adapters satisfy runtime Protocol checks without third-party SDKs."""
    gemini_adapter = ElivaGeminiAdapter()
    soniox_adapter = ElivaSonioxAdapter()
    gtts_adapter = ElivaGoogleTTSAdapter()

    assert isinstance(gemini_adapter, LLMProvider)
    assert isinstance(soniox_adapter, STTProvider)
    assert isinstance(gtts_adapter, TTSProvider)


def test_no_hardware_or_network_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that constructing adapters performs no audio device or network calls."""
    # Ensure environment variables are clear
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)

    gemini = ElivaGeminiAdapter()
    soniox = ElivaSonioxAdapter()
    gtts = ElivaGoogleTTSAdapter()

    assert gemini.model_name == "gemini-3.5-flash-lite"
    assert not soniox.is_held
    assert not gtts.is_playing()


# ==============================================================================
# 2. ElivaGeminiAdapter Tests
# ==============================================================================


def test_gemini_missing_api_key_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing API key produces sanitized LLMProviderError without crash."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    adapter = ElivaGeminiAdapter(api_key=None)

    with pytest.raises(LLMProviderError) as exc_info:
        adapter.generate_response(
            prompt="Hello",
            filtered_context="",
            language="en",
        )

    err = exc_info.value
    assert err.code == "LLM_PROVIDER_ERROR"
    assert err.provider_name == "gemini"
    assert err.details.get("reason") == "MISSING_API_KEY"


def test_gemini_missing_sdk_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing google-genai package raises sanitized LLMProviderError."""
    import builtins

    orig_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        fromlist = kwargs.get("fromlist") or (args[2] if len(args) > 2 else None)
        if "genai" in name or (fromlist and "genai" in fromlist):
            raise ImportError("No module named 'google.genai'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key")
    adapter = ElivaGeminiAdapter(api_key="test-dummy-key")

    with pytest.raises(LLMProviderError) as exc_info:
        adapter.generate_response(
            prompt="Hello",
            filtered_context="",
            language="bn",
        )

    err = exc_info.value
    assert err.code == "LLM_PROVIDER_ERROR"
    assert err.details.get("reason") == "SDK_NOT_INSTALLED"


def test_gemini_injected_generator_bangla_and_english() -> None:
    """Verify injected generator function handles English and Bangla prompts."""
    recorded_calls: list[dict[str, Any]] = []

    def mock_generator(
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> str:
        recorded_calls.append(
            {
                "prompt": prompt,
                "context": filtered_context,
                "language": language,
                "history": list(session_history) if session_history else [],
            }
        )
        if language == "bn":
            return "আমি রোবোটেক ভ্যালির রোবট।"
        return "I am the Robotech Valley robot."

    adapter = ElivaGeminiAdapter(generator_fn=mock_generator)

    # 1. Bangla request
    reply_bn = adapter.generate_response(
        prompt="তোমার নাম কি?",
        filtered_context="Fact: Eliva is a humanoid robot.",
        language="bn",
    )
    assert reply_bn == "আমি রোবোটেক ভ্যালির রোবট।"
    assert recorded_calls[0]["language"] == "bn"
    assert recorded_calls[0]["prompt"] == "তোমার নাম কি?"

    # 2. English request
    reply_en = adapter.generate_response(
        prompt="What is your name?",
        filtered_context="Fact: Eliva is a humanoid robot.",
        language="en",
    )
    assert reply_en == "I am the Robotech Valley robot."
    assert recorded_calls[1]["language"] == "en"
    assert recorded_calls[1]["prompt"] == "What is your name?"


def test_gemini_exception_mapping_and_sanitization() -> None:
    """Verify exceptions from underlying generator are mapped to typed errors without leaks."""

    # 1. TimeoutError mapping
    def timeout_gen(**_kwargs: Any) -> str:
        raise TimeoutError("Request deadline exceeded")

    adapter_timeout = ElivaGeminiAdapter(generator_fn=timeout_gen)
    with pytest.raises(ProviderTimeoutError) as exc_info:
        adapter_timeout.generate_response("query", "", "en")
    assert exc_info.value.code == "PROVIDER_TIMEOUT"
    assert exc_info.value.provider_name == "gemini"

    # 2. ConnectionError mapping
    def net_gen(**_kwargs: Any) -> str:
        raise ConnectionError("DNS resolution failed")

    adapter_net = ElivaGeminiAdapter(generator_fn=net_gen)
    with pytest.raises(ProviderNetworkError) as exc_info:
        adapter_net.generate_response("query", "", "bn")
    assert exc_info.value.code == "PROVIDER_NETWORK_ERROR"

    # 3. Secret redaction check
    def secret_leaking_gen(**_kwargs: Any) -> str:
        raise RuntimeError("Authorization failure with secret_api_key_12345")

    adapter_secret = ElivaGeminiAdapter(generator_fn=secret_leaking_gen)
    with pytest.raises(LLMProviderError) as exc_info:
        adapter_secret.generate_response("query", "", "bn")
    err = exc_info.value
    serialized = err.to_dict()
    # The raw secret string must never appear in serialized error dictionary
    assert "secret_api_key_12345" not in serialized["message"]
    assert "secret_api_key_12345" not in str(serialized["details"])


def test_gemini_no_shared_state_between_sessions() -> None:
    """Verify dialogue history passed to generator is scoped strictly per invocation."""
    passed_histories: list[list[DialogueTurn]] = []

    def tracking_gen(
        session_history: list[DialogueTurn] | None = None,
        **_kwargs: Any,
    ) -> str:
        passed_histories.append(list(session_history) if session_history else [])
        return "Acknowledged"

    adapter = ElivaGeminiAdapter(generator_fn=tracking_gen)

    history_session_1 = [DialogueTurn(role="user", text="Session 1 query", language="en")]
    history_session_2 = [DialogueTurn(role="user", text="Session 2 query", language="bn")]

    adapter.generate_response("Query 1", "", "en", session_history=history_session_1)
    adapter.generate_response("Query 2", "", "bn", session_history=history_session_2)

    assert len(passed_histories) == 2
    assert len(passed_histories[0]) == 1
    assert passed_histories[0][0].text == "Session 1 query"
    assert len(passed_histories[1]) == 1
    assert passed_histories[1][0].text == "Session 2 query"


# ==============================================================================
# 3. ElivaSonioxAdapter Tests
# ==============================================================================


def test_soniox_missing_api_key_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing API key produces sanitized STTProviderError."""
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    adapter = ElivaSonioxAdapter(api_key=None)

    with pytest.raises(STTProviderError) as exc_info:
        adapter.listen_utterance(language_hint="bn")

    err = exc_info.value
    assert err.code == "STT_PROVIDER_ERROR"
    assert err.provider_name == "soniox"
    assert err.details.get("reason") == "MISSING_API_KEY"


def test_soniox_missing_sdk_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing audio/STT dependencies raise STTProviderError."""
    monkeypatch.setenv("SONIOX_API_KEY", "test-soniox-key")
    adapter = ElivaSonioxAdapter(api_key="test-soniox-key")

    with pytest.raises(STTProviderError) as exc_info:
        adapter.listen_utterance(language_hint="en")

    err = exc_info.value
    assert err.code == "STT_PROVIDER_ERROR"
    assert err.details.get("reason") == "SDK_NOT_INSTALLED"


def test_soniox_hold_and_release() -> None:
    """Verify hold() mutes input and release() restores input processing."""
    adapter = ElivaSonioxAdapter(listen_fn=lambda **_kwargs: "Transcribed audio")

    assert not adapter.is_held
    assert adapter.listen_utterance() == "Transcribed audio"

    adapter.hold()
    assert adapter.is_held
    # Held adapter returns empty transcription immediately without invoking listener
    assert adapter.listen_utterance() == ""

    adapter.release()
    assert not adapter.is_held
    assert adapter.listen_utterance() == "Transcribed audio"


def test_soniox_injected_listener_bangla_and_english() -> None:
    """Verify injected listener passes language hint and partial callbacks."""
    partials: list[str] = []

    def mock_listener(
        language_hint: SupportedLanguage = "bn",
        on_partial_text: Any | None = None,
    ) -> str:
        if on_partial_text:
            on_partial_text("partial text")
        if language_hint == "bn":
            return "বাংলা ট্রান্সক্রিপশন"
        return "English transcription"

    adapter = ElivaSonioxAdapter(listen_fn=mock_listener)

    res_bn = adapter.listen_utterance(
        language_hint="bn",
        on_partial_text=lambda t: partials.append(t),
    )
    assert res_bn == "বাংলা ট্রান্সক্রিপশন"
    assert partials == ["partial text"]

    res_en = adapter.listen_utterance(language_hint="en")
    assert res_en == "English transcription"


def test_soniox_exception_mapping() -> None:
    """Verify errors from underlying listener are mapped to typed errors."""

    def net_listener(**_kwargs: Any) -> str:
        raise ConnectionError("WebSocket dropped")

    adapter_net = ElivaSonioxAdapter(listen_fn=net_listener)
    with pytest.raises(ProviderNetworkError) as exc_info:
        adapter_net.listen_utterance()
    assert exc_info.value.code == "PROVIDER_NETWORK_ERROR"

    def timeout_listener(**_kwargs: Any) -> str:
        raise TimeoutError("Listen deadline exceeded")

    adapter_timeout = ElivaSonioxAdapter(listen_fn=timeout_listener)
    with pytest.raises(ProviderTimeoutError) as exc_info:
        adapter_timeout.listen_utterance()
    assert exc_info.value.code == "PROVIDER_TIMEOUT"


# ==============================================================================
# 4. ElivaGoogleTTSAdapter Tests
# ==============================================================================


def test_tts_missing_credentials_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing credentials or SDK raises sanitized TTSProviderError."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    adapter = ElivaGoogleTTSAdapter()

    with pytest.raises(TTSProviderError) as exc_info:
        adapter.speak("Test utterance", language="bn")

    err = exc_info.value
    assert err.code == "TTS_PROVIDER_ERROR"
    assert err.provider_name == "google_tts"
    assert err.details.get("reason") in ("SDK_NOT_INSTALLED", "MISSING_CREDENTIALS")


def test_tts_voice_resolution() -> None:
    """Verify language code and voice name mapping for English and Bangla."""
    adapter = ElivaGoogleTTSAdapter(
        voice_bn="bn-BD-Standard-A",
        voice_en="en-US-Chirp3-HD-Charon",
    )

    voice_bn, code_bn = adapter._resolve_voice("bn")
    assert voice_bn == "bn-BD-Standard-A"
    assert code_bn == "bn-BD"

    voice_en, code_en = adapter._resolve_voice("en")
    assert voice_en == "en-US-Chirp3-HD-Charon"
    assert code_en == "en-US"


def test_tts_injected_speaker_and_interrupt() -> None:
    """Verify injected speaker execution and interrupt state tracking."""
    spoken: list[dict[str, Any]] = []

    def mock_speaker(text: str, language: SupportedLanguage, blocking: bool) -> None:
        spoken.append({"text": text, "language": language, "blocking": blocking})

    adapter = ElivaGoogleTTSAdapter(speak_fn=mock_speaker)

    adapter.speak("Hello from robot", language="en", blocking=True)
    assert len(spoken) == 1
    assert spoken[0]["text"] == "Hello from robot"
    assert spoken[0]["language"] == "en"
    assert spoken[0]["blocking"] is True

    # Interrupt
    adapter.interrupt()
    assert not adapter.is_playing()


def test_tts_interruption_stops_playback_and_discards_subsequent_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify interruption halts sounddevice, triggers callback, and discards speech."""
    interrupt_called = threading.Event()
    sd_stop_called = threading.Event()

    class FakeSoundDevice:
        @staticmethod
        def stop() -> None:
            sd_stop_called.set()

    # Monkeypatch sounddevice in sys.modules to simulate presence without physical hardware
    import sys

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice)

    adapter = ElivaGoogleTTSAdapter(
        interrupt_fn=lambda: interrupt_called.set(),
    )

    adapter.interrupt()
    assert interrupt_called.is_set()
    assert sd_stop_called.is_set()
    assert not adapter.is_playing()


def test_tts_blocking_and_nonblocking_execution() -> None:
    """Verify blocking=True waits for completion while blocking=False returns promptly."""
    playback_started = threading.Event()
    playback_proceed = threading.Event()
    recorded_blocking: list[bool] = []

    def mock_speaker(text: str, language: SupportedLanguage, blocking: bool) -> None:
        _ = (text, language)
        recorded_blocking.append(blocking)
        playback_started.set()
        if blocking:
            playback_proceed.wait(timeout=2.0)

    adapter = ElivaGoogleTTSAdapter(speak_fn=mock_speaker)

    # 1. Non-blocking call: must return immediately without waiting for playback_proceed
    adapter.speak("Nonblocking speech", language="en", blocking=False)
    assert playback_started.is_set()
    assert recorded_blocking == [False]

    # 2. Blocking call: must block until playback_proceed is signaled
    playback_started.clear()
    thread_done = threading.Event()

    def run_blocking() -> None:
        adapter.speak("Blocking speech", language="en", blocking=True)
        thread_done.set()

    t = threading.Thread(target=run_blocking, daemon=True)
    t.start()

    assert playback_started.wait(timeout=1.0)
    # Thread should be waiting on playback_proceed
    assert not thread_done.is_set()

    # Release blocker
    playback_proceed.set()
    assert thread_done.wait(timeout=1.0)
    assert recorded_blocking == [False, True]
    assert not adapter.is_playing()


def test_tts_error_and_interruption_leave_is_playing_consistent() -> None:
    """Verify is_playing() remains consistent on errors or mid-speech interruption."""

    def failing_speaker(**_kwargs: Any) -> None:
        raise RuntimeError("Playback device fault")

    adapter_err = ElivaGoogleTTSAdapter(speak_fn=failing_speaker)
    with pytest.raises(HardwareUnavailableError):
        adapter_err.speak("Faulty playback", language="bn", blocking=True)

    assert not adapter_err.is_playing()

    # Interruption mid-speech
    proceed_gate = threading.Event()
    inside_speak = threading.Event()

    def interruptible_speaker(**_kwargs: Any) -> None:
        inside_speak.set()
        proceed_gate.wait(timeout=2.0)

    adapter_interrupted = ElivaGoogleTTSAdapter(speak_fn=interruptible_speaker)

    def speak_worker() -> None:
        adapter_interrupted.speak("Interrupted playback", language="en", blocking=True)

    t = threading.Thread(target=speak_worker, daemon=True)
    t.start()

    assert inside_speak.wait(timeout=1.0)
    assert adapter_interrupted.is_playing()

    adapter_interrupted.interrupt()
    assert not adapter_interrupted.is_playing()

    proceed_gate.set()
    t.join(timeout=1.0)
    assert not adapter_interrupted.is_playing()


def test_soniox_hold_release_lifecycle_with_injected_stream() -> None:
    """Verify hold/release controls stream muting and lifecycle callbacks without leaking audio."""
    stream_muted = threading.Event()

    adapter = ElivaSonioxAdapter(
        hold_fn=lambda: stream_muted.set(),
        release_fn=lambda: stream_muted.clear(),
        listen_fn=lambda **_kwargs: "Active transcript" if not stream_muted.is_set() else "",
    )

    assert not adapter.is_held

    # 1. Normal listening
    assert adapter.listen_utterance() == "Active transcript"

    # 2. Hold speech stream
    adapter.hold()
    assert adapter.is_held
    assert stream_muted.is_set()
    # While held, listen_utterance returns empty utterance
    assert adapter.listen_utterance() == ""

    # 3. Release speech stream
    adapter.release()
    assert not adapter.is_held
    assert not stream_muted.is_set()
    assert adapter.listen_utterance() == "Active transcript"


def test_tts_exception_mapping() -> None:
    """Verify errors from underlying speaker are mapped to typed errors."""

    def net_speaker(**_kwargs: Any) -> None:
        raise ConnectionError("Google Cloud endpoint unreachable")

    adapter_net = ElivaGoogleTTSAdapter(speak_fn=net_speaker)
    with pytest.raises(ProviderNetworkError) as exc_info:
        adapter_net.speak("Hello", language="en")
    assert exc_info.value.code == "PROVIDER_NETWORK_ERROR"

    def hw_speaker(**_kwargs: Any) -> None:
        raise RuntimeError("PipeWire ALSA device not found")

    adapter_hw = ElivaGoogleTTSAdapter(speak_fn=hw_speaker)
    with pytest.raises(HardwareUnavailableError) as exc_info:
        adapter_hw.speak("Hello", language="bn")
    assert exc_info.value.code == "HARDWARE_UNAVAILABLE"
    assert exc_info.value.device_type == "speaker"


# ==============================================================================
# 5. Full DialogueCoordinator Round-Trip Integration
# ==============================================================================


def test_adapters_integrate_with_dialogue_coordinator(tmp_path: Any) -> None:
    """Verify Eliva adapters integrate seamlessly with DialogueCoordinator."""
    event_bus = EventBus()
    csv_file = tmp_path / "knowledge.csv"
    csv_file.write_text(
        "Question (English),Answer (English),প্রশ্ন (বাংলা),উত্তর (বাংলা)\n"
        "What is Eliva?,Eliva is a humanoid robot.,এলিভা কি?,এলিভা একটি মানবাকৃতির রোবট।\n",
        encoding="utf-8",
    )
    knowledge_service = KnowledgeService(csv_path=csv_file)

    spoken_responses: list[str] = []

    gemini_adapter = ElivaGeminiAdapter(
        generator_fn=lambda prompt, **_kwargs: f"Answer to {prompt}",
    )
    soniox_adapter = ElivaSonioxAdapter(
        listen_fn=lambda **_kwargs: "What is Eliva?",
    )
    tts_adapter = ElivaGoogleTTSAdapter(
        speak_fn=lambda text, **_kwargs: spoken_responses.append(text),
    )

    coordinator = DialogueCoordinator(
        event_bus=event_bus,
        knowledge_service=knowledge_service,
        llm_provider=gemini_adapter,
        stt_provider=soniox_adapter,
        tts_provider=tts_adapter,
        default_language="en",
    )

    # Start session
    event_bus.publish(
        SessionStartedEvent(
            session_id="session-eliva-1",
            person_id=None,
        )
    )

    # 1. Successful turn using all three adapters
    turn = coordinator.handle_turn(language="en")
    assert turn.role == "model"
    assert turn.text == "Answer to What is Eliva?"
    assert spoken_responses == ["Answer to What is Eliva?"]

    # 2. TTS error does not crash coordinator or lose text response
    def failing_speaker(**_kwargs: Any) -> None:
        raise TTSProviderError("Speaker buffer overflow", provider_name="google_tts")

    failing_tts = ElivaGoogleTTSAdapter(speak_fn=failing_speaker)
    coordinator_with_failing_tts = DialogueCoordinator(
        event_bus=event_bus,
        knowledge_service=knowledge_service,
        llm_provider=gemini_adapter,
        stt_provider=soniox_adapter,
        tts_provider=failing_tts,
        default_language="en",
    )

    turn2 = coordinator_with_failing_tts.handle_turn(user_text="Another question", language="en")
    assert turn2.text == "Answer to Another question"
    assert turn2.metadata.get("tts_error") is True

    # 3. LLM error produces localized fallback without exposing internal error
    def failing_gen(**_kwargs: Any) -> str:
        raise LLMProviderError("Quota exceeded", provider_name="gemini")

    failing_llm = ElivaGeminiAdapter(generator_fn=failing_gen)
    coordinator_with_failing_llm = DialogueCoordinator(
        event_bus=event_bus,
        knowledge_service=knowledge_service,
        llm_provider=failing_llm,
        stt_provider=soniox_adapter,
        tts_provider=tts_adapter,
        default_language="en",
    )

    turn3 = coordinator_with_failing_llm.handle_turn(user_text="Will fail", language="en")
    assert "unable to process your request" in turn3.text
    assert turn3.metadata.get("error") == "LLM_PROVIDER_ERROR"
