"""Unit tests for local wake-word detection ("Hey KB"), shared audio capture pipeline,
and 8-state conversation lifecycle with 60-second inactivity timeout.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.contracts.dialogue import MockLLMProvider
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.integrations.audio.shared_capture import SharedMicrophoneCapture
from dh360_hi.integrations.audio.wake_detector import (
    WakeWordDetector,
    WakeWordEvent,
)


class FakeSTTProvider:
    """Mock STT provider for testing coordinator voice turns."""

    def __init__(
        self,
        transcription: str = "What are the visiting hours?",
        transcriptions: list[str] | None = None,
    ) -> None:
        self.transcription = transcription
        self.transcriptions = list(transcriptions) if transcriptions is not None else None
        self.listen_calls = 0
        self.is_held = False

    def hold(self) -> None:
        self.is_held = True

    def release(self) -> None:
        self.is_held = False

    def listen_utterance(self, language_hint: str = "bn", on_partial_text: Any = None) -> str:
        self.listen_calls += 1
        if self.transcriptions is not None:
            if self.transcriptions:
                return self.transcriptions.pop(0)
            return ""
        return self.transcription


class FakeTTSProvider:
    """Mock TTS provider tracking playback calls."""

    def __init__(self, on_speak_fn: Any = None) -> None:
        self.spoken_texts: list[str] = []
        self.interrupted = False
        self._is_playing = False
        self.on_speak_fn = on_speak_fn

    def is_playing(self) -> bool:
        return self._is_playing

    def speak(self, text: str, language: str = "bn", blocking: bool = False) -> None:
        self._is_playing = True
        try:
            self.spoken_texts.append(text)
            if self.on_speak_fn is not None:
                self.on_speak_fn(text)
        finally:
            self._is_playing = False

    def interrupt(self) -> None:
        self.interrupted = True
        self._is_playing = False


# ==============================================================================
# WakeWordDetector Tests
# ==============================================================================


def test_wake_word_detector_detects_phrase_and_debounces():
    """Verify detector triggers on 'Hey KB' and debounces within cooldown."""

    def mock_detect(chunk: bytes) -> WakeWordEvent | None:
        if b"HEY_KB" in chunk:
            return WakeWordEvent(detected=True, phrase="hey kb")
        return None

    detector = WakeWordDetector(
        wake_phrase="hey kb",
        cooldown_seconds=1.0,
        detect_fn=mock_detect,
    )
    detector.initialize()

    # 1. Normal unrelated audio
    event = detector.process_audio_chunk(b"random ambient noise")
    assert event is None

    # 2. Audio containing 'Hey KB'
    event = detector.process_audio_chunk(b"prefix HEY_KB suffix")
    assert event is not None
    assert event.detected is True
    assert event.phrase == "hey kb"

    # 3. Immediate repeat within 1.0s cooldown is suppressed
    repeat_event = detector.process_audio_chunk(b"prefix HEY_KB suffix")
    assert repeat_event is None

    # 4. After disabling, wake detection stops
    detector.disable()
    assert detector.is_enabled is False
    disabled_event = detector.process_audio_chunk(b"prefix HEY_KB suffix")
    assert disabled_event is None

    # 5. Re-enable restores detection (after cooldown)
    time.sleep(1.05)
    detector.enable()
    assert detector.is_enabled is True
    resumed_event = detector.process_audio_chunk(b"prefix HEY_KB suffix")
    assert resumed_event is not None
    assert resumed_event.detected is True


def test_wake_word_detector_regex_matching():
    """Verify acoustic variant matching logic for 'Hey KB'."""
    detector = WakeWordDetector(wake_phrase="hey kb")

    assert detector._matches_wake_phrase("hey kb") is True
    assert detector._matches_wake_phrase("Hey KB!") is True
    assert detector._matches_wake_phrase("hey k b") is True
    assert detector._matches_wake_phrase("hey kay bee") is True
    assert detector._matches_wake_phrase("hi kb") is True
    assert detector._matches_wake_phrase("something else entirely") is False
    assert detector._matches_wake_phrase("what time is it") is False


# ==============================================================================
# SharedMicrophoneCapture & Pre-roll Buffer Tests
# ==============================================================================


def test_shared_capture_pre_roll_buffer_retention_and_routing():
    """Verify shared capture buffers ~1.5s of pre-roll chunks and feeds STT queue."""
    wake_events = []

    def on_wake(event: WakeWordEvent) -> None:
        wake_events.append(event)

    def mock_detect(chunk: bytes) -> WakeWordEvent | None:
        if b"WAKE_PHRASE" in chunk:
            return WakeWordEvent(detected=True, phrase="hey kb")
        return None

    detector = WakeWordDetector(detect_fn=mock_detect, cooldown_seconds=0.1)
    capture = SharedMicrophoneCapture(
        wake_detector=detector,
        on_wake_word=on_wake,
        chunk_ms=30,
        pre_roll_seconds=0.3,  # 10 chunks buffer
    )

    # 1. Feed idle chunks
    for i in range(15):
        capture.feed_simulated_chunk(f"idle_chunk_{i}".encode())

    # Pre-roll buffer should hold at most 10 chunks
    assert len(capture._pre_roll_buffer) == 10
    assert b"idle_chunk_14" in capture._pre_roll_buffer[-1]

    # 2. Feed wake trigger chunk
    capture.feed_simulated_chunk(b"WAKE_PHRASE")
    assert len(wake_events) == 1
    assert wake_events[0].phrase == "hey kb"

    # 3. Transition to active STT session
    consumer_queue = capture.begin_active_session()
    assert capture._is_active_session is True
    assert detector.is_enabled is False  # Wake word paused during active turn

    # Pre-roll chunks should immediately be in the queue
    received_pre_roll = []
    while not consumer_queue.empty():
        received_pre_roll.append(consumer_queue.get_nowait())
    assert len(received_pre_roll) >= 10

    # 4. Subsequent chunks routed to queue, not detector
    capture.feed_simulated_chunk(b"user_speech_frame_1")
    queued_frame = consumer_queue.get_nowait()
    assert queued_frame == b"user_speech_frame_1"

    # 5. End active session returns pipeline to idle
    capture.end_active_session()
    assert capture._is_active_session is False
    assert detector.is_enabled is True


def test_shared_capture_mute_drops_audio():
    """Verify muting capture drops all audio chunks and stops wake detection."""
    detector = MagicMock()
    capture = SharedMicrophoneCapture(wake_detector=detector)

    capture.mute()
    assert capture.is_muted is True

    capture.feed_simulated_chunk(b"audio_while_muted")
    # Pre-roll buffer remains empty
    assert len(capture._pre_roll_buffer) == 0
    # Wake detector process_audio_chunk never called
    detector.process_audio_chunk.assert_not_called()

    capture.unmute()
    assert capture.is_muted is False


# ==============================================================================
# DialogueCoordinator 8-State Machine & Inactivity Timeout Tests
# ==============================================================================


@pytest.fixture
def mock_coordinator():
    bus = EventBus()
    knowledge_svc = KnowledgeService(institution_type="education")
    llm = MockLLMProvider()
    stt = FakeSTTProvider(transcription="Tell me about AIUB.")
    tts = FakeTTSProvider()

    detector = WakeWordDetector(cooldown_seconds=0.1)
    capture = SharedMicrophoneCapture(wake_detector=detector)

    coord = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_svc,
        llm_provider=llm,
        stt_provider=stt,
        tts_provider=tts,
        default_language="en",
        shared_capture=capture,
        inactivity_timeout_seconds=0.25,  # Fast 250ms timeout for testing
    )
    coord._set_state("idle")
    yield coord
    coord.close()


def test_coordinator_wake_word_triggers_listening(mock_coordinator):
    """Verify 'Hey KB' event transitions coordinator from IDLE to LISTENING."""
    state_transitions = []
    mock_coordinator.register_state_callback(lambda s, _meta: state_transitions.append(s))

    assert mock_coordinator.current_state == "idle"

    # Send wake word event
    mock_coordinator.on_wake_word_detected(WakeWordEvent(detected=True, phrase="hey kb"))

    # State transitions to listening immediately
    assert "listening" in state_transitions
    assert mock_coordinator.current_state in ("listening", "thinking", "speaking")


def test_coordinator_post_answer_remains_in_listening_and_inactivity_timeout_returns_to_idle(
    mock_coordinator,
):
    """Verify after speaking answer, state returns to LISTENING, and inactivity timeout returns to IDLE."""
    state_transitions = []
    mock_coordinator.register_state_callback(lambda s, _meta: state_transitions.append(s))

    # Execute a turn
    turn = mock_coordinator.handle_turn(user_text="Hello robot")
    assert turn.role == "model"

    # Immediately after turn, coordinator is in LISTENING to allow follow-up questions
    assert mock_coordinator.current_state == "listening"
    assert "answering" in state_transitions
    assert "listening" in state_transitions

    # Wait for the 250ms inactivity timeout to fire
    time.sleep(0.4)

    # Coordinator must automatically return to IDLE and resume wake-word detection
    assert mock_coordinator.current_state == "idle"
    assert state_transitions[-1] == "idle"


def test_coordinator_mute_and_unmute_lifecycle(mock_coordinator):
    """Verify left tap mute sets MUTED and unmute returns to IDLE."""
    mock_coordinator._set_state("listening")
    assert mock_coordinator.current_state == "listening"

    # 1. Mute
    is_muted = mock_coordinator.mute_microphone()
    assert is_muted is True
    assert mock_coordinator.current_state == "muted"
    assert mock_coordinator.is_microphone_muted is True
    assert mock_coordinator._shared_capture.is_muted is True

    # 2. Wake word during mute is ignored
    mock_coordinator.on_wake_word_detected(WakeWordEvent(detected=True, phrase="hey kb"))
    assert mock_coordinator.current_state == "muted"

    # 3. Unmute
    is_muted = mock_coordinator.unmute_microphone()
    assert is_muted is False
    assert mock_coordinator.current_state == "idle"
    assert mock_coordinator.is_microphone_muted is False
    assert mock_coordinator._shared_capture.is_muted is False


def test_coordinator_interruption_transitions_to_listening(mock_coordinator):
    """Verify right tap interruption stops TTS and transitions to LISTENING for follow-ups."""
    state_transitions = []
    mock_coordinator.register_state_callback(lambda s, _meta: state_transitions.append(s))

    mock_coordinator._set_state("answering")
    mock_coordinator.interrupt()

    assert mock_coordinator._tts_provider.interrupted is True
    assert "interrupted" in state_transitions

    # After 1.0s animation, state transitions to listening
    time.sleep(1.2)
    assert mock_coordinator.current_state == "listening"

    # After inactivity timeout, state transitions to idle
    time.sleep(0.4)
    assert mock_coordinator.current_state == "idle"


def test_coordinator_wake_word_and_follow_up_utterance_flow():
    """Verify 'Hey KB' opens active voice session, strips wake words, captures
    'Who are you?' and follow-up questions without re-triggering wake word,
    and returns to IDLE after inactivity."""
    bus = EventBus()
    knowledge_svc = KnowledgeService(institution_type="education")
    llm = MockLLMProvider()
    stt = FakeSTTProvider(
        transcriptions=[
            "Hey KB",  # Pre-roll chunk alone -> stripped, continues listening
            "Who are you?",  # Spoken prompt -> answered
            "What is your purpose?",  # Follow-up question without wake word -> answered
            "",  # Inactivity silence
        ]
    )
    tts = FakeTTSProvider()
    capture = SharedMicrophoneCapture(wake_detector=WakeWordDetector())

    coord = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_svc,
        llm_provider=llm,
        stt_provider=stt,
        tts_provider=tts,
        default_language="en",
        shared_capture=capture,
        inactivity_timeout_seconds=0.3,  # Fast 300ms timeout for test
    )
    coord._set_state("idle")

    state_transitions = []
    coord.register_state_callback(lambda s, _meta: state_transitions.append(s))

    try:
        # 1. Trigger wake word
        coord.on_wake_word_detected(WakeWordEvent(detected=True, phrase="hey kb"))
        assert coord._is_active_voice_session is True

        # Wait for worker thread to process turns
        # 1st: "Hey KB" stripped -> continues
        # 2nd: "Who are you?" answered -> speaks
        # 3rd: "What is your purpose?" answered -> speaks
        # 4th: "" silence -> wait for inactivity timeout (0.3s)
        time.sleep(1.0)

        # Verify spoken turns were delivered
        assert len(tts.spoken_texts) == 2
        assert any("Who are you?" in txt for txt in tts.spoken_texts)
        assert any("What is your purpose?" in txt for txt in tts.spoken_texts)

        # Verify state transitions: listening -> thinking -> answering -> listening -> thinking -> answering -> listening -> idle
        assert "thinking" in state_transitions
        assert "answering" in state_transitions
        assert coord.current_state == "idle"
        assert coord._is_active_voice_session is False

    finally:
        coord.close()


def test_shared_capture_half_duplex_pause_and_resume():
    """Verify SharedMicrophoneCapture drops audio, disables wake detector,
    and clears buffers while speaker playback is active."""
    wake_events = []

    def mock_detect(chunk: bytes) -> WakeWordEvent | None:
        if b"WAKE_PHRASE" in chunk:
            return WakeWordEvent(detected=True, phrase="hey kb")
        return None

    detector = WakeWordDetector(detect_fn=mock_detect, cooldown_seconds=0.01)
    capture = SharedMicrophoneCapture(
        wake_detector=detector,
        on_wake_word=lambda evt: wake_events.append(evt),
        chunk_ms=30,
        pre_roll_seconds=0.3,
    )

    # 1. Normal idle audio feeds pre-roll buffer
    capture.feed_simulated_chunk(b"chunk1")
    capture.feed_simulated_chunk(b"chunk2")
    assert len(capture._pre_roll_buffer) == 2

    # 2. Pause for speaker playback (half-duplex lock engaged)
    capture.pause_for_playback()
    assert capture.is_playback_active is True
    # Buffer is purged immediately
    assert len(capture._pre_roll_buffer) == 0
    assert detector.is_enabled is False

    # 3. Audio arriving while speaker is playing is dropped (not buffered, no wake trigger)
    event = capture.feed_simulated_chunk(b"chunk_during_playback_WAKE_PHRASE")
    assert event is None
    assert len(wake_events) == 0
    assert len(capture._pre_roll_buffer) == 0

    # 4. Resume after speaker playback completes (half-duplex lock released)
    capture.resume_after_playback()
    assert capture.is_playback_active is False
    assert len(capture._pre_roll_buffer) == 0  # Stale echo buffer cleared
    assert detector.is_enabled is True

    # 5. Audio arriving after playback completes is captured normally
    event = capture.feed_simulated_chunk(b"user_speech_WAKE_PHRASE")
    assert event is not None
    assert event.phrase == "hey kb"
    assert len(wake_events) == 1
    assert len(capture._pre_roll_buffer) == 1


def test_coordinator_prevents_microphone_listening_during_tts_playback():
    """Verify DialogueCoordinator engages half-duplex lock during TTS playback,
    preventing wake detection, self-hearing, and manual Robot Mic activation."""
    bus = EventBus()
    knowledge_svc = KnowledgeService(institution_type="education")
    llm = MockLLMProvider()
    stt = FakeSTTProvider("User spoke")

    playback_active_during_speak = []
    capture_ref = None
    coord_ref = None

    def on_speak_hook(text: str) -> None:
        # Check half-duplex state while TTS is actively speaking
        playback_active_during_speak.append(capture_ref.is_playback_active)
        playback_active_during_speak.append(coord_ref.is_speaker_active)

        # Attempt to feed microphone audio containing wake word while robot speaks -> must be ignored!
        wake_evt = capture_ref.feed_simulated_chunk(b"WAKE_PHRASE_DURING_TTS")
        assert wake_evt is None

        # Attempt manual Robot Mic activation while robot speaks -> must be rejected with SPEAKER_ACTIVE!
        busy_turn = coord_ref.handle_turn(user_text=None, language="en")
        assert busy_turn.metadata.get("error") == "SPEAKER_ACTIVE"

    tts = FakeTTSProvider(on_speak_fn=on_speak_hook)

    def mock_detect(chunk: bytes) -> WakeWordEvent | None:
        if b"WAKE_PHRASE" in chunk:
            return WakeWordEvent(detected=True, phrase="hey kb")
        return None

    detector = WakeWordDetector(detect_fn=mock_detect, cooldown_seconds=0.01)
    capture = SharedMicrophoneCapture(wake_detector=detector)
    capture_ref = capture

    coord = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_svc,
        llm_provider=llm,
        stt_provider=stt,
        tts_provider=tts,
        default_language="en",
        shared_capture=capture,
        inactivity_timeout_seconds=0.5,
    )
    coord_ref = coord
    coord._set_state("idle")

    try:
        # Execute turn with speech playback
        turn = coord.handle_turn(user_text="Tell me something", speak_response=True)
        assert turn.role == "model"

        # Verify half-duplex state was active during speak hook
        assert len(playback_active_during_speak) == 2
        assert playback_active_during_speak[0] is True  # capture.is_playback_active
        assert playback_active_during_speak[1] is True  # coord.is_speaker_active

        # After playback finishes, half-duplex lock is released and capture resumes
        assert capture.is_playback_active is False
        assert coord.is_speaker_active is False
        assert coord.current_state == "listening"

    finally:
        coord.close()


def test_tts_failure_and_interrupt_safely_resumes_capture():
    """Verify that if TTS fails with an exception or is interrupted,
    the microphone half-duplex lock is never left permanently engaged."""
    bus = EventBus()
    knowledge_svc = KnowledgeService(institution_type="education")
    llm = MockLLMProvider()
    stt = FakeSTTProvider("User spoke")

    class FailingTTSProvider:
        def __init__(self) -> None:
            self._is_playing = False

        def is_playing(self) -> bool:
            return self._is_playing

        def speak(self, text: str, language: str = "bn", blocking: bool = False) -> None:
            raise RuntimeError("Hardware speaker disconnected during playback")

        def interrupt(self) -> None:
            pass

    tts = FailingTTSProvider()
    capture = SharedMicrophoneCapture(wake_detector=WakeWordDetector())

    coord = DialogueCoordinator(
        event_bus=bus,
        knowledge_service=knowledge_svc,
        llm_provider=llm,
        stt_provider=stt,
        tts_provider=tts,
        default_language="en",
        shared_capture=capture,
    )
    coord._set_state("idle")

    try:
        # Execute turn where TTS raises RuntimeError
        turn = coord.handle_turn(user_text="Hello", speak_response=True)
        assert turn.metadata.get("tts_error") is True

        # Half-duplex lock MUST be safely released despite TTS failure!
        assert capture.is_playback_active is False
        assert coord.is_speaker_active is False

    finally:
        coord.close()
