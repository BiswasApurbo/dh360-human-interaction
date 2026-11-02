"""Local, lightweight wake-word detector ("Hey KB") for Raspberry Pi 5.

Architectural and Privacy Invariants:
1. Strictly Local: Audio frames are processed entirely on-device (CPU-only).
   Zero cloud APIs, network requests, or external telemetry.
2. Ephemeral Processing: Microphone audio chunks are inspected in-memory and
   immediately discarded; audio recordings are NEVER persisted to disk or cloud.
3. Debounced and State-Aware: Avoids duplicate triggers using a configurable
   cooldown timer and explicit active/dormant toggle.
4. Dependency Isolation: Gracefully falls back when Vosk or native libraries
   are unavailable or during test suite execution.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Accepted acoustic variants of "Hey KB" in Vosk vocabulary
DEFAULT_WAKE_VARIANTS = (
    "hey k b",
    "hey kay bee",
    "hi k b",
    "hi kay bee",
)


@dataclass
class WakeWordEvent:
    """Represents a detected wake-word instance."""

    detected: bool
    phrase: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0


class WakeWordDetector:
    """Lightweight local keyword spotter targeting 'Hey KB'."""

    def __init__(
        self,
        wake_phrase: str = "hey kb",
        sample_rate: int = 16000,
        model_path: str | Path | None = None,
        cooldown_seconds: float = 2.0,
        detect_fn: Callable[[bytes], WakeWordEvent | None] | None = None,
    ) -> None:
        self.wake_phrase = wake_phrase.strip().lower()
        self.sample_rate = sample_rate
        self.cooldown_seconds = cooldown_seconds
        self._detect_fn = detect_fn

        self._lock = threading.Lock()
        self._is_enabled: bool = True
        self._last_detected_time: float = 0.0

        # Vosk engine handle
        self._vosk_model: Any | None = None
        self._vosk_recognizer: Any | None = None
        self._vosk_initialized: bool = False
        self._model_path = model_path

        # Custom wake phrases list for grammar
        self._variants = set(DEFAULT_WAKE_VARIANTS)
        self._variants.add(self.wake_phrase)

    def initialize(self) -> bool:
        """Initialize local keyword spotter engine if available."""
        with self._lock:
            if self._vosk_initialized:
                return True

            if self._detect_fn is not None:
                self._vosk_initialized = True
                return True

            try:
                import vosk

                # Suppress verbose Kaldi logs to stderr
                vosk.SetLogLevel(-1)

                if self._model_path and Path(self._model_path).exists():
                    self._vosk_model = vosk.Model(str(self._model_path))
                else:
                    # Automatically resolves ~/.cache/vosk or default small model
                    self._vosk_model = vosk.Model(lang="en-us")

                # Grammar constrains recognition strictly to wake variants + unknown
                grammar_list = list(self._variants) + ["[unk]"]
                grammar_json = json.dumps(grammar_list)

                self._vosk_recognizer = vosk.KaldiRecognizer(
                    self._vosk_model,
                    self.sample_rate,
                    grammar_json,
                )
                self._vosk_initialized = True
                logger.info("WakeWordDetector: local Vosk keyword model initialized for phrase '%s'", self.wake_phrase)
                return True

            except Exception as exc:
                logger.warning(
                    "WakeWordDetector: local Vosk engine initialization deferred or unavailable: %s",
                    exc,
                )
                self._vosk_initialized = False
                return False

    @property
    def is_enabled(self) -> bool:
        """Whether wake-word detection is actively inspecting incoming audio."""
        with self._lock:
            return self._is_enabled

    def enable(self) -> None:
        """Enable wake-word inspection."""
        with self._lock:
            self._is_enabled = True
            self.reset_state()

    def disable(self) -> None:
        """Disable wake-word inspection (e.g. while active dialogue turn is underway)."""
        with self._lock:
            self._is_enabled = False
            self.reset_state()

    def reset_state(self) -> None:
        """Reset internal recognizer stream buffer."""
        if self._vosk_recognizer is not None:
            try:
                self._vosk_recognizer.Reset()
            except Exception:
                pass

    def _matches_wake_phrase(self, text: str) -> bool:
        """Check if transcribed phrase contains any wake word variant."""
        clean = re.sub(r"[^\w\s]", "", text.strip().lower())
        if not clean:
            return False
        normalized = clean.replace("kb", "k b")
        for variant in self._variants:
            clean_variant = re.sub(r"[^\w\s]", "", variant).replace("kb", "k b")
            if clean_variant in normalized:
                return True
        return False

    def process_audio_chunk(self, audio_data: bytes) -> WakeWordEvent | None:
        """Process a raw PCM chunk (16kHz 16-bit mono) and detect 'Hey KB'.

        Args:
            audio_data: Raw PCM audio bytes.

        Returns:
            WakeWordEvent if 'Hey KB' detected and cooldown passed, else None.
        """
        if not audio_data:
            return None

        with self._lock:
            if not self._is_enabled:
                return None

            now = time.time()
            if now - self._last_detected_time < self.cooldown_seconds:
                return None

            # 1. Injected custom / mock detector (testing or custom acoustic)
            if self._detect_fn is not None:
                event = self._detect_fn(audio_data)
                if event and event.detected:
                    self._last_detected_time = now
                    return event
                return None

            # 2. Local Vosk Kaldi Keyword Recognizer
            if not self._vosk_initialized and not self.initialize():
                return None

            if self._vosk_recognizer is None:
                return None

            try:
                # AcceptWaveform returns True on full phrase boundary
                if self._vosk_recognizer.AcceptWaveform(audio_data):
                    res_json = self._vosk_recognizer.Result()
                    res_data = json.loads(res_json)
                    text = res_data.get("text", "")
                    if self._matches_wake_phrase(text):
                        logger.info("WakeWordDetector: 'Hey KB' wake word detected! (Result: %s)", text)
                        self._last_detected_time = now
                        self.reset_state()
                        return WakeWordEvent(detected=True, phrase=text)
                else:
                    # Partial result check for ultra-low latency response
                    partial_json = self._vosk_recognizer.PartialResult()
                    partial_data = json.loads(partial_json)
                    partial_text = partial_data.get("partial", "")
                    if self._matches_wake_phrase(partial_text):
                        logger.info("WakeWordDetector: 'Hey KB' detected via partial recognition: '%s'", partial_text)
                        self._last_detected_time = now
                        self.reset_state()
                        return WakeWordEvent(detected=True, phrase=partial_text)

            except Exception as exc:
                logger.debug("WakeWordDetector: error processing audio chunk: %s", exc)

        return None
