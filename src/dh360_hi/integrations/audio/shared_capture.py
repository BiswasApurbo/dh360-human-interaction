"""Shared microphone capture pipeline with circular pre-roll buffer.

Guarantees:
1. Single hardware capture stream avoiding ALSA / PortAudio device contention (EBUSY).
2. Continuous circular pre-roll buffer (~1.5s) preventing speech loss after wake detection.
3. Decoupled distribution:
   - Idle state: audio frames are sent to local WakeWordDetector.
   - Active dialogue state: audio frames + pre-roll buffer are piped to Soniox STT.
4. Fault tolerance: automatically recovers from audio device disconnections without restarting service.
5. Zero persistent audio storage: all processing strictly in-memory.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import queue
import threading
from collections.abc import Callable

from dh360_hi.integrations.audio.wake_detector import WakeWordDetector, WakeWordEvent

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
MIC_SAMPLE_RATE = 16000  # Default target rate
CHUNK_MS = 30
CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_MS / 1000)  # 480 samples = 960 bytes
PRE_ROLL_SECONDS = 1.5
MAX_PRE_ROLL_CHUNKS = int(PRE_ROLL_SECONDS * 1000 / CHUNK_MS)  # ~50 chunks = 48KB


class SharedMicrophoneCapture:
    """Manages a single hardware microphone stream and distributes frames to listeners."""

    def __init__(
        self,
        wake_detector: WakeWordDetector | None = None,
        sample_rate: int = TARGET_SAMPLE_RATE,
        chunk_ms: int = CHUNK_MS,
        pre_roll_seconds: float = PRE_ROLL_SECONDS,
        on_wake_word: Callable[[WakeWordEvent], None] | None = None,
        device_index: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.wake_detector = wake_detector or WakeWordDetector(sample_rate=sample_rate)
        self.on_wake_word = on_wake_word
        self.device_index = device_index

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Ring buffer storing recent chunks for pre-roll
        max_chunks = max(10, int(pre_roll_seconds * 1000 / chunk_ms))
        self._pre_roll_buffer: collections.deque[bytes] = collections.deque(maxlen=max_chunks)

        # Active speech consumer queue
        self._active_consumer_queue: queue.Queue[bytes | None] | None = None
        self._is_active_session: bool = False
        self._is_muted: bool = False
        self._is_hardware_available: bool = False

    @property
    def is_running(self) -> bool:
        """Whether capture background worker is active."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_muted(self) -> bool:
        """Whether audio capture is muted."""
        with self._lock:
            return self._is_muted

    @property
    def is_hardware_available(self) -> bool:
        """Whether a physical microphone capture device is operational."""
        with self._lock:
            return self._is_hardware_available

    def start(self) -> None:
        """Start the background audio capture thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._capture_worker,
                name="KBCON-SharedMicCapture",
                daemon=True,
            )
            self._thread.start()
            logger.info("SharedMicrophoneCapture: background capture worker started")

    def stop(self) -> None:
        """Stop the background capture thread."""
        self._stop_event.set()
        with self._lock:
            q = self._active_consumer_queue
            if q is not None:
                with contextlib.suppress(Exception):
                    q.put_nowait(None)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        logger.info("SharedMicrophoneCapture: background capture worker stopped")

    def mute(self) -> None:
        """Mute microphone: drops audio chunks and pauses wake-word detection."""
        with self._lock:
            self._is_muted = True
            self._pre_roll_buffer.clear()
            if self.wake_detector is not None:
                self.wake_detector.disable()
        logger.info("SharedMicrophoneCapture: muted")

    def unmute(self) -> None:
        """Unmute microphone: resumes audio capture and wake-word monitoring."""
        with self._lock:
            self._is_muted = False
            self._pre_roll_buffer.clear()
            if self.wake_detector is not None:
                self.wake_detector.enable()
        logger.info("SharedMicrophoneCapture: unmuted")

    def begin_active_session(self) -> queue.Queue[bytes | None]:
        """Switch pipeline from idle wake detection to active speech recognition.

        Returns a Queue that immediately receives all pre-roll audio frames
        followed by continuous live PCM chunks.
        """
        with self._lock:
            self._is_active_session = True
            if self.wake_detector is not None:
                self.wake_detector.disable()

            consumer_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=120)

            # Pre-load pre-roll buffer chunks
            for chunk in self._pre_roll_buffer:
                with contextlib.suppress(queue.Full):
                    consumer_queue.put_nowait(chunk)

            self._active_consumer_queue = consumer_queue
            logger.info(
                "SharedMicrophoneCapture: active dialogue session started with %d pre-roll chunks",
                len(self._pre_roll_buffer),
            )
            return consumer_queue

    def end_active_session(self) -> None:
        """End active dialogue session and return pipeline to idle wake detection."""
        with self._lock:
            self._is_active_session = False
            q = self._active_consumer_queue
            self._active_consumer_queue = None
            if q is not None:
                with contextlib.suppress(Exception):
                    q.put_nowait(None)

            self._pre_roll_buffer.clear()
            if not self._is_muted and self.wake_detector is not None:
                self.wake_detector.enable()

            logger.info(
                "SharedMicrophoneCapture: ended active session; returned to idle wake-word monitoring"
            )

    def feed_simulated_chunk(self, chunk: bytes) -> WakeWordEvent | None:
        """Feed synthetic audio chunk (useful for testing or virtual pipes)."""
        return self._process_chunk(chunk)

    def _process_chunk(self, chunk: bytes) -> WakeWordEvent | None:
        """Route single audio chunk according to active state."""
        with self._lock:
            if self._is_muted:
                return None

            # 1. Maintain circular pre-roll buffer
            self._pre_roll_buffer.append(chunk)

            # 2. If active speech recognition is ongoing, feed STT consumer queue
            if self._is_active_session and self._active_consumer_queue is not None:
                try:
                    self._active_consumer_queue.put_nowait(chunk)
                except queue.Full:
                    # Drop oldest if queue fills up to avoid memory growth
                    pass
                return None

            # 3. Otherwise, in Idle state, run local wake detector
            detector = self.wake_detector

        if detector is not None and detector.is_enabled:
            event = detector.process_audio_chunk(chunk)
            if event and event.detected:
                if self.on_wake_word is not None:
                    try:
                        self.on_wake_word(event)
                    except Exception as exc:
                        logger.warning("SharedMicrophoneCapture: wake callback error: %s", exc)
                return event

        return None

    def _capture_worker(self) -> None:
        """Continuous background capture loop with automatic device reconnection."""
        while not self._stop_event.is_set():
            stream = None
            try:
                import sounddevice as sd

                # Open 16kHz mono int16 stream
                stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=self.chunk_samples,
                    device=self.device_index,
                )
                stream.start()

                with self._lock:
                    self._is_hardware_available = True
                logger.info("SharedMicrophoneCapture: audio input stream opened successfully")

                while not self._stop_event.is_set():
                    audio_data, overflowed = stream.read(self.chunk_samples)
                    if overflowed:
                        logger.debug("SharedMicrophoneCapture: audio buffer overflowed")
                    raw_bytes = audio_data.tobytes()
                    self._process_chunk(raw_bytes)

            except Exception as exc:
                with self._lock:
                    self._is_hardware_available = False
                logger.warning(
                    "SharedMicrophoneCapture: audio device error (%s); retrying in 2s...",
                    type(exc).__name__,
                )
                # Sleep with stop event awareness
                self._stop_event.wait(2.0)

            finally:
                if stream is not None:
                    with contextlib.suppress(Exception):
                        stream.stop()
                        stream.close()
