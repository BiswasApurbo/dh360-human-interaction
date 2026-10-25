"""DH360-compatible provider adapters wrapping Eliva speech and AI integrations.

Provides concrete adapters conforming to:
- `LLMProvider`: Generative conversational AI via Gemini
- `STTProvider`: Speech-to-text recognition via Soniox
- `TTSProvider`: Text-to-speech audio synthesis via Google Cloud TTS

Architectural Invariants:
1. Zero top-level SDK imports: Optional third-party packages (google-genai,
   sounddevice, soxr, google-cloud-texttospeech) are never imported at module level.
   DH360 and its test suites can be imported and executed without these packages installed.
2. Safe lifecycle and resource isolation: Microphones, sound devices, and network
   sessions are never initialized at import time or upon adapter construction.
3. Isolated conversational history: LLM dialogue history is owned strictly by DH360
   and passed ephemerally per turn. No shared global chat session is maintained across users.
4. Typed error taxonomy & redaction: All provider exceptions are translated to typed
   DH360 ApplicationErrors (`LLMProviderError`, `STTProviderError`, `TTSProviderError`,
   `ProviderNetworkError`, `ProviderTimeoutError`, `HardwareUnavailableError`).
   Raw exception messages, API keys, private tokens, and sensitive prompts are never leaked.
5. Dependency injection: Adapters support optional injected clients and callables for
   offline, credential-free, hardware-free testing.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ==============================================================================
# Eliva Character & System Instructions
# ==============================================================================

DEFAULT_ELIVA_SYSTEM_INSTRUCTION = """\
You are Eliva, a humanoid robot built by Robotech Valley.

CORE IDENTITY & OVERVIEW
    - You are a physical humanoid robot, not an AI model or software assistant.
    - You are smart, approachable, polite, and friendly.
    - STRICT IDENTITY RULE: You must ONLY say
      "আমি এলিভা। রোবোটেক ভ্যালির একটি মানবাকৃতির রোবট। আমি মানুষকে সাহায্য করি।"
      if the user explicitly asks about YOU (e.g., "Who are you?", "তুমি কে?").
    - NEVER give your identity response if the user asks about another person.

KNOWLEDGE SOURCE HIERARCHY
    1. Primary Knowledge: Use the provided context for questions about Robotech Valley,
       company background, product specs, team, and internal history.
    2. Event Knowledge: Use the Bangladesh Innovation Fair 2026 details provided in this context.
    3. Real-Time Web Knowledge: Use search for weather, external news, and general world facts.
    4. Missing Information: If information is not found in the provided facts,
       event details, or web search, state politely in one simple sentence that you do
       not have that information yet.

LANGUAGE & VOICE CONTROL
    - Your default and primary language is Bangla (written in Bangla script).
    - If the user types in Bangla or Banglish (Bangla written using Latin alphabet),
      ALWAYS reply strictly in Bangla script.
    - Reply in English ONLY if the user asks their question entirely in English.
    - Never mix Bangla and English words in the same sentence.

TEXT-TO-SPEECH (TTS) OPTIMIZATION RULES
    - Keep all responses under three short sentences unless explicitly requested otherwise.
    - Speak only in plain text complete sentences.
    - Never use markdown formatting like bold text, italics, or headers.
    - Never use bullet points, dash lists, numbered lists, emojis, or symbols.
    - Never output brackets, quotes, or special characters.
    - Spell out years and numbers in full words for proper pronunciation
      (e.g.: use "দুই হাজার ছাব্বিশ" instead of "২০২৬").

GREETING PROTOCOL
    - When greeted by a visitor, reply with:
      "আসসালামু আলাইকুম। বাংলাদেশ ইনোভেশন ফেয়ার দুই হাজার ছাব্বিশে আপনাকে স্বাগতম।"
    - Follow up by naturally responding to their prompt.

EVENT CONTEXT
    - Event: Bangladesh Innovation Fair 2026 (বাংলাদেশ ইনোভেশন ফেয়ার দুই হাজার ছাব্বিশ).
    - Theme: Innovate to Market (ইনোভেশন টু মার্কেট).
    - Organizers: Ministry of Science and Technology and NMST.
    - Location: National Museum of Science and Technology premises, Agargaon, Dhaka.
    - Dates: September 12 to September 14, 2026.

EVENT PURPOSE STATEMENT
    - If asked why you are here or about the event, say:
      "আমি রোবোটেক ভ্যালি থেকে আসা রোবট এলিভা। আমি বাংলাদেশ ইনোভেশন ফেয়ার দুই হাজার "
      "ছাব্বিশে সবাইকে সাহায্য করতে এসেছি।"

BEHAVIORAL BOUNDARIES
    - Maintain a high-energy, warm, and welcoming tone suitable for an exhibition.
    - Never break character.

CRITICAL LOCAL KNOWLEDGE RULE:
    - If the user query is prefixed with "[LOCAL KNOWLEDGE MATCHES]", you MUST base
      your response STRICTLY on the facts provided in that prefix.
    - Overwrite any internal knowledge or pre-trained facts with the local matches.
    - Do not mention phrases like "According to local knowledge" to the user."""


# ==============================================================================
# 1. ElivaGeminiAdapter (LLMProvider)
# ==============================================================================


class ElivaGeminiAdapter(LLMProvider):
    """Adapter wrapping Google GenAI / Gemini for DH360 conversational dialogue.

    Implements `LLMProvider`. Generates natural language responses using isolated
    single-turn context assembly without persistent cross-session state.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3.5-flash-lite",
        system_instruction: str = DEFAULT_ELIVA_SYSTEM_INSTRUCTION,
        client: Any | None = None,
        generator_fn: Callable[..., str] | None = None,
    ) -> None:
        """Initialize ElivaGeminiAdapter without performing network or SDK operations.

        Args:
            api_key: Optional Gemini API key. Defaults to GEMINI_API_KEY env var.
            model_name: Gemini model identifier.
            system_instruction: System prompt enforcing Eliva persona and guidelines.
            client: Optional injected GenAI client for testing.
            generator_fn: Optional injected generation callable for testing.
        """
        self._api_key = api_key
        self.model_name = model_name
        self.system_instruction = system_instruction
        self._client = client
        self._generator_fn = generator_fn

    def _get_api_key(self) -> str:
        key = self._api_key or os.getenv("GEMINI_API_KEY", "")
        return key.strip()

    def generate_response(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> str:
        """Generate conversational response using pre-filtered context and isolated history.

        Args:
            prompt: User query or utterance text.
            filtered_context: Pre-filtered knowledge facts assembled by DH360 KnowledgeService.
            language: Target language ('bn' or 'en').
            session_history: Isolated turn history scoped only to the current person's session.

        Returns:
            Plaintext response text formatted for natural speech output.

        Raises:
            LLMProviderError: If generation fails, SDK is missing, or API key is absent.
            ProviderTimeoutError: If Gemini API request times out.
            ProviderNetworkError: If network connectivity fails.
        """
        if not prompt or not prompt.strip():
            return ""

        # 1. Injected generator callable (testing / custom runner)
        if self._generator_fn is not None:
            try:
                return self._generator_fn(
                    prompt=prompt,
                    filtered_context=filtered_context,
                    language=language,
                    session_history=session_history,
                )
            except (LLMProviderError, ProviderTimeoutError, ProviderNetworkError):
                raise
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    message="Gemini generator timed out.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc
            except ConnectionError as exc:
                raise ProviderNetworkError(
                    message="Gemini connection failed.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc
            except Exception as exc:
                logger.error("ElivaGeminiAdapter: generation failed: %s", type(exc).__name__)
                raise LLMProviderError(
                    message="Gemini language model generation failed.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc

        # 2. Injected or lazily constructed Google GenAI client
        client = self._client
        if client is None:
            api_key = self._get_api_key()
            if not api_key:
                raise LLMProviderError(
                    message="Gemini API key is not configured.",
                    provider_name="gemini",
                    details={"reason": "MISSING_API_KEY", "model_name": self.model_name},
                )

            try:
                from google import genai  # type: ignore[import-untyped]

                client = genai.Client(api_key=api_key)
            except ImportError as exc:
                raise LLMProviderError(
                    message="Optional dependency 'google-genai' is not installed.",
                    provider_name="gemini",
                    details={"reason": "SDK_NOT_INSTALLED", "model_name": self.model_name},
                ) from exc
            except Exception as exc:
                logger.error(
                    "ElivaGeminiAdapter: client initialization failed: %s", type(exc).__name__
                )
                raise LLMProviderError(
                    message="Failed to initialize Gemini client.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc

        # 3. Assemble prompt with pre-filtered context and language directive
        assembled_prompt = self._assemble_prompt(prompt, filtered_context, language)

        # 4. Generate content ephemerally (no persistent cross-session state)
        try:
            from google.genai import types  # type: ignore[import-untyped]

            config = types.GenerateContentConfig(
                system_instruction=self.system_instruction,
            )

            # Pass history turns if provided for this specific session
            contents: list[Any] = []
            if session_history:
                for turn in session_history:
                    role_tag = "user" if turn.role == "user" else "model"
                    contents.append(
                        types.Content(
                            role=role_tag,
                            parts=[types.Part.from_text(text=turn.text)],
                        )
                    )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=assembled_prompt)],
                )
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            response_text = (response.text or "").strip()
            return response_text

        except (LLMProviderError, ProviderTimeoutError, ProviderNetworkError):
            raise
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                message="Gemini request timed out.",
                provider_name="gemini",
                details={"model_name": self.model_name},
            ) from exc
        except ConnectionError as exc:
            raise ProviderNetworkError(
                message="Gemini network communication failed.",
                provider_name="gemini",
                details={"model_name": self.model_name},
            ) from exc
        except Exception as exc:
            logger.error(
                "ElivaGeminiAdapter: generate_content failed: %s",
                type(exc).__name__,
            )
            # Guarantee raw error messages with sensitive parameters are never leaked
            raise LLMProviderError(
                message="Gemini language model inference failed.",
                provider_name="gemini",
                details={"model_name": self.model_name},
            ) from exc

    def _assemble_prompt(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
    ) -> str:
        """Format user prompt with pre-filtered facts and language directive."""
        sections: list[str] = []
        if filtered_context and filtered_context.strip():
            sections.append(f"[LOCAL KNOWLEDGE MATCHES]:\n{filtered_context.strip()}")

        if language == "en":
            sections.append("[LANGUAGE DIRECTIVE]: Respond completely in natural English.")
        elif language == "bn":
            sections.append("[LANGUAGE DIRECTIVE]: Respond strictly in Bangla script.")

        sections.append(f"[USER QUESTION]: {prompt.strip()}")
        return "\n\n".join(sections)


# ==============================================================================
# 2. ElivaSonioxAdapter (STTProvider)
# ==============================================================================


class ElivaSonioxAdapter(STTProvider):
    """Adapter wrapping Soniox real-time WebSocket speech-to-text recognition.

    Implements `STTProvider`. Captures speech via blocking or injected listener,
    supporting English and Bangla transcription and hold/release muting.
    """

    def __init__(
        self,
        api_key: str | None = None,
        ws_url: str = "wss://stt-rt.soniox.com/transcribe-websocket",
        model: str = "stt-rt-v3",
        listen_fn: Callable[..., str] | None = None,
        hold_fn: Callable[[], None] | None = None,
        release_fn: Callable[[], None] | None = None,
        is_held_fn: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize ElivaSonioxAdapter without opening microphone or network devices."""
        self._api_key = api_key
        self.ws_url = ws_url
        self.model = model
        self._listen_fn = listen_fn
        self._hold_fn = hold_fn
        self._release_fn = release_fn
        self._is_held_fn = is_held_fn
        self._is_held: bool = False
        self._lock = threading.Lock()
        self._stt_module: Any | None = None

    def _get_api_key(self) -> str:
        key = self._api_key or os.getenv("SONIOX_API_KEY", "")
        return key.strip()

    def hold(self) -> None:
        """Mute or hold speech input processing."""
        with self._lock:
            self._is_held = True

        if self._hold_fn is not None:
            try:
                self._hold_fn()
            except Exception as exc:
                logger.warning("ElivaSonioxAdapter: hold_fn failed: %s", type(exc).__name__)
        elif self._stt_module is not None:
            try:
                self._stt_module.hold_stt()
            except Exception as exc:
                logger.warning("ElivaSonioxAdapter: hold_stt failed: %s", type(exc).__name__)

    def release(self) -> None:
        """Resume speech input processing."""
        with self._lock:
            self._is_held = False

        if self._release_fn is not None:
            try:
                self._release_fn()
            except Exception as exc:
                logger.warning("ElivaSonioxAdapter: release_fn failed: %s", type(exc).__name__)
        elif self._stt_module is not None:
            try:
                self._stt_module.release_stt()
            except Exception as exc:
                logger.warning("ElivaSonioxAdapter: release_stt failed: %s", type(exc).__name__)

    @property
    def is_held(self) -> bool:
        """Return True if speech processing is currently held/muted."""
        with self._lock:
            if self._is_held_fn is not None:
                return self._is_held_fn()
            return self._is_held

    def listen_utterance(
        self,
        language_hint: SupportedLanguage = "bn",
        on_partial_text: Callable[[str], None] | None = None,
    ) -> str:
        """Capture and transcribe a single spoken utterance.

        Args:
            language_hint: Acoustic language hint ('bn' or 'en').
            on_partial_text: Optional live callback for partial transcriptions.

        Returns:
            Transcribed text utterance.

        Raises:
            STTProviderError: If STT fails, API key is missing, or SDK is not installed.
            HardwareUnavailableError: If microphone hardware cannot be opened.
            ProviderNetworkError: If WebSocket connection fails.
            ProviderTimeoutError: If transcription times out.
        """
        # 1. Check hold state
        with self._lock:
            if self._is_held:
                return ""

        # 2. Injected listen callable (testing / custom hardware worker)
        if self._listen_fn is not None:
            try:
                return self._listen_fn(
                    language_hint=language_hint,
                    on_partial_text=on_partial_text,
                )
            except (
                STTProviderError,
                HardwareUnavailableError,
                ProviderNetworkError,
                ProviderTimeoutError,
            ):
                raise
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    message="Soniox speech recognition timed out.",
                    provider_name="soniox",
                ) from exc
            except ConnectionError as exc:
                raise ProviderNetworkError(
                    message="Soniox WebSocket connection failed.",
                    provider_name="soniox",
                ) from exc
            except Exception as exc:
                logger.error("ElivaSonioxAdapter: listener failed: %s", type(exc).__name__)
                raise STTProviderError(
                    message="Speech-to-text transcription failed.",
                    provider_name="soniox",
                ) from exc

        # 3. Check credentials and dependencies lazily
        api_key = self._get_api_key()
        if not api_key:
            raise STTProviderError(
                message="Soniox API key is not configured.",
                provider_name="soniox",
                details={"reason": "MISSING_API_KEY"},
            )

        try:
            import numpy as np  # type: ignore[import-untyped]  # noqa: F401
            import sounddevice as sd  # type: ignore[import-untyped]  # noqa: F401
            import soxr  # type: ignore[import-untyped]  # noqa: F401
            from websockets.sync.client import connect  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise STTProviderError(
                message="Required STT dependencies (sounddevice, soxr) are not installed.",
                provider_name="soniox",
                details={"reason": "SDK_NOT_INSTALLED"},
            ) from exc

        # 4. Invoke staged Eliva Soniox listener if dependencies are met
        try:
            from dh360_hi.integrations.eliva import soniox  # type: ignore[import-not-found]

            self._stt_module = soniox
            with self._lock:
                if self._is_held:
                    soniox.hold_stt()
                else:
                    soniox.release_stt()

            # Configure language hint
            soniox.LANGUAGE_HINTS = [language_hint]
            transcription = soniox.listen_once(live_callback=on_partial_text)
            return (transcription or "").strip()
        except (
            STTProviderError,
            HardwareUnavailableError,
            ProviderNetworkError,
            ProviderTimeoutError,
        ):
            raise
        except ConnectionError as exc:
            raise ProviderNetworkError(
                message="Failed to connect to Soniox transcription WebSocket.",
                provider_name="soniox",
            ) from exc
        except Exception as exc:
            logger.error("ElivaSonioxAdapter: execution error: %s", type(exc).__name__)
            # If error relates to audio hardware or ALSA
            err_str = str(exc).lower()
            if "device" in err_str or "portaudio" in err_str or "audio" in err_str:
                raise HardwareUnavailableError(
                    message="Microphone audio input device is unavailable.",
                    device_type="microphone",
                    provider_name="soniox",
                ) from exc
            raise STTProviderError(
                message="Speech-to-text recognition failed.",
                provider_name="soniox",
            ) from exc


# ==============================================================================
# 3. ElivaGoogleTTSAdapter (TTSProvider)
# ==============================================================================


class ElivaGoogleTTSAdapter(TTSProvider):
    """Adapter wrapping Google Cloud Text-to-Speech (Chirp3/HD) for DH360 audio playback.

    Implements `TTSProvider`. Performs text-to-speech synthesis and streaming playback,
    with lazy hardware initialization, interrupt capabilities, and language support.
    """

    def __init__(
        self,
        credentials_file: str | None = None,
        credentials_json: str | None = None,
        voice_bn: str = "bn-BD-Standard-A",
        voice_en: str = "en-US-Chirp3-HD-Charon",
        speak_fn: Callable[..., None] | None = None,
        interrupt_fn: Callable[[], None] | None = None,
        is_playing_fn: Callable[[], bool] | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize ElivaGoogleTTSAdapter without opening audio hardware streams."""
        self._credentials_file = credentials_file
        self._credentials_json = credentials_json
        self.voice_bn = voice_bn
        self.voice_en = voice_en
        self._speak_fn = speak_fn
        self._interrupt_fn = interrupt_fn
        self._is_playing_fn = is_playing_fn
        self._client = client
        self._is_playing: bool = False
        self._interrupted: bool = False
        self._lock = threading.Lock()

    def is_playing(self) -> bool:
        """Return True if audio playback is currently in progress."""
        with self._lock:
            if self._interrupted:
                return False
            if self._is_playing_fn is not None:
                return self._is_playing_fn()
            if self._is_playing:
                return True
        try:
            import sounddevice as sd  # type: ignore[import-untyped]

            stream = sd.get_stream()
            return bool(stream and getattr(stream, "active", False))
        except Exception:
            return False

    def interrupt(self) -> None:
        """Immediately abort active playback and discard audio buffers."""
        with self._lock:
            self._interrupted = True
            self._is_playing = False
        logger.info("ElivaGoogleTTSAdapter: interrupt signal recorded")

        if self._interrupt_fn is not None:
            try:
                self._interrupt_fn()
            except Exception as exc:
                logger.warning(
                    "ElivaGoogleTTSAdapter: interrupt_fn failed: %s",
                    type(exc).__name__,
                )

        try:
            import sounddevice as sd  # type: ignore[import-untyped]

            sd.stop()
        except Exception as exc:
            logger.debug(
                "ElivaGoogleTTSAdapter: sounddevice stop error: %s",
                type(exc).__name__,
            )

    def speak(
        self,
        text: str,
        language: SupportedLanguage = "bn",
        blocking: bool = True,
    ) -> None:
        """Synthesize and play audio for the provided text.

        Args:
            text: Text content to synthesize and speak.
            language: Target voice language code ('bn' or 'en').
            blocking: Whether to block until audio playback completes.

        Raises:
            TTSProviderError: If synthesis fails, SDK is missing, or credentials absent.
            HardwareUnavailableError: If speaker hardware or sound server is unavailable.
            ProviderNetworkError: If communication with Google Cloud fails.
            ProviderTimeoutError: If synthesis request times out.
        """
        if not text or not text.strip():
            return

        with self._lock:
            self._interrupted = False
            self._is_playing = True

        try:
            # 1. Injected speak callable (testing / custom audio player)
            if self._speak_fn is not None:
                try:
                    self._speak_fn(
                        text=text,
                        language=language,
                        blocking=blocking,
                    )
                finally:
                    with self._lock:
                        if blocking or self._interrupted:
                            self._is_playing = False
                return

            # 2. Check dependencies lazily
            try:
                import sounddevice as sd  # type: ignore[import-untyped]  # noqa: F401
                from google.cloud import texttospeech  # type: ignore[import-untyped]
                from google.oauth2 import service_account  # type: ignore[import-untyped]
            except ImportError as exc:
                raise TTSProviderError(
                    message="Optional dependencies for Google Cloud TTS are not installed.",
                    provider_name="google_tts",
                    details={"reason": "SDK_NOT_INSTALLED"},
                ) from exc

            # 3. Check credentials lazily
            client = self._client
            if client is None:
                client = self._init_google_client(texttospeech, service_account)

            # 4. Resolve voice and language code
            voice_name, language_code = self._resolve_voice(language)

            # 5. Synthesize and play
            self._synthesize_and_stream(
                client,
                texttospeech,
                text,
                voice_name,
                language_code,
                blocking=blocking,
            )

        except (
            TTSProviderError,
            HardwareUnavailableError,
            ProviderNetworkError,
            ProviderTimeoutError,
        ):
            raise
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                message="Google TTS synthesis request timed out.",
                provider_name="google_tts",
            ) from exc
        except ConnectionError as exc:
            raise ProviderNetworkError(
                message="Google Cloud TTS network connection failed.",
                provider_name="google_tts",
            ) from exc
        except Exception as exc:
            logger.error("ElivaGoogleTTSAdapter: speak failed: %s", type(exc).__name__)
            err_str = str(exc).lower()
            if "pipewire" in err_str or "device" in err_str or "sound" in err_str:
                raise HardwareUnavailableError(
                    message="Speaker audio output device is unavailable.",
                    device_type="speaker",
                    provider_name="google_tts",
                ) from exc
            raise TTSProviderError(
                message="Text-to-speech synthesis failed.",
                provider_name="google_tts",
            ) from exc
        finally:
            with self._lock:
                if blocking or self._interrupted:
                    self._is_playing = False

    def _resolve_voice(self, language: SupportedLanguage) -> tuple[str, str]:
        """Resolve voice name and BCP-47 language code."""
        if language == "en":
            return self.voice_en, "en-US"
        return self.voice_bn, "bn-BD"

    def _init_google_client(self, tts_mod: Any, sa_mod: Any) -> Any:
        """Initialize Google Cloud TextToSpeechClient using available credentials."""
        sa_json = self._credentials_json or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        sa_file = self._credentials_file or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if sa_json:
            import json

            credentials_info = json.loads(sa_json)
            creds = sa_mod.Credentials.from_service_account_info(credentials_info)
            return tts_mod.TextToSpeechClient(credentials=creds)
        elif sa_file and os.path.isfile(sa_file):
            creds = sa_mod.Credentials.from_service_account_file(sa_file)
            return tts_mod.TextToSpeechClient(credentials=creds)

        # Attempt default credentials
        try:
            return tts_mod.TextToSpeechClient()
        except Exception as exc:
            raise TTSProviderError(
                message="Google Cloud credentials are not configured.",
                provider_name="google_tts",
                details={"reason": "MISSING_CREDENTIALS"},
            ) from exc

    def _synthesize_and_stream(
        self,
        client: Any,
        tts_mod: Any,
        text: str,
        voice_name: str,
        language_code: str,
        blocking: bool = True,
    ) -> None:
        """Perform TTS synthesis and stream to sound device."""
        import sounddevice as sd  # type: ignore[import-untyped]

        synthesis_input = tts_mod.SynthesisInput(text=text)
        voice = tts_mod.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
        )
        audio_config = tts_mod.AudioConfig(
            audio_encoding=tts_mod.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        with self._lock:
            if self._interrupted:
                return

        # Play audio buffer via sounddevice
        try:
            import numpy as np  # type: ignore[import-untyped]

            audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
            sd.play(audio_data, samplerate=24000)
            if blocking:
                sd.wait()
        except Exception as exc:
            with self._lock:
                if self._interrupted:
                    return
            raise HardwareUnavailableError(
                message="Audio playback output device failed.",
                device_type="speaker",
                provider_name="google_tts",
            ) from exc
