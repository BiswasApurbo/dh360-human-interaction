"""FastAPI router exposing the conversational dialogue coordinator."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dh360_hi.api.dependencies import get_dialogue_coordinator
from dh360_hi.api.security import (
    OperatorPrincipal,
    OperatorScope,
    require_scope,
)
from dh360_hi.contracts.dialogue import SupportedLanguage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dialogue", tags=["dialogue"])
_dialogue_interact = Depends(require_scope(OperatorScope.DIALOGUE_INTERACT))


class DialogueTurnRequest(BaseModel):
    """Validated request body for conversational dialogue turns.

    Security Boundaries:
    - Extra fields (such as client-supplied person_id, display_name, is_authorized,
      or session_id) are strictly forbidden to prevent identity spoofing or privilege
      escalation from untrusted clients.
    - Text length is restricted to prevent buffer exhaustion.
    - Language is validated strictly to supported codes ('bn', 'en').
    """

    model_config = ConfigDict(extra="forbid")

    text: StrictStr = Field(
        ...,
        min_length=1,
        max_length=2000,
        pattern=r".*\S.*",
        description="User utterance or query text.",
    )
    language: SupportedLanguage = Field(
        default="bn",
        description="Target dialogue language ('bn' for Bangla, 'en' for English).",
    )
    speak_response: bool = Field(
        default=True,
        description="Whether to speak the response via TTS audio.",
    )


class DialogueVoiceTurnRequest(BaseModel):
    """Validated request body for voice-activated dialogue turns.

    Security Boundaries:
    - Extra fields (such as client-supplied person_id, display_name, is_authorized,
      or session_id) are strictly forbidden to prevent identity spoofing or privilege
      escalation from untrusted clients.
    - Language is validated strictly to supported codes ('bn', 'en').
    """

    model_config = ConfigDict(extra="forbid")

    language: SupportedLanguage = Field(
        default="bn",
        description="Target dialogue language ('bn' for Bangla, 'en' for English).",
    )
    speak_response: bool = Field(
        default=True,
        description="Whether to speak the response via TTS audio.",
    )


class DialogueTurnResponse(BaseModel):
    """Sanitized response representation of a completed dialogue turn."""

    role: str = Field(description="Turn participant role ('user', 'model', or 'system').")
    text: str = Field(description="Plaintext response utterance.")
    timestamp: float = Field(description="Epoch seconds when the turn was processed.")
    language: str = Field(description="Language associated with this turn ('bn' or 'en').")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Sanitized non-sensitive turn metadata.",
    )


class DialogueVoiceTurnResponse(BaseModel):
    """Sanitized response representation of a completed voice dialogue turn."""

    user_text: str = Field(description="Transcribed user question utterance.")
    role: str = Field(description="Turn participant role ('user', 'model', or 'system').")
    text: str = Field(description="Plaintext response utterance.")
    timestamp: float = Field(description="Epoch seconds when the turn was processed.")
    language: str = Field(description="Language associated with this turn ('bn' or 'en').")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Sanitized non-sensitive turn metadata.",
    )


class DialogueTranscribeRequest(BaseModel):
    """Validated request body for transcribing client-recorded audio (e.g. from browser mic)."""

    model_config = ConfigDict(extra="forbid")

    audio_base64: StrictStr = Field(
        ...,
        min_length=1,
        description="Base64-encoded audio data.",
    )
    mime_type: StrictStr = Field(
        default="audio/webm",
        description="MIME type of the audio stream.",
    )
    language: SupportedLanguage = Field(
        default="bn",
        description="Target dialogue language hint ('bn' or 'en').",
    )


class DialogueTranscribeResponse(BaseModel):
    """Sanitized response representation of transcribed audio."""

    text: str = Field(description="Transcribed plain text utterance.")
    language: str = Field(description="Language of transcription.")


class DialogueBrowserVoiceStreamRequest(BaseModel):
    """Payload for single-shot browser voice audio streaming."""

    model_config = ConfigDict(extra="forbid")

    audio_base64: StrictStr = Field(
        ...,
        min_length=1,
        description="Base64-encoded audio payload from browser microphone.",
    )
    mime_type: StrictStr = Field(
        default="audio/webm",
        description="MIME type of recorded audio data.",
    )
    language: SupportedLanguage = Field(
        default="bn",
        description="Requested interaction language ('bn' or 'en').",
    )
    speak_response: bool = Field(
        default=False,
        description="Whether the edge robot should speak the answer via TTS.",
    )


_cached_transcribe_client: Any | None = None
_cached_transcribe_key: str | None = None


def get_transcribe_client(api_key: str) -> Any:
    """Retrieve or initialize the cached GenAI client for audio transcription."""
    global _cached_transcribe_client, _cached_transcribe_key
    if _cached_transcribe_client is not None and _cached_transcribe_key == api_key:
        return _cached_transcribe_client
    from google import genai

    _cached_transcribe_client = genai.Client(api_key=api_key)
    _cached_transcribe_key = api_key
    return _cached_transcribe_client


def transcribe_audio_payload(
    audio_bytes: bytes,
    mime_type: str,
    language_hint: str = "bn",
) -> str:
    """Transcribe audio bytes using Gemini multimodal STT or fallback, returning plain text."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        env_paths = [
            "/home/eliva/.config/dh360/backend.env",
            os.path.expanduser("~/.config/dh360/backend.env"),
        ]
        for p in env_paths:
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        for line in f:
                            if line.startswith("GEMINI_API_KEY="):
                                gemini_key = line.strip().split("=", 1)[1].strip("\"'")
                                break
                except Exception:
                    pass
            if gemini_key:
                break

    if gemini_key:
        try:
            from google.genai import types

            client = get_transcribe_client(gemini_key)
            lang_directive = "Bangla" if language_hint == "bn" else "English"
            prompt = (
                f"Transcribe the spoken audio verbatim in {lang_directive} or English. "
                "Return ONLY the plain transcribed words. "
                "Do NOT include explanations, timestamps, brackets, quotes, punctuation commentary, or markdown formatting. "
                "If the audio contains silence, background noise, or no intelligible words, return an empty string."
            )
            clean_mime = mime_type.split(";")[0].strip() or "audio/webm"
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime),
                    prompt,
                ],
            )
            text = (response.text or "").strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1].strip()
            return text
        except Exception as exc:
            logger.warning("Gemini audio transcription failed: %s", exc)

    return ""


@router.post(
    "/turn",
    response_model=DialogueTurnResponse,
    status_code=status.HTTP_200_OK,
)
async def handle_dialogue_turn(
    payload: DialogueTurnRequest,
    request: Request,
    _operator: OperatorPrincipal = _dialogue_interact,
) -> DialogueTurnResponse:
    """Execute a single conversational dialogue turn through DialogueCoordinator.

    Authenticates via bearer token requiring the `dialogue:interact` operator scope.
    Delegates to the shared application DialogueCoordinator running asynchronously
    without blocking the FastAPI event loop.
    """
    coordinator = get_dialogue_coordinator(request)
    tenant_id = request.headers.get("X-Tenant-ID")

    try:
        turn = await coordinator.handle_turn_async(
            user_text=payload.text,
            language=payload.language,
            speak_response=payload.speak_response,
            tenant_id=tenant_id,
        )
    except TypeError:
        try:
            turn = await coordinator.handle_turn_async(
                user_text=payload.text,
                language=payload.language,
                speak_response=payload.speak_response,
            )
        except TypeError:
            turn = await coordinator.handle_turn_async(
                user_text=payload.text,
                language=payload.language,
            )

    return DialogueTurnResponse(
        role=turn.role,
        text=turn.text,
        timestamp=turn.timestamp,
        language=turn.language,
        metadata=turn.metadata,
    )


@router.post(
    "/turn/stream",
    status_code=status.HTTP_200_OK,
)
async def handle_dialogue_turn_stream(
    payload: DialogueTurnRequest,
    request: Request,
    _operator: OperatorPrincipal = _dialogue_interact,
) -> StreamingResponse:
    """Stream conversational response tokens incrementally via Server-Sent Events (SSE)."""
    coordinator = get_dialogue_coordinator(request)
    tenant_id = request.headers.get("X-Tenant-ID")

    async def sse_event_generator():
        try:
            async for evt in coordinator.handle_turn_stream_async(
                user_text=payload.text,
                language=payload.language,
                speak_response=payload.speak_response,
                tenant_id=tenant_id,
            ):
                event_type = evt.get("event", "message")
                event_data = json.dumps(evt.get("data", {}), ensure_ascii=False)
                yield f"event: {event_type}\ndata: {event_data}\n\n"
        except Exception as exc:
            logger.exception("Error during turn streaming: %s", exc)
            err_data = json.dumps({"error": "STREAM_ERROR", "message": str(exc)})
            yield f"event: error\ndata: {err_data}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/turn/browser-voice-stream",
    status_code=status.HTTP_200_OK,
)
async def handle_dialogue_browser_voice_stream(
    payload: DialogueBrowserVoiceStreamRequest,
    request: Request,
    _operator: OperatorPrincipal = _dialogue_interact,
) -> StreamingResponse:
    """Transcribe browser voice audio and stream model response tokens in a single request."""
    coordinator = get_dialogue_coordinator(request)
    tenant_id = request.headers.get("X-Tenant-ID")

    try:
        audio_bytes = base64.b64decode(payload.audio_base64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 audio payload: {exc}",
        ) from exc

    async def sse_voice_generator():
        # 1. Transcribe audio payload
        try:
            recognized_text = await asyncio.to_thread(
                transcribe_audio_payload,
                audio_bytes,
                payload.mime_type,
                payload.language,
            )
        except Exception as exc:
            logger.warning("Browser voice transcription failed: %s", exc)
            recognized_text = ""

        # Emit user_transcribed event immediately so UI displays what user said
        yield f"event: user_transcribed\ndata: {json.dumps({'text': recognized_text}, ensure_ascii=False)}\n\n"

        if not recognized_text:
            err_msg = (
                "কোনো কথা শনাক্ত করা যায়নি। দয়া করে মাইক্রোফোনের কাছে এসে আবার বলুন।"
                if payload.language == "bn"
                else "No intelligible speech detected. Please speak closer to your microphone."
            )
            yield f"event: error\ndata: {json.dumps({'error': 'NO_SPEECH_DETECTED', 'message': err_msg}, ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {json.dumps({'role': 'model', 'text': err_msg, 'language': payload.language, 'timestamp': time.time(), 'metadata': {'error': 'NO_SPEECH_DETECTED'}}, ensure_ascii=False)}\n\n"
            return

        # 2. Stream conversational dialogue tokens
        try:
            async for evt in coordinator.handle_turn_stream_async(
                user_text=recognized_text,
                language=payload.language,
                speak_response=payload.speak_response,
                tenant_id=tenant_id,
            ):
                event_type = evt.get("event", "message")
                event_data = json.dumps(evt.get("data", {}), ensure_ascii=False)
                yield f"event: {event_type}\ndata: {event_data}\n\n"
        except Exception as exc:
            logger.exception("Error during voice turn streaming: %s", exc)
            err_data = json.dumps({"error": "STREAM_ERROR", "message": str(exc)})
            yield f"event: error\ndata: {err_data}\n\n"

    return StreamingResponse(
        sse_voice_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/turn/voice",
    response_model=DialogueVoiceTurnResponse,
    status_code=status.HTTP_200_OK,
)
async def handle_dialogue_voice_turn(
    payload: DialogueVoiceTurnRequest,
    request: Request,
    _operator: OperatorPrincipal = _dialogue_interact,
) -> DialogueVoiceTurnResponse:
    """Execute a voice-activated dialogue turn through DialogueCoordinator.

    Captures microphone utterance via the configured STTProvider (Soniox),
    resolves context and generates response via LLMProvider (Gemini),
    and speaks the answer via TTSProvider (Google Cloud TTS).
    """
    coordinator = get_dialogue_coordinator(request)
    tenant_id = request.headers.get("X-Tenant-ID")

    try:
        turn = await coordinator.handle_turn_async(
            user_text=None,
            language=payload.language,
            speak_response=payload.speak_response,
            tenant_id=tenant_id,
        )
    except TypeError:
        try:
            turn = await coordinator.handle_turn_async(
                user_text=None,
                language=payload.language,
                speak_response=payload.speak_response,
            )
        except TypeError:
            turn = await coordinator.handle_turn_async(
                user_text=None,
                language=payload.language,
            )

    user_text = str(turn.metadata.get("user_text", "")).strip()

    return DialogueVoiceTurnResponse(
        user_text=user_text,
        role=turn.role,
        text=turn.text,
        timestamp=turn.timestamp,
        language=turn.language,
        metadata=turn.metadata,
    )


@router.post(
    "/interrupt",
    status_code=status.HTTP_200_OK,
)
async def interrupt_dialogue(
    request: Request,
    _operator: OperatorPrincipal = _dialogue_interact,
) -> dict[str, str]:
    """Abort in-flight speech synthesis playback or microphone listening immediately."""
    coordinator = get_dialogue_coordinator(request)
    coordinator.interrupt()
    return {"status": "interrupted"}


@router.post(
    "/transcribe",
    response_model=DialogueTranscribeResponse,
    status_code=status.HTTP_200_OK,
)
async def handle_dialogue_transcribe(
    payload: DialogueTranscribeRequest,
    request: Request,
    _operator: OperatorPrincipal = _dialogue_interact,
) -> DialogueTranscribeResponse:
    """Transcribe uploaded client audio (e.g. from browser mic) using backend STT.

    Authenticates via bearer token requiring the `dialogue:interact` operator scope.
    """
    try:
        audio_bytes = base64.b64decode(payload.audio_base64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 audio data: {exc}",
        ) from exc

    if len(audio_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio payload exceeds maximum 10MB limit.",
        )

    clean_mime = payload.mime_type.split(";")[0].strip() or "audio/webm"
    transcription = transcribe_audio_payload(
        audio_bytes=audio_bytes,
        mime_type=clean_mime,
        language_hint=payload.language,
    )

    return DialogueTranscribeResponse(
        text=transcription,
        language=payload.language,
    )
