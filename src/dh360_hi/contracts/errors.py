"""Transport-neutral application errors and exception taxonomy.

All application exceptions inherit from `ApplicationError` and provide structured
error codes and contextual metadata suitable for transport mapping (e.g. HTTP status
codes or JSON-RPC error responses).
"""

from __future__ import annotations

from typing import Any


class ApplicationError(Exception):
    """Base exception for all application-level errors."""

    def __init__(
        self,
        message: str,
        code: str = "APPLICATION_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert error to a structured dictionary."""
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class EntityNotFoundError(ApplicationError):
    """Raised when a requested domain entity (person, session, batch) is not found."""

    def __init__(
        self,
        entity_type: str,
        entity_id: str,
        message: str | None = None,
    ) -> None:
        msg = message or f"{entity_type} with ID '{entity_id}' not found."
        super().__init__(
            message=msg,
            code="ENTITY_NOT_FOUND",
            details={"entity_type": entity_type, "entity_id": entity_id},
        )
        self.entity_type = entity_type
        self.entity_id = entity_id


class ValidationAppError(ApplicationError):
    """Raised when an incoming command, filter, or payload fails validation."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        d = dict(details or {})
        if field:
            d["field"] = field
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details=d,
        )
        self.field = field


class ConflictAppError(ApplicationError):
    """Raised when a command conflicts with existing state.

    Examples include duplicate resources or concurrent workflows.
    """

    def __init__(
        self,
        message: str,
        conflict_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        d = dict(details or {})
        if conflict_key:
            d["conflict_key"] = conflict_key
        super().__init__(
            message=message,
            code="CONFLICT_ERROR",
            details=d,
        )
        self.conflict_key = conflict_key


class ServiceUnavailableError(ApplicationError):
    """Raised when a requested subsystem or dependency is unconfigured, stopped, or unavailable."""

    def __init__(
        self,
        subsystem: str,
        reason: str | None = None,
    ) -> None:
        sub_reason = reason or "Not initialized or unconfigured"
        msg = f"Subsystem '{subsystem}' is unavailable: {sub_reason}"
        super().__init__(
            message=msg,
            code="SERVICE_UNAVAILABLE",
            details={"subsystem": subsystem, "reason": reason},
        )
        self.subsystem = subsystem
        self.reason = reason


class EnrollmentAppError(ApplicationError):
    """Raised when a face enrollment operation fails or is rejected."""

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        d = dict(details or {})
        if session_id:
            d["session_id"] = session_id
        if reason:
            d["reason"] = reason
        super().__init__(
            message=message,
            code="ENROLLMENT_ERROR",
            details=d,
        )
        self.session_id = session_id
        self.reason = reason


SAFE_PROVIDER_MESSAGES: dict[str, str] = {
    "PROVIDER_ERROR": "Dialogue or speech provider operation failed.",
    "LLM_PROVIDER_ERROR": "Language model provider failed to generate response.",
    "STT_PROVIDER_ERROR": "Speech-to-text recognition provider failed.",
    "TTS_PROVIDER_ERROR": "Text-to-speech synthesis provider failed.",
    "PROVIDER_TIMEOUT": "Provider operation timed out.",
    "PROVIDER_NETWORK_ERROR": "Provider network communication failed.",
    "HARDWARE_UNAVAILABLE": "Required audio hardware device is unavailable.",
}

SAFE_DIAGNOSTIC_KEYS: frozenset[str] = frozenset(
    {
        "provider",
        "timeout_seconds",
        "device_type",
        "status_code",
        "is_retryable",
        "retry_after_seconds",
        "model_name",
        "audio_sample_rate",
        "audio_channels",
        "tokens",
    }
)


class ProviderError(ApplicationError):
    """Base exception for all speech, dialogue, and generative model provider failures.

    Security & Redaction Invariants:
    1. Third-party raw exception messages, tokens, credentials, prompts, confidential
       context, or sensitive transcripts are never exposed in serialized output.
    2. Serialization produces standardized safe error codes and canonical error messages.
    3. Contextual details are strictly restricted to an allowlist of non-sensitive
       diagnostic keys (SAFE_DIAGNOSTIC_KEYS).
    """

    def __init__(
        self,
        message: str,
        provider_name: str,
        code: str = "PROVIDER_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        d = dict(details or {})
        d["provider"] = provider_name
        super().__init__(
            message=message,
            code=code,
            details=d,
        )
        self.provider_name = provider_name

    @property
    def safe_message(self) -> str:
        """Return standardized, non-sensitive error message safe for serialization."""
        return SAFE_PROVIDER_MESSAGES.get(self.code, "Provider operation failed.")

    def to_dict(self) -> dict[str, Any]:
        """Convert error to a structured dictionary guaranteed safe for serialization.

        Guarantees that raw third-party exception strings, prompts, credentials,
        and sensitive transcripts are never exposed in serialized error output.
        """
        safe_details: dict[str, Any] = {}
        for key, val in self.details.items():
            if key in SAFE_DIAGNOSTIC_KEYS:
                # Disallow non-primitive or complex structures
                if isinstance(val, (int, float, bool)):
                    safe_details[key] = val
                elif isinstance(val, str):
                    # Only allow short, safe strings (e.g. identifier names);
                    # exclude secrets, credentials, and newlines.
                    lower_val = val.lower()
                    if (
                        len(val) <= 64
                        and "\n" not in val
                        and not any(
                            sub in lower_val
                            for sub in (
                                "key",
                                "token",
                                "secret",
                                "bearer",
                                "private",
                                "credential",
                            )
                        )
                    ):
                        safe_details[key] = val

        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.safe_message,
            "details": safe_details,
        }


class LLMProviderError(ProviderError):
    """Raised when generative conversational model inference fails."""

    def __init__(
        self,
        message: str,
        provider_name: str = "llm",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            provider_name=provider_name,
            code="LLM_PROVIDER_ERROR",
            details=details,
        )


class STTProviderError(ProviderError):
    """Raised when speech-to-text recognition or audio ingestion fails."""

    def __init__(
        self,
        message: str,
        provider_name: str = "stt",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            provider_name=provider_name,
            code="STT_PROVIDER_ERROR",
            details=details,
        )


class TTSProviderError(ProviderError):
    """Raised when text-to-speech audio synthesis or playback fails."""

    def __init__(
        self,
        message: str,
        provider_name: str = "tts",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            provider_name=provider_name,
            code="TTS_PROVIDER_ERROR",
            details=details,
        )


class ProviderTimeoutError(ProviderError):
    """Raised when a remote speech or language model provider times out."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        timeout_seconds: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        d = dict(details or {})
        if timeout_seconds is not None:
            d["timeout_seconds"] = timeout_seconds
        super().__init__(
            message=message,
            provider_name=provider_name,
            code="PROVIDER_TIMEOUT",
            details=d,
        )
        self.timeout_seconds = timeout_seconds


class ProviderNetworkError(ProviderError):
    """Raised when communication with a remote speech or language API fails."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            provider_name=provider_name,
            code="PROVIDER_NETWORK_ERROR",
            details=details,
        )


class HardwareUnavailableError(ProviderError):
    """Raised when required audio hardware (microphone, speaker, sound server) is unavailable."""

    def __init__(
        self,
        message: str,
        device_type: str,
        provider_name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        d = dict(details or {})
        d["device_type"] = device_type
        super().__init__(
            message=message,
            provider_name=provider_name,
            code="HARDWARE_UNAVAILABLE",
            details=d,
        )
        self.device_type = device_type
