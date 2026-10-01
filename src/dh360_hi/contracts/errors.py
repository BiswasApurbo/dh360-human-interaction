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
