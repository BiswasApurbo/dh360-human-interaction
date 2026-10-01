"""Unit tests for Application Error hierarchy and dictionary serialization."""

from __future__ import annotations

from dh360_hi.contracts.errors import (
    ApplicationError,
    ConflictAppError,
    EnrollmentAppError,
    EntityNotFoundError,
    ServiceUnavailableError,
    ValidationAppError,
)


class TestApplicationErrors:
    def test_base_application_error(self) -> None:
        """Test base ApplicationError instantiation and dictionary serialization."""
        err = ApplicationError("General error occurred", code="CUSTOM_CODE", details={"k": "v"})
        assert str(err) == "General error occurred"
        assert err.code == "CUSTOM_CODE"
        d = err.to_dict()
        assert d["error"] == "ApplicationError"
        assert d["code"] == "CUSTOM_CODE"
        assert d["details"] == {"k": "v"}

    def test_entity_not_found_error(self) -> None:
        """Test EntityNotFoundError details and formatting."""
        err = EntityNotFoundError(entity_type="Person", entity_id="p-123")
        assert "p-123" in str(err)
        assert err.code == "ENTITY_NOT_FOUND"
        assert err.entity_type == "Person"
        assert err.entity_id == "p-123"
        d = err.to_dict()
        assert d["details"]["entity_id"] == "p-123"

    def test_validation_app_error(self) -> None:
        """Test ValidationAppError details and field metadata."""
        err = ValidationAppError("Display name cannot be empty", field="display_name")
        assert err.code == "VALIDATION_ERROR"
        assert err.field == "display_name"
        d = err.to_dict()
        assert d["details"]["field"] == "display_name"

    def test_conflict_app_error(self) -> None:
        """Test ConflictAppError details and conflict key."""
        err = ConflictAppError("Active enrollment already running", conflict_key="active_session")
        assert err.code == "CONFLICT_ERROR"
        assert err.conflict_key == "active_session"
        d = err.to_dict()
        assert d["details"]["conflict_key"] == "active_session"

    def test_service_unavailable_error(self) -> None:
        """Test ServiceUnavailableError subsystem metadata."""
        err = ServiceUnavailableError(subsystem="enrollment", reason="Not initialized")
        assert err.code == "SERVICE_UNAVAILABLE"
        assert err.subsystem == "enrollment"
        d = err.to_dict()
        assert d["details"]["subsystem"] == "enrollment"

    def test_enrollment_app_error(self) -> None:
        """Test EnrollmentAppError details."""
        err = EnrollmentAppError(
            "Enrollment timeout exceeded",
            session_id="esess-1",
            reason="timeout",
        )
        assert err.code == "ENROLLMENT_ERROR"
        assert err.session_id == "esess-1"
        assert err.reason == "timeout"
        d = err.to_dict()
        assert d["details"]["session_id"] == "esess-1"
