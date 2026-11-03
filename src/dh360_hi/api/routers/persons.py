"""Authenticated, sanitized operator access to person profiles."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dh360_hi.api.security import (
    OperatorPrincipal,
    OperatorScope,
    OperatorSecurityError,
    require_scope,
)
from dh360_hi.application.service import HumanInteractionService
from dh360_hi.backend.exceptions import PersonNotFoundError
from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.contracts.dto import CreatePersonCommand, UpdatePersonCommand
from dh360_hi.contracts.errors import EntityNotFoundError, ValidationAppError

router = APIRouter(prefix="/persons", tags=["persons"])
_person_read = Depends(require_scope(OperatorScope.PERSONS_READ))
_person_create = Depends(require_scope(OperatorScope.PERSONS_CREATE))
_person_update = Depends(require_scope(OperatorScope.PERSONS_UPDATE))
_person_delete = Depends(require_scope(OperatorScope.PERSONS_DELETE))
_biometrics_reset = Depends(require_scope(OperatorScope.BIOMETRICS_RESET))
_consent_manage = Depends(require_scope(OperatorScope.CONSENT_MANAGE))


class QAPairRequest(BaseModel):
    """Question-and-answer pair schema for requests."""

    model_config = ConfigDict(extra="forbid")

    question: StrictStr = Field(min_length=1)
    answer: StrictStr = Field(min_length=1)


class PersonCreateRequest(BaseModel):
    """Strict request body for creating a profile with optional context."""

    model_config = ConfigDict(extra="forbid")

    display_name: StrictStr = Field(min_length=1)
    preferred_name: StrictStr | None = None
    role: StrictStr | None = None
    department: StrictStr | None = None
    office_room: StrictStr | None = None
    working_hours: StrictStr | None = None
    profile_description: StrictStr | None = None
    qa_pairs: list[QAPairRequest] | None = None
    tenant_id: StrictStr | None = None
    robot_id: StrictStr | None = None


class PersonUpdateRequest(BaseModel):
    """Strict profile update payload with optional context."""

    model_config = ConfigDict(extra="forbid")

    display_name: StrictStr | None = Field(default=None, min_length=1)
    preferred_name: StrictStr | None = Field(default=None, min_length=1)
    status: PersonStatus | None = None
    consent_status: ConsentStatus | None = None
    role: StrictStr | None = None
    department: StrictStr | None = None
    office_room: StrictStr | None = None
    working_hours: StrictStr | None = None
    profile_description: StrictStr | None = None
    qa_pairs: list[QAPairRequest] | None = None
    tenant_id: StrictStr | None = None
    robot_id: StrictStr | None = None


class PersonConsentUpdateRequest(BaseModel):
    """Strict request body for updating person consent."""

    model_config = ConfigDict(extra="forbid")

    consent_status: ConsentStatus
    notes: StrictStr | None = None


def _get_service(request: Request) -> HumanInteractionService:
    service = cast(
        HumanInteractionService | None,
        getattr(request.app.state, "service", None),
    )
    if service is None:
        raise HTTPException(status_code=503, detail="Application service is not initialized")
    return service


@router.get("")
def list_persons(
    request: Request,
    status: PersonStatus | None = None,
    consent_status: ConsentStatus | None = None,
    robot_id: str | None = None,
    _operator: OperatorPrincipal = _person_read,
) -> list[dict[str, Any]]:
    """List sanitized person profiles through the application service."""
    service = _get_service(request)
    tenant_id = request.headers.get("X-Tenant-ID")
    effective_robot_id = request.headers.get("X-Robot-ID") or robot_id
    profiles = service.list_persons(
        status=status,
        consent_status=consent_status,
        tenant_id=tenant_id,
        robot_id=effective_robot_id,
    )
    return [profile.to_dict() for profile in profiles]


@router.get("/{person_id}")
def get_person(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _person_read,
) -> dict[str, Any]:
    """Retrieve a sanitized person profile or return the existing not-found contract."""
    service = _get_service(request)
    profile = service.get_person(person_id)
    if profile is None:
        raise EntityNotFoundError("person", person_id)

    tenant_id = request.headers.get("X-Tenant-ID")
    if tenant_id and profile.tenant_id != tenant_id:
        raise EntityNotFoundError("person", person_id)

    return profile.to_dict()


@router.post("", status_code=status.HTTP_201_CREATED)
def create_person(
    payload: PersonCreateRequest,
    request: Request,
    _operator: OperatorPrincipal = _person_create,
) -> dict[str, Any]:
    """Create a profile with active status and pending consent by default."""
    tenant_id = request.headers.get("X-Tenant-ID") or payload.tenant_id
    robot_id = request.headers.get("X-Robot-ID") or payload.robot_id

    command = CreatePersonCommand(
        display_name=payload.display_name,
        preferred_name=payload.preferred_name,
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.PENDING,
        notes=None,
        person_id=None,
        role=payload.role,
        department=payload.department,
        office_room=payload.office_room,
        working_hours=payload.working_hours,
        profile_description=payload.profile_description,
        qa_pairs=(
            [qa.model_dump() for qa in payload.qa_pairs] if payload.qa_pairs is not None else None
        ),
        tenant_id=tenant_id,
        robot_id=robot_id,
    )
    return _get_service(request).create_person(command).to_dict()


@router.patch("/{person_id}")
def update_person(
    person_id: str,
    payload: PersonUpdateRequest,
    request: Request,
    operator: OperatorPrincipal = _person_update,
) -> dict[str, Any]:
    """Update profile fields through the existing application service."""
    service = _get_service(request)
    existing = service.get_person(person_id)
    if existing is None:
        raise EntityNotFoundError("person", person_id)

    tenant_id = request.headers.get("X-Tenant-ID")
    if tenant_id and existing.tenant_id != tenant_id:
        raise EntityNotFoundError("person", person_id)

    if all(
        value is None
        for value in (
            payload.display_name,
            payload.preferred_name,
            payload.status,
            payload.consent_status,
            payload.role,
            payload.department,
            payload.office_room,
            payload.working_hours,
            payload.profile_description,
            payload.qa_pairs,
            payload.tenant_id,
            payload.robot_id,
        )
    ):
        raise ValidationAppError("At least one profile field must be provided", field="body")

    if payload.display_name is not None and not payload.display_name.strip():
        raise ValidationAppError(
            "display_name cannot be empty or whitespace",
            field="display_name",
        )
    if payload.preferred_name is not None and not payload.preferred_name.strip():
        raise ValidationAppError(
            "preferred_name cannot be empty or whitespace",
            field="preferred_name",
        )

    if payload.consent_status is not None and OperatorScope.CONSENT_MANAGE not in operator.scopes:
        raise OperatorSecurityError(
            403,
            "INSUFFICIENT_SCOPE",
            "The operator credential does not grant the required scope",
        )
    command = UpdatePersonCommand(
        display_name=payload.display_name,
        preferred_name=payload.preferred_name,
        status=payload.status,
        consent_status=payload.consent_status,
        notes=None,
        role=payload.role,
        department=payload.department,
        office_room=payload.office_room,
        working_hours=payload.working_hours,
        profile_description=payload.profile_description,
        qa_pairs=(
            [qa.model_dump() for qa in payload.qa_pairs] if payload.qa_pairs is not None else None
        ),
        tenant_id=payload.tenant_id,
        robot_id=payload.robot_id,
    )
    try:
        return service.update_person(person_id, command).to_dict()
    except PersonNotFoundError as exc:
        raise EntityNotFoundError("person", person_id) from exc


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _person_delete,
) -> Response:
    """Delete a person profile and its associated biometrics."""
    service = _get_service(request)
    existing = service.get_person(person_id)
    if existing is None:
        raise EntityNotFoundError("person", person_id)

    tenant_id = request.headers.get("X-Tenant-ID")
    if tenant_id and existing.tenant_id != tenant_id:
        raise EntityNotFoundError("person", person_id)

    try:
        deleted = service.delete_person(person_id)
    except PersonNotFoundError as exc:
        raise EntityNotFoundError("person", person_id) from exc
    if not deleted:
        raise EntityNotFoundError("person", person_id)
    auth_ctrl = getattr(request.app.state, "authorization_controller", None)
    if auth_ctrl is not None and hasattr(auth_ctrl, "invalidate_person"):
        auth_ctrl.invalidate_person(person_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{person_id}/biometrics", status_code=status.HTTP_204_NO_CONTENT)
def reset_person_biometrics(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _biometrics_reset,
) -> Response:
    """Remove a person's biometric records without deleting the profile."""
    service = _get_service(request)
    existing = service.get_person(person_id)
    if existing is None:
        raise EntityNotFoundError("person", person_id)

    tenant_id = request.headers.get("X-Tenant-ID")
    if tenant_id and existing.tenant_id != tenant_id:
        raise EntityNotFoundError("person", person_id)

    try:
        service.reset_person_biometrics(person_id)
    except PersonNotFoundError as exc:
        raise EntityNotFoundError("person", person_id) from exc
    auth_ctrl = getattr(request.app.state, "authorization_controller", None)
    if auth_ctrl is not None and hasattr(auth_ctrl, "invalidate_person"):
        auth_ctrl.invalidate_person(person_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{person_id}/consent")
def update_person_consent(
    person_id: str,
    payload: PersonConsentUpdateRequest,
    request: Request,
    _operator: OperatorPrincipal = _consent_manage,
) -> dict[str, Any]:
    """Explicitly record or update a person's consent with an immutable audit entry."""
    service = _get_service(request)
    existing = service.get_person(person_id)
    if existing is None:
        raise EntityNotFoundError("person", person_id)

    tenant_id = request.headers.get("X-Tenant-ID")
    if tenant_id and existing.tenant_id != tenant_id:
        raise EntityNotFoundError("person", person_id)

    changed_by_uid = request.headers.get("X-User-ID") or "operator"
    changed_by_name = request.headers.get("X-User-Name") or request.headers.get("X-User-Email")

    profile, audit_entry = service.update_person_consent(
        person_id=person_id,
        new_status=payload.consent_status,
        changed_by_uid=changed_by_uid,
        changed_by_name=changed_by_name,
        notes=payload.notes,
        tenant_id=tenant_id,
    )
    return {
        "person": profile.to_dict(),
        "audit": audit_entry,
    }


@router.get("/{person_id}/consent-history")
def get_person_consent_history(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _person_read,
) -> list[dict[str, Any]]:
    """Retrieve the chronological immutable consent audit history for a person."""
    service = _get_service(request)
    existing = service.get_person(person_id)
    if existing is None:
        raise EntityNotFoundError("person", person_id)

    tenant_id = request.headers.get("X-Tenant-ID")
    if tenant_id and existing.tenant_id != tenant_id:
        raise EntityNotFoundError("person", person_id)

    return service.get_consent_audit_history(person_id, tenant_id=tenant_id)
