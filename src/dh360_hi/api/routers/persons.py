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
    _operator: OperatorPrincipal = _person_read,
) -> list[dict[str, Any]]:
    """List sanitized person profiles through the application service."""
    service = _get_service(request)
    profiles = service.list_persons(status=status, consent_status=consent_status)
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
    return profile.to_dict()


@router.post("", status_code=status.HTTP_201_CREATED)
def create_person(
    payload: PersonCreateRequest,
    request: Request,
    _operator: OperatorPrincipal = _person_create,
) -> dict[str, Any]:
    """Create a profile with active status and pending consent by default."""
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
    )
    try:
        return _get_service(request).update_person(person_id, command).to_dict()
    except PersonNotFoundError as exc:
        raise EntityNotFoundError("person", person_id) from exc


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _person_delete,
) -> Response:
    """Delete a person profile and its associated biometrics."""
    try:
        deleted = _get_service(request).delete_person(person_id)
    except PersonNotFoundError as exc:
        raise EntityNotFoundError("person", person_id) from exc
    if not deleted:
        raise EntityNotFoundError("person", person_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{person_id}/biometrics", status_code=status.HTTP_204_NO_CONTENT)
def reset_person_biometrics(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _biometrics_reset,
) -> Response:
    """Remove a person's biometric records without deleting the profile."""
    try:
        _get_service(request).reset_person_biometrics(person_id)
    except PersonNotFoundError as exc:
        raise EntityNotFoundError("person", person_id) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
