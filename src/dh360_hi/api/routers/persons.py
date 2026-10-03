"""Authenticated, sanitized operator access to person profiles."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dh360_hi.api.security import OperatorPrincipal, OperatorScope, require_scope
from dh360_hi.application.service import HumanInteractionService
from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.contracts.dto import CreatePersonCommand
from dh360_hi.contracts.errors import EntityNotFoundError

router = APIRouter(prefix="/persons", tags=["persons"])
_person_read = Depends(require_scope(OperatorScope.PERSONS_READ))
_person_create = Depends(require_scope(OperatorScope.PERSONS_CREATE))


class PersonCreateRequest(BaseModel):
    """Strict request body for creating a profile without accepting consent or notes."""

    model_config = ConfigDict(extra="forbid")

    display_name: StrictStr = Field(min_length=1)
    preferred_name: StrictStr | None = None


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
    )
    return _get_service(request).create_person(command).to_dict()
