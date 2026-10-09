"""Operator-controlled server-side camera enrollment endpoints.

Exposes the lifecycle of a camera face-enrollment session that runs entirely on the
server: the operator opens a capture source, advances it one frame at a time, then
commits or cancels. No biometric image data is accepted over HTTP; only session
lifecycle commands are exchanged.

The HTTP request cannot select a capture source: sessions always capture through the
server-configured camera FrameSource.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from dh360_hi.api.security import (
    OperatorPrincipal,
    OperatorScope,
    require_scope,
)
from dh360_hi.application.service import HumanInteractionService
from dh360_hi.contracts.errors import (
    ConflictAppError,
    EntityNotFoundError,
)
from dh360_hi.vision.frame import WebcamFrameSource

router = APIRouter(prefix="/persons", tags=["enrollment"])
_enrollment_manage = Depends(require_scope(OperatorScope.ENROLLMENT_MANAGE))


class EnrollmentStartRequest(BaseModel):
    """Strict request body for starting a server-side enrollment session.

    The body intentionally carries no capture-source selection: HTTP clients cannot
    choose a camera, video file, or synthetic source. Synthetic and video frame
    sources remain internal test-only helpers.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["add_samples", "replace"] = "add_samples"


def _get_service(request: Request) -> HumanInteractionService:
    service = cast(
        HumanInteractionService | None,
        getattr(request.app.state, "service", None),
    )
    if service is None:
        raise HTTPException(status_code=503, detail="Application service is not initialized")
    return service


def _require_person(service: HumanInteractionService, person_id: str) -> None:
    """Raise the standard 404 contract when the target person does not exist."""
    if service.get_person(person_id) is None:
        raise EntityNotFoundError("person", person_id)


def _require_active_session(service: HumanInteractionService, person_id: str) -> None:
    """Raise the standard 404 contract when no active session matches the person."""
    progress = service.get_enrollment_progress()
    if progress is None or progress.person_id != person_id:
        raise EntityNotFoundError("enrollment_session", person_id)


@router.post(
    "/{person_id}/enrollment/sessions",
    status_code=status.HTTP_201_CREATED,
)
def start_enrollment_session(
    person_id: str,
    payload: EnrollmentStartRequest,
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> dict[str, Any]:
    """Open a server-side capture source and start an enrollment session for a person."""
    service = _get_service(request)
    _require_person(service, person_id)
    if service.get_enrollment_progress() is not None:
        raise ConflictAppError("An enrollment session is already active")
    capture = WebcamFrameSource(camera_index=service.config.camera_index)
    progress = service.start_enrollment(
        person_id=person_id,
        mode=payload.mode,
        capture_source=capture,
    )
    return progress.to_dict()


@router.get("/{person_id}/enrollment/sessions")
def get_enrollment_session(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> dict[str, Any]:
    """Return the current progress of the active enrollment session for a person."""
    service = _get_service(request)
    progress = service.get_enrollment_progress()
    if progress is None or progress.person_id != person_id:
        raise EntityNotFoundError("enrollment_session", person_id)
    return progress.to_dict()


@router.post("/{person_id}/enrollment/sessions/step")
def step_enrollment_session(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> dict[str, Any]:
    """Capture and process exactly one server-side camera frame for the active session."""
    service = _get_service(request)
    _require_active_session(service, person_id)
    progress = service.step_enrollment()
    return progress.to_dict()


@router.post("/{person_id}/enrollment/sessions/commit")
def commit_enrollment_session(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> dict[str, Any]:
    """Transactionally persist collected samples within a safe person-mutation barrier."""
    service = _get_service(request)
    _require_active_session(service, person_id)
    result = service.commit_enrollment()
    return result.to_dict()


@router.delete(
    "/enrollment/sessions/current",
    status_code=status.HTTP_200_OK,
)
def cancel_any_enrollment_session(
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> dict[str, Any]:
    """Force cancel any active enrollment session, regardless of person_id."""
    service = _get_service(request)
    progress = service.get_enrollment_progress()
    if progress is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No active enrollment session to cancel",
        )
    service.cancel_enrollment("Cancelled via generic API cleanup")
    return {"cancelled": True, "person_id": progress.person_id}


@router.post("/{person_id}/enrollment/sessions/cancel")
def cancel_enrollment_session(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> dict[str, Any]:
    """Cancel the active enrollment session without persisting any samples."""
    service = _get_service(request)
    _require_active_session(service, person_id)
    service.cancel_enrollment("Cancelled via API")
    return {"cancelled": True, "person_id": person_id}
