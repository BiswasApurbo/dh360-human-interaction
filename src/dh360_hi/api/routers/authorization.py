"""Authorization session lifecycle and decision retrieval endpoints."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from dh360_hi.api.security import OperatorPrincipal, OperatorScope, require_scope
from dh360_hi.edge.authorization_controller import AuthorizationController

router = APIRouter(prefix="", tags=["authorization"])
_authorization_manage = Depends(require_scope(OperatorScope.AUTHORIZATION_MANAGE))


def _get_controller(request: Request) -> AuthorizationController:
    """Retrieve the shared AuthorizationController from app state or raise 503."""
    controller = cast(
        AuthorizationController | None,
        getattr(request.app.state, "authorization_controller", None),
    )
    if controller is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authorization controller is not enabled or initialized",
        )
    return controller


@router.post(
    "/authorization/sessions",
    status_code=status.HTTP_201_CREATED,
)
def start_authorization_session(
    request: Request,
    _operator: OperatorPrincipal = _authorization_manage,
) -> dict[str, Any]:
    """Start an active authorization session on the live controller.

    Returns HTTP 201 with the sanitized AuthorizationSession snapshot.
    Returns HTTP 409 if an active session already exists.
    Returns HTTP 503 if the authorization controller is unavailable.
    """
    controller = _get_controller(request)
    try:
        session = controller.start()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return session.to_dict()


@router.get(
    "/authorization/sessions/current",
    status_code=status.HTTP_200_OK,
)
def get_current_authorization_session(
    request: Request,
    _operator: OperatorPrincipal = _authorization_manage,
) -> dict[str, Any] | None:
    """Retrieve the current authorization session snapshot or null.

    Returns HTTP 200 with the sanitized AuthorizationSession snapshot or null.
    Returns HTTP 503 if the authorization controller is unavailable.
    """
    controller = _get_controller(request)
    session = controller.get_current()
    return session.to_dict() if session is not None else None


@router.delete(
    "/authorization/sessions/current",
    status_code=status.HTTP_200_OK,
)
def cancel_current_authorization_session(
    request: Request,
    _operator: OperatorPrincipal = _authorization_manage,
) -> dict[str, Any]:
    """Cancel the currently active authorization session.

    Returns HTTP 200 with the cancelled AuthorizationSession snapshot.
    Returns HTTP 409 if no session is active.
    Returns HTTP 503 if the authorization controller is unavailable.
    """
    controller = _get_controller(request)
    try:
        session = controller.cancel()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return session.to_dict()


@router.get(
    "/authorization/sessions/{authorization_id}/context",
    status_code=status.HTTP_200_OK,
)
def get_authorized_session_context(
    authorization_id: str,
    request: Request,
    _operator: OperatorPrincipal = _authorization_manage,
) -> dict[str, Any]:
    """Retrieve privileged person context for an AUTHORIZED session.

    Returns HTTP 200 with AuthorizedPersonContextDTO when AUTHORIZED.
    Returns HTTP 403 when session is in any other status.
    Returns HTTP 404 when session is missing or ID does not match.
    """
    controller = _get_controller(request)
    session = controller.get_current()
    if session is None or session.authorization_id != authorization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Authorization session '{authorization_id}' not found",
        )

    if session.status != "authorized" or not session.person_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Session is in state '{session.status.value}', "
                "privileged context is restricted to authorized sessions"
            ),
        )

    service = getattr(request.app.state, "service", None)
    if service is None or not hasattr(service, "person_service"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application service is not initialized",
        )

    context = service.person_service.get_person_context(session.person_id)
    if context is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person context for '{session.person_id}' not found",
        )

    return {
        "person_id": context.person_id,
        "display_name": context.display_name,
        "preferred_name": context.preferred_name,
        "role": context.role,
        "department": context.department,
        "office_room": context.office_room,
        "working_hours": context.working_hours,
        "profile_description": context.profile_description,
        "qa_pairs": [{"question": p.question, "answer": p.answer} for p in context.qa_pairs],
    }
