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
