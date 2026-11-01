"""Current interaction context and summary endpoints."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request

from dh360_hi.api.security import OperatorPrincipal, OperatorScope, require_scope
from dh360_hi.application.service import HumanInteractionService

router = APIRouter(prefix="", tags=["interaction"])
_interaction_read = Depends(require_scope(OperatorScope.INTERACTION_READ))


def _get_service(request: Request) -> HumanInteractionService:
    svc = cast(HumanInteractionService | None, getattr(request.app.state, "service", None))
    if svc is None:
        raise HTTPException(status_code=503, detail="Application service is not initialized")
    return svc


@router.get("/interaction/context")
def get_interaction_context(
    request: Request,
    _operator: OperatorPrincipal = _interaction_read,
) -> dict[str, Any]:
    """Return the current serialized interaction context snapshot."""
    service = _get_service(request)
    return service.get_interaction_context().to_dict()


@router.get("/interaction/summary/latest")
def get_latest_interaction_summary(
    request: Request,
    _operator: OperatorPrincipal = _interaction_read,
) -> dict[str, Any] | None:
    """Return the most recent completed interaction summary if one exists."""
    service = _get_service(request)
    summary = service.get_last_summary()
    return None if summary is None else summary.to_dict()
