"""Health and runtime status endpoints."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request

from dh360_hi.application.service import HumanInteractionService

router = APIRouter(prefix="", tags=["health"])


def _get_service(request: Request) -> HumanInteractionService:
    svc = cast(HumanInteractionService | None, getattr(request.app.state, "service", None))
    if svc is None:
        raise HTTPException(status_code=503, detail="Application service is not initialized")
    return svc


@router.get("/health")
def get_health(request: Request) -> dict[str, Any]:
    """Return the comprehensive service health snapshot."""
    service = _get_service(request)
    snapshot = service.get_health_snapshot()
    return snapshot.to_dict()


@router.get("/runtime/state")
def get_runtime_state(request: Request) -> str:
    """Return the runtime state enum value."""
    service = _get_service(request)
    return service.get_runtime_state().value


@router.get("/runtime/metrics")
def get_runtime_metrics(request: Request) -> dict[str, Any]:
    """Return sanitized runtime metrics."""
    service = _get_service(request)
    dto = service.get_runtime_metrics()
    return dto.to_dict()
