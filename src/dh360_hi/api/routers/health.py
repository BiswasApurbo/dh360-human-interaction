"""Health and runtime status endpoints."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from dh360_hi.api.security import (
    OperatorPrincipal,
    OperatorScope,
    _bearer_credentials,
    get_optional_operator_principal,
    require_scope,
)
from dh360_hi.application.service import HumanInteractionService

router = APIRouter(prefix="", tags=["health"])
_interaction_read = Depends(require_scope(OperatorScope.INTERACTION_READ))


def _get_service(request: Request) -> HumanInteractionService:
    svc = cast(HumanInteractionService | None, getattr(request.app.state, "service", None))
    if svc is None:
        raise HTTPException(status_code=503, detail="Application service is not initialized")
    return svc


@router.get("/health")
async def get_health(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = _bearer_credentials,
) -> dict[str, Any]:
    """Return service health snapshot.

    Public / unauthenticated callers receive high-level status without leaking
    internal host paths (e.g. database file paths), enrollment counts, or
    internal runtime diagnostics. Authenticated operators with INTERACTION_READ,
    SETTINGS_READ, or SETTINGS_MANAGE receive full subsystem details and metrics.
    """
    service = _get_service(request)
    snapshot = service.get_health_snapshot()

    operator = await get_optional_operator_principal(request, credentials)
    has_privileged_scope = operator is not None and bool(
        operator.scopes.intersection(
            {
                OperatorScope.INTERACTION_READ,
                OperatorScope.SETTINGS_READ,
                OperatorScope.SETTINGS_MANAGE,
            }
        )
    )

    if has_privileged_scope:
        return snapshot.to_dict()

    # Sanitized response for public / unauthenticated access:
    # Only expose overall status, timestamp, and high-level component status names.
    # Exclude internal file paths, thread states, identity counts, and metrics.
    sanitized_subsystems: dict[str, Any] = {}
    for name, sub in snapshot.subsystems.items():
        sanitized_subsystems[name] = {
            "name": sub.name,
            "status": sub.status.value,
            "message": sub.message,
            "details": {},
        }

    return {
        "status": snapshot.status.value,
        "timestamp_iso": snapshot.timestamp_iso,
        "subsystems": sanitized_subsystems,
        "metrics": None,
    }


@router.get("/runtime/state")
def get_runtime_state(
    request: Request,
    _operator: OperatorPrincipal = _interaction_read,
) -> str:
    """Return the runtime state enum value."""
    service = _get_service(request)
    return service.get_runtime_state().value


@router.get("/runtime/metrics")
def get_runtime_metrics(
    request: Request,
    _operator: OperatorPrincipal = _interaction_read,
) -> dict[str, Any]:
    """Return sanitized runtime metrics."""
    service = _get_service(request)
    dto = service.get_runtime_metrics()
    return dto.to_dict()
