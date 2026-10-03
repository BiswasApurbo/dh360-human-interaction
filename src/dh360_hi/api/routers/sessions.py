"""Session history and analytics endpoints."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Query, Request

from dh360_hi.application.service import HumanInteractionService
from dh360_hi.backend.models import SessionQueryFilter
from dh360_hi.contracts.errors import EntityNotFoundError

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _get_service(request: Request) -> HumanInteractionService:
    svc = cast(HumanInteractionService | None, getattr(request.app.state, "service", None))
    if svc is None:
        raise HTTPException(status_code=503, detail="Application service is not initialized")
    return svc


@router.get("")
def list_sessions(
    request: Request,
    person_id: str | None = None,
    limit: int = Query(50, gt=0),
    offset: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    """List historical interaction sessions with pagination."""
    service = _get_service(request)
    sessions = service.list_sessions(person_id=person_id, limit=limit, offset=offset)
    return [item.to_dict() for item in sessions]


@router.get("/count")
def count_sessions(request: Request, person_id: str | None = None) -> dict[str, int]:
    """Count historical interaction sessions."""
    service = _get_service(request)
    return {"count": service.count_sessions(person_id=person_id)}


@router.get("/analytics")
def get_aggregate_analytics(
    request: Request,
    person_id: str | None = None,
    is_identified: bool | None = None,
    started_after: float | None = None,
    started_before: float | None = None,
    ended_after: float | None = None,
    ended_before: float | None = None,
    min_duration_seconds: float | None = Query(default=None, ge=0.0),
    max_duration_seconds: float | None = Query(default=None, ge=0.0),
    limit: int = Query(50, gt=0),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Return SQL-side aggregate session analytics."""
    service = _get_service(request)
    filter_obj = SessionQueryFilter(
        person_id=person_id,
        is_identified=is_identified,
        started_after=started_after,
        started_before=started_before,
        ended_after=ended_after,
        ended_before=ended_before,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
        limit=limit,
        offset=offset,
    )
    return service.get_aggregate_analytics(filter=filter_obj).to_dict()


@router.get("/{session_id}")
def get_session(request: Request, session_id: str) -> dict[str, Any]:
    """Retrieve a single historical session record."""
    service = _get_service(request)
    item = service.get_session(session_id)
    if item is None:
        raise EntityNotFoundError("session", session_id)
    return item.to_dict()
