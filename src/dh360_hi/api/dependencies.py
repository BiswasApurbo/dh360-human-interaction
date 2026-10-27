"""FastAPI dependencies for the thin application transport layer."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from fastapi import HTTPException, Request

from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
from dh360_hi.application.factory import create_interaction_service
from dh360_hi.application.loader import load_config
from dh360_hi.application.service import HumanInteractionService
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.contracts.errors import ApplicationError, ServiceUnavailableError


def _build_service() -> HumanInteractionService:
    """Create the application service using the existing composition root."""
    config = load_config(validate=True)
    return create_interaction_service(config=config, enable_perception=True)


def get_service(request: Request) -> HumanInteractionService:
    """Fetch the singleton service instance from app state."""
    svc = cast(HumanInteractionService | None, request.app.state.service)
    if svc is None:
        raise HTTPException(status_code=503, detail="Application service is not initialized")
    return svc


def get_service_or_503() -> Callable[[Request], HumanInteractionService]:
    """Dependency wrapper that fails cleanly when the service is unavailable."""

    def _resolver(request: Request) -> HumanInteractionService:
        svc = cast(HumanInteractionService | None, request.app.state.service)
        if svc is None:
            raise HTTPException(status_code=503, detail="Application service is not initialized")
        return svc

    return _resolver


def get_dialogue_coordinator(request: Request) -> DialogueCoordinator:
    """Fetch the singleton DialogueCoordinator instance from app state."""
    coord = cast(
        DialogueCoordinator | None,
        getattr(request.app.state, "dialogue_coordinator", None),
    )
    if coord is None:
        raise HTTPException(
            status_code=503,
            detail="Dialogue coordinator is not configured or initialized",
        )
    return coord


def get_knowledge_service(request: Request) -> KnowledgeService:
    """Fetch or lazily initialize the KnowledgeService instance from app state."""
    knowledge_svc = cast(
        KnowledgeService | None,
        getattr(request.app.state, "knowledge_service", None),
    )
    if knowledge_svc is not None:
        return knowledge_svc

    coord = getattr(request.app.state, "dialogue_coordinator", None)
    if coord is not None and hasattr(coord, "_knowledge_service"):
        knowledge_svc = coord._knowledge_service
        request.app.state.knowledge_service = knowledge_svc
        return knowledge_svc

    svc = cast(HumanInteractionService | None, getattr(request.app.state, "service", None))
    person_svc = getattr(svc, "person_service", None) if svc is not None else None
    knowledge_svc = KnowledgeService(person_service=person_svc)
    request.app.state.knowledge_service = knowledge_svc
    return knowledge_svc


def _translate_app_error(exc: ApplicationError) -> HTTPException:
    """Map existing application errors to HTTP status codes."""
    if isinstance(exc, ServiceUnavailableError):
        return HTTPException(status_code=503, detail=exc.to_dict())
    if exc.__class__.__name__ in {"ValidationAppError", "ValidationError"}:
        return HTTPException(status_code=422, detail=exc.to_dict())
    if exc.__class__.__name__ in {"EntityNotFoundError"}:
        return HTTPException(status_code=404, detail=exc.to_dict())
    if exc.__class__.__name__ in {"ConflictAppError"}:
        return HTTPException(status_code=409, detail=exc.to_dict())
    return HTTPException(status_code=400, detail=exc.to_dict())


def service_dependency() -> Callable[[Request], HumanInteractionService]:
    """Compatibility dependency wrapper for routers."""
    return get_service


def unwrap_service_error() -> None:
    """No-op placeholder for compatibility with signatures using dependency injection."""
    return None
