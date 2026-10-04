"""FastAPI application exposing the existing application service over HTTP."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dh360_hi.api.event_hub import InteractionEventHub
from dh360_hi.api.routers.enrollment import router as enrollment_router
from dh360_hi.api.routers.events import router as events_router
from dh360_hi.api.routers.health import router as health_router
from dh360_hi.api.routers.interaction import router as interaction_router
from dh360_hi.api.routers.persons import router as persons_router
from dh360_hi.api.routers.sessions import router as sessions_router
from dh360_hi.api.security import OperatorSecurityError
from dh360_hi.application.factory import create_interaction_service
from dh360_hi.application.loader import load_config
from dh360_hi.backend.exceptions import ValidationError as BackendValidationError
from dh360_hi.contracts.errors import ApplicationError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create and tear down the application service for the API lifecycle."""
    cfg = load_config(validate=True)
    service = create_interaction_service(config=cfg, enable_perception=False)
    hub = InteractionEventHub()
    hub.attach(service)
    app.state.service = service
    app.state.event_hub = hub
    app.state.operator_auth = cfg.api
    try:
        yield
    finally:
        hub.detach()
        with suppress(Exception):
            service.stop(timeout=3.0)


def create_app() -> FastAPI:
    """Construct the FastAPI application with the existing service wired in."""
    app = FastAPI(title="dh360-human-interaction API", lifespan=lifespan)

    @app.exception_handler(ApplicationError)
    async def handle_app_error(_request: Request, exc: ApplicationError) -> JSONResponse:
        status = 400
        if exc.__class__.__name__ in {"ServiceUnavailableError"}:
            status = 503
        elif exc.__class__.__name__ in {"ValidationAppError", "ValidationError"}:
            status = 422
        elif exc.__class__.__name__ in {"EntityNotFoundError"}:
            status = 404
        elif exc.__class__.__name__ in {"ConflictAppError"}:
            status = 409
        elif exc.__class__.__name__ in {"EnrollmentAppError"}:
            status = 422
        return JSONResponse(status_code=status, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "VALIDATION_ERROR", "details": exc.errors()},
        )

    @app.exception_handler(OperatorSecurityError)
    async def handle_operator_security_error(
        _request: Request,
        exc: OperatorSecurityError,
    ) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
            headers=headers,
        )

    @app.exception_handler(BackendValidationError)
    async def handle_backend_validation_error(
        _request: Request,
        exc: BackendValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "VALIDATION_ERROR", "details": {"message": str(exc)}},
        )

    app.include_router(health_router)
    app.include_router(interaction_router)
    app.include_router(sessions_router)
    app.include_router(persons_router)
    app.include_router(enrollment_router)
    app.include_router(events_router)
    return app
