"""FastAPI application exposing the existing application service over HTTP."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dh360_hi.api.event_hub import InteractionEventHub
from dh360_hi.api.routers.authorization import router as authorization_router
from dh360_hi.api.routers.demo import router as demo_router
from dh360_hi.api.routers.dialogue import router as dialogue_router
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
from dh360_hi.contracts.authorization import AuthorizationSession
from dh360_hi.contracts.errors import ApplicationError
from dh360_hi.contracts.integration import AuthorizationUpdatedIntegrationEvent
from dh360_hi.edge.authorization_controller import AuthorizationController

if TYPE_CHECKING:
    from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
    from dh360_hi.application.service import HumanInteractionService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create and tear down the application service for the API lifecycle."""
    # Reuse injected/pre-existing service if already provided
    service: HumanInteractionService | None = getattr(app.state, "service", None)
    if service is None:
        cfg = load_config(validate=True)
        service = create_interaction_service(config=cfg, enable_perception=False)
        app.state.service = service
        app.state.operator_auth = cfg.api
    else:
        cfg = load_config(validate=True)
        if getattr(app.state, "operator_auth", None) is None:
            app.state.operator_auth = cfg.api

    # Reuse or attach EventHub
    hub: InteractionEventHub | None = getattr(app.state, "event_hub", None)
    if hub is None:
        hub = InteractionEventHub()
        hub.attach(service)
        app.state.event_hub = hub

    # If an AuthorizationController was already injected, use it;
    # otherwise, if the service has an active runtime with a SessionManager, construct it.
    auth_ctrl: AuthorizationController | None = getattr(app.state, "authorization_controller", None)
    if auth_ctrl is None:
        if service.runtime is not None and service.runtime.session_manager is not None:
            timeout_sec = (
                cfg.runtime.session_timeout_seconds
                if hasattr(cfg.runtime, "session_timeout_seconds")
                else 30.0
            )

            def _on_auth_transition(s: AuthorizationSession) -> None:
                if hub is not None:
                    hub.publish(AuthorizationUpdatedIntegrationEvent.from_session(s))

            auth_ctrl = AuthorizationController(
                session_manager=service.runtime.session_manager,
                timeout_seconds=timeout_sec,
                on_transition=_on_auth_transition,
            )
            app.state.authorization_controller = auth_ctrl
        else:
            app.state.authorization_controller = None
    elif auth_ctrl._on_transition is None and hub is not None:
        # If injected without callback, wire to event hub
        def _on_injected_transition(s: AuthorizationSession) -> None:
            if hub is not None:
                hub.publish(AuthorizationUpdatedIntegrationEvent.from_session(s))

        auth_ctrl._on_transition = _on_injected_transition

    if auth_ctrl is not None and service is not None:
        service.authorization_controller = auth_ctrl

    try:
        yield
    finally:
        # Deterministic shutdown ordering:
        # 1. Close DialogueCoordinator if present
        dialogue_coord: DialogueCoordinator | None = getattr(
            app.state, "dialogue_coordinator", None
        )
        if dialogue_coord is not None:
            with suppress(Exception):
                dialogue_coord.close()

        # 2. Close AuthorizationController so it ceases processing frame results
        ctrl: AuthorizationController | None = getattr(app.state, "authorization_controller", None)
        if ctrl is not None:
            with suppress(Exception):
                ctrl.close()

        # 3. Detach InteractionEventHub listeners
        event_hub: InteractionEventHub | None = getattr(app.state, "event_hub", None)
        if event_hub is not None:
            event_hub.detach()

        # 4. Stop application service and runtime workers
        svc: HumanInteractionService | None = getattr(app.state, "service", None)
        if svc is not None:
            svc.authorization_controller = None
            with suppress(Exception):
                svc.stop(timeout=3.0)


def create_app(
    service: HumanInteractionService | None = None,
    authorization_controller: AuthorizationController | None = None,
    dialogue_coordinator: DialogueCoordinator | None = None,
) -> FastAPI:
    """Construct the FastAPI application with service and controller wired in."""
    app = FastAPI(title="dh360-human-interaction API", lifespan=lifespan)
    cfg = load_config(validate=True)

    if service is None:
        service = create_interaction_service(config=cfg, enable_perception=False)
    app.state.service = service
    app.state.operator_auth = cfg.api
    app.state.dialogue_coordinator = dialogue_coordinator

    hub = InteractionEventHub()
    hub.attach(service)
    app.state.event_hub = hub

    if authorization_controller is not None:
        if authorization_controller._on_transition is None:

            def _on_auth_trans(s: AuthorizationSession) -> None:
                hub.publish(AuthorizationUpdatedIntegrationEvent.from_session(s))

            authorization_controller._on_transition = _on_auth_trans
        app.state.authorization_controller = authorization_controller
        service.authorization_controller = authorization_controller
    elif service.runtime is not None and service.runtime.session_manager is not None:
        timeout_sec = (
            cfg.runtime.session_timeout_seconds
            if hasattr(cfg.runtime, "session_timeout_seconds")
            else 30.0
        )

        def _on_auth_trans_created(s: AuthorizationSession) -> None:
            hub.publish(AuthorizationUpdatedIntegrationEvent.from_session(s))

        app.state.authorization_controller = AuthorizationController(
            session_manager=service.runtime.session_manager,
            timeout_seconds=timeout_sec,
            on_transition=_on_auth_trans_created,
        )
        service.authorization_controller = app.state.authorization_controller
    else:
        app.state.authorization_controller = None

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
        elif exc.__class__.__name__ in {"ProviderTimeoutError"}:
            status = 504
        elif exc.__class__.__name__ in {
            "ProviderNetworkError",
            "LLMProviderError",
            "STTProviderError",
            "TTSProviderError",
            "ProviderError",
        }:
            status = 502
        elif exc.__class__.__name__ in {"HardwareUnavailableError"}:
            status = 503
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
    app.include_router(authorization_router)
    app.include_router(demo_router)
    app.include_router(dialogue_router)
    return app
