"""KBCON Humanoid Robot Kiosk Display and Touchscreen Interaction Router.

Provides dedicated endpoints for the robot's physical touchscreen display,
including fullscreen eye animation display, live state synchronization via SSE,
touchscreen microphone muting, and speech interruption.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from dh360_hi.backend.robot_registry import RobotRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kiosk", tags=["Kiosk"])
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class KioskStateResponse(BaseModel):
    """Authoritative physical robot state payload for the touchscreen display."""

    robot_id: str = Field(..., description="Unique physical robot identifier")
    name: str = Field(..., description="Configured robot display name")
    language: str = Field(..., description="Configured authoritative response language (bn or en)")
    mode: str = Field(..., description="Operational domain mode (education or healthcare)")
    tenant_id: str | None = Field(None, description="Owning tenant/company identifier")
    state: str = Field(
        ...,
        description="Current interaction state (idle, listening, thinking, speaking, muted, interrupted, error)",
    )
    is_authorized: bool = Field(
        ..., description="Whether a face is currently recognized and authorized"
    )
    display_name: str | None = Field(
        None, description="Authorized person's display name if recognized"
    )
    is_muted: bool = Field(..., description="Whether microphone recording is actively muted")
    last_user_query: str = Field("", description="Latest recognized user speech prompt")
    last_robot_response: str = Field("", description="Latest spoken robot response")


class KioskTurnRequest(BaseModel):
    """Optional turn trigger request from touchscreen interface."""

    text: str | None = Field(None, description="Direct user query prompt text")
    speak_response: bool = Field(True, description="Whether to synthesize audio response")


def _get_current_kiosk_state(request: Request) -> dict[str, Any]:
    """Assemble a complete snapshot of the active robot's identity and interaction state."""
    active_robot = RobotRegistry.get_instance().get_active_robot()
    coord = getattr(request.app.state, "dialogue_coordinator", None)
    auth_ctrl = getattr(request.app.state, "authorization_controller", None)

    is_authorized = False
    display_name = None
    if auth_ctrl is not None:
        current_auth = auth_ctrl.get_current()
        if (
            current_auth is not None
            and current_auth.is_active
            and current_auth.is_authorized
            and current_auth.person_id is not None
        ):
            is_authorized = True
            display_name = current_auth.display_name or current_auth.person_id

    interaction_state = "idle"
    is_muted = False
    last_user_query = ""
    last_robot_response = ""

    if coord is not None:
        interaction_state = coord.current_state
        is_muted = coord.is_microphone_muted
        last_user_query = coord.last_user_query
        last_robot_response = coord.last_robot_response

    return {
        "robot_id": active_robot.robot_id,
        "name": active_robot.name,
        "language": active_robot.language,
        "mode": active_robot.mode,
        "tenant_id": active_robot.tenant_id,
        "state": interaction_state,
        "is_authorized": is_authorized,
        "display_name": display_name,
        "is_muted": is_muted,
        "last_user_query": last_user_query,
        "last_robot_response": last_robot_response,
    }


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def serve_kiosk_page() -> Response:
    """Serve the dedicated KBCON touchscreen kiosk application page."""
    html_path = STATIC_DIR / "kiosk.html"
    if not html_path.is_file():
        return HTMLResponse(
            content="<h1>KBCON Kiosk Page Under Construction</h1>",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return FileResponse(
        path=str(html_path),
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/state", response_model=KioskStateResponse)
async def get_kiosk_state(request: Request) -> JSONResponse:
    """Retrieve the authoritative runtime state of the robot display and audio pipelines."""
    snapshot = _get_current_kiosk_state(request)
    return JSONResponse(content=snapshot)


@router.post("/mute")
async def toggle_kiosk_mute(request: Request) -> JSONResponse:
    """Toggle microphone mute from the left 50% touchscreen tap zone."""
    coord = getattr(request.app.state, "dialogue_coordinator", None)
    if coord is None:
        return JSONResponse(
            content={"error": "DIALOGUE_COORDINATOR_UNAVAILABLE"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    is_muted = coord.toggle_microphone_mute()
    return JSONResponse(
        content={
            "status": "success",
            "is_muted": is_muted,
            "state": coord.current_state,
        }
    )


@router.post("/interrupt")
async def trigger_kiosk_interrupt(request: Request) -> JSONResponse:
    """Immediately interrupt ongoing speech synthesis and audio from right 50% tap zone."""
    coord = getattr(request.app.state, "dialogue_coordinator", None)
    if coord is None:
        return JSONResponse(
            content={"error": "DIALOGUE_COORDINATOR_UNAVAILABLE"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    coord.interrupt()
    return JSONResponse(
        content={
            "status": "success",
            "state": "interrupted",
        }
    )


@router.post("/turn")
async def handle_kiosk_turn(payload: KioskTurnRequest, request: Request) -> JSONResponse:
    """Execute a dialogue turn directly from the touchscreen kiosk interface."""
    coord = getattr(request.app.state, "dialogue_coordinator", None)
    if coord is None:
        return JSONResponse(
            content={"error": "DIALOGUE_COORDINATOR_UNAVAILABLE"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    turn = await coord.handle_turn_async(
        user_text=payload.text,
        speak_response=payload.speak_response,
    )

    return JSONResponse(
        content={
            "role": turn.role,
            "text": turn.text,
            "language": turn.language,
            "metadata": turn.metadata,
        }
    )


@router.get("/events")
async def stream_kiosk_events(request: Request) -> StreamingResponse:
    """Stream real-time interaction states and face authorization updates via Server-Sent Events."""
    coord = getattr(request.app.state, "dialogue_coordinator", None)
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=30)
    loop = asyncio.get_running_loop()

    def _state_listener(new_state: str, metadata: dict[str, Any]) -> None:
        try:
            snapshot = _get_current_kiosk_state(request)
            snapshot["state"] = new_state
            if "user_text" in metadata:
                snapshot["last_user_query"] = metadata["user_text"]
            if "response_text" in metadata:
                snapshot["last_robot_response"] = metadata["response_text"]
            loop.call_soon_threadsafe(
                lambda: event_queue.put_nowait(snapshot) if not event_queue.full() else None
            )
        except Exception as exc:
            logger.debug("Kiosk event queue error: %s", exc)

    if coord is not None:
        coord.register_state_callback(_state_listener)

    async def _event_generator() -> AsyncGenerator[str, None]:
        try:
            # Yield initial state immediately on connect
            initial_state = _get_current_kiosk_state(request)
            yield f"data: {json.dumps(initial_state)}\n\n"

            while True:
                if await request.is_disconnected():
                    break

                try:
                    state_update = await asyncio.wait_for(event_queue.get(), timeout=3.0)
                    yield f"data: {json.dumps(state_update)}\n\n"
                except TimeoutError:
                    # Send periodic heartbeat snapshot
                    heartbeat_snapshot = _get_current_kiosk_state(request)
                    yield f"data: {json.dumps(heartbeat_snapshot)}\n\n"
        finally:
            if coord is not None:
                coord.unregister_state_callback(_state_listener)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
