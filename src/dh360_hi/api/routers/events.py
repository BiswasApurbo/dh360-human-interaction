"""Live interaction event/state SSE stream for operator/UI clients."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from dh360_hi.api.event_hub import (
    HEARTBEAT_SECONDS,
    RETRY_MILLIS,
    InteractionEventHub,
    format_context_frame,
    format_event_frame,
    format_heartbeat_frame,
    format_resync_frame,
    format_retry_frame,
)
from dh360_hi.api.security import OperatorPrincipal, OperatorScope, require_scope
from dh360_hi.application.service import HumanInteractionService
from dh360_hi.contracts.integration import BaseIntegrationEvent

router = APIRouter(prefix="", tags=["events"])
_interaction_read = Depends(require_scope(OperatorScope.INTERACTION_READ))


def _get_service(request: Request) -> HumanInteractionService:
    svc = cast(HumanInteractionService | None, getattr(request.app.state, "service", None))
    if svc is None:
        raise HTTPException(status_code=503, detail="Application service is not initialized")
    return svc


def _get_hub(request: Request) -> InteractionEventHub:
    hub = cast(InteractionEventHub | None, getattr(request.app.state, "event_hub", None))
    if hub is None:
        raise HTTPException(status_code=503, detail="Event stream is not initialized")
    return hub


def _parse_last_event_id(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value >= 0 else None


async def _wait_for_items(
    sync_queue: queue.Queue[tuple[int, BaseIntegrationEvent]],
) -> list[tuple[int, BaseIntegrationEvent]]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + HEARTBEAT_SECONDS
    while True:
        items: list[tuple[int, BaseIntegrationEvent]] = []
        while True:
            try:
                items.append(sync_queue.get_nowait())
            except queue.Empty:
                break
        if items:
            return items
        if loop.time() >= deadline:
            return []
        await asyncio.sleep(0.05)



@router.get("/interaction/events")
async def stream_interaction_events(
    request: Request,
    _operator: OperatorPrincipal = _interaction_read,
) -> StreamingResponse:
    """Stream sanitized interaction events and state snapshots over SSE."""
    service = _get_service(request)
    hub = _get_hub(request)
    last_event_id = _parse_last_event_id(request.headers.get("last-event-id"))

    async def event_generator() -> AsyncIterator[str]:
        sub_id, sync_queue = hub.subscribe()
        try:
            yield format_retry_frame(RETRY_MILLIS)
            yield format_context_frame(service.get_interaction_context())
            replay, expired = hub.replay_since(last_event_id)
            if expired:
                yield format_resync_frame(service.get_interaction_context())
            else:
                for seq, evt in replay:
                    yield format_event_frame(seq, evt)
            if hub.consume_resync(sub_id):
                yield format_resync_frame(service.get_interaction_context())
            while True:
                if await request.is_disconnected():
                    break
                drained = await _wait_for_items(sync_queue)
                if await request.is_disconnected():
                    break
                if not drained:
                    yield format_heartbeat_frame()
                    continue
                if hub.consume_resync(sub_id):
                    yield format_resync_frame(service.get_interaction_context())
                for seq, evt in drained:
                    yield format_event_frame(seq, evt)
        finally:
            hub.unsubscribe(sub_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
