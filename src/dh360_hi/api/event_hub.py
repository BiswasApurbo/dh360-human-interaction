"""Thread-safe synchronous core for the live interaction event/state stream."""

from __future__ import annotations

import contextlib
import json
import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from dh360_hi.contracts.dto import InteractionContextDTO
from dh360_hi.contracts.integration import (
    BaseIntegrationEvent,
    InteractionContextUpdatedIntegrationEvent,
)

BUFFER_SIZE = 64
QUEUE_SIZE = 256
HEARTBEAT_SECONDS = 15.0
RETRY_MILLIS = 3000


class _ServiceProtocol(Protocol):
    """Minimal service surface the hub attaches to (no repository/DB access)."""

    def add_integration_event_listener(self, callback: Any) -> int: ...
    def remove_integration_event_listener(self, listener_id: int) -> bool: ...
    def add_context_listener(self, callback: Any) -> int: ...
    def remove_context_listener(self, listener_id: int) -> bool: ...


@dataclass
class _Subscriber:
    """Per-connection bounded buffer with overflow resync signalling."""

    buffer: queue.Queue[tuple[int, BaseIntegrationEvent]] = field(
        default_factory=lambda: queue.Queue(maxsize=QUEUE_SIZE)
    )
    needs_resync: bool = False


class InteractionEventHub:
    """Synchronous thread-safe fan-out hub for sanitized interaction events."""

    def __init__(self, buffer_size: int = BUFFER_SIZE, queue_size: int = QUEUE_SIZE) -> None:
        self._buffer_size = max(1, buffer_size)
        self._queue_size = max(1, queue_size)
        self._lock = threading.Lock()
        self._next_seq = 1
        self._ring: deque[tuple[int, BaseIntegrationEvent]] = deque(maxlen=self._buffer_size)
        self._subscribers: dict[int, _Subscriber] = {}
        self._next_subscriber_id = 1
        self._service: _ServiceProtocol | None = None
        self._integration_listener_id: int | None = None
        self._context_listener_id: int | None = None

    @property
    def latest_seq(self) -> int:
        """Highest sequence assigned so far (0 when no event published)."""
        with self._lock:
            return self._next_seq - 1

    @property
    def subscriber_count(self) -> int:
        """Current number of attached SSE subscribers."""
        with self._lock:
            return len(self._subscribers)

    def publish(self, event: BaseIntegrationEvent) -> int:
        """Assign the next sequence, buffer, and fan out without blocking."""
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            self._ring.append((seq, event))
            for subscriber in self._subscribers.values():
                try:
                    subscriber.buffer.put_nowait((seq, event))
                except queue.Full:
                    with contextlib.suppress(queue.Empty):
                        subscriber.buffer.get_nowait()
                    subscriber.needs_resync = True
                    with contextlib.suppress(queue.Full):  # just emptied
                        subscriber.buffer.put_nowait((seq, event))
            return seq

    def subscribe(self) -> tuple[int, queue.Queue[tuple[int, BaseIntegrationEvent]]]:
        """Register one SSE connection and return its id and live buffer."""
        with self._lock:
            sub_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            subscriber = _Subscriber(buffer=queue.Queue(maxsize=self._queue_size))
            self._subscribers[sub_id] = subscriber
            return sub_id, subscriber.buffer

    def unsubscribe(self, subscriber_id: int) -> bool:
        """Remove one SSE connection; safe to call twice."""
        with self._lock:
            return self._subscribers.pop(subscriber_id, None) is not None

    def consume_resync(self, subscriber_id: int) -> bool:
        """Consume and clear the overflow resync flag for one subscriber."""
        with self._lock:
            subscriber = self._subscribers.get(subscriber_id)
            if subscriber is None:
                return False
            flag = subscriber.needs_resync
            subscriber.needs_resync = False
            return flag

    def replay_since(
        self, last_event_id: int | None
    ) -> tuple[list[tuple[int, BaseIntegrationEvent]], bool]:
        """Select buffered events newer than ``last_event_id``."""
        with self._lock:
            if last_event_id is None:
                return [], False
            if not self._ring:
                return [], False
            oldest = self._ring[0][0]
            latest = self._ring[-1][0]
            if last_event_id >= latest:
                return [], False
            if last_event_id < oldest - 1:
                return [], True
            frames = [(seq, evt) for seq, evt in self._ring if seq > last_event_id]
            return frames, False

    def attach(self, service: _ServiceProtocol) -> None:
        """Attach service listener channels; idempotent for the same service."""
        with self._lock:
            if self._service is service and self._integration_listener_id is not None:
                return
            old_service = self._service
            old_integration = self._integration_listener_id
            old_context = self._context_listener_id
        if old_service is not None:
            if old_integration is not None:
                with contextlib.suppress(Exception):  # defensive detach
                    old_service.remove_integration_event_listener(old_integration)
            if old_context is not None:
                with contextlib.suppress(Exception):  # defensive detach
                    old_service.remove_context_listener(old_context)
        integration_id = service.add_integration_event_listener(self._on_integration_event)
        context_id = service.add_context_listener(self._on_context)
        with self._lock:
            self._service = service
            self._integration_listener_id = integration_id
            self._context_listener_id = context_id

    def detach(self) -> None:
        """Remove service listeners; safe to call when never attached."""
        with self._lock:
            service = self._service
            integration_id = self._integration_listener_id
            context_id = self._context_listener_id
            self._service = None
            self._integration_listener_id = None
            self._context_listener_id = None
        if service is None:
            return
        if integration_id is not None:
            with contextlib.suppress(Exception):  # defensive detach
                service.remove_integration_event_listener(integration_id)
        if context_id is not None:
            with contextlib.suppress(Exception):  # defensive detach
                service.remove_context_listener(context_id)

    def _on_integration_event(self, event: BaseIntegrationEvent) -> None:
        """Service integration-channel callback (runs on producer thread)."""
        self.publish(event)

    def _on_context(self, context: InteractionContextDTO) -> None:
        """Service context-channel callback wrapped as a sanitized event."""
        self.publish(InteractionContextUpdatedIntegrationEvent(context=context))


def format_event_frame(seq: int, event: BaseIntegrationEvent) -> str:
    """Format one sequenced integration event as an SSE frame."""
    payload = json.dumps(event.to_dict(), separators=(",", ":"))
    return f"id: {seq}\nevent: {event.event_type}\ndata: {payload}\n\n"


def format_context_frame(context: InteractionContextDTO) -> str:
    """Format the initial snapshot frame (no id; never moves Last-Event-ID)."""
    payload = json.dumps(context.to_dict(), separators=(",", ":"))
    return "event: context\n" f"data: {payload}\n\n"


def format_resync_frame(context: InteractionContextDTO) -> str:
    """Format a recovery snapshot frame (no id; client re-derives state)."""
    payload = json.dumps(context.to_dict(), separators=(",", ":"))
    return "event: resync\n" f"data: {payload}\n\n"


def format_retry_frame(retry_millis: int = RETRY_MILLIS) -> str:
    """Format the SSE reconnect-hint preamble frame."""
    return f"retry: {retry_millis}\n\n"


def format_heartbeat_frame() -> str:
    """Format an SSE comment heartbeat (never affects Last-Event-ID)."""
    return ": hb\n\n"

