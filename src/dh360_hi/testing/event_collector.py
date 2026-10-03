"""Thread-safe event collector for recording and inspecting integration events."""

from __future__ import annotations

import threading
from typing import TypeVar

from dh360_hi.contracts.events import BaseEvent
from dh360_hi.contracts.integration import BaseIntegrationEvent

T = TypeVar("T", bound=BaseIntegrationEvent)
D = TypeVar("D", bound=BaseEvent)


class ScenarioEventCollector:
    """Collects and stores emitted integration and domain events during scenario execution."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._integration_events: list[BaseIntegrationEvent] = []
        self._domain_events: list[BaseEvent] = []

    def on_integration_event(self, event: BaseIntegrationEvent) -> None:
        """Listener callback for external integration events."""
        with self._lock:
            self._integration_events.append(event)

    def on_domain_event(self, event: BaseEvent) -> None:
        """Listener callback for internal domain events."""
        with self._lock:
            self._domain_events.append(event)

    def get_integration_events(self) -> list[BaseIntegrationEvent]:
        """Retrieve a shallow copy of all recorded integration events."""
        with self._lock:
            return list(self._integration_events)

    def get_domain_events(self) -> list[BaseEvent]:
        """Retrieve a shallow copy of all recorded domain events."""
        with self._lock:
            return list(self._domain_events)

    def get_event_types(self) -> list[str]:
        """Return list of event_type strings for all recorded integration events in order."""
        with self._lock:
            return [e.event_type for e in self._integration_events]

    def find_first_event(self, event_type: str) -> BaseIntegrationEvent | None:
        """Find the first integration event matching the given event_type string."""
        with self._lock:
            for event in self._integration_events:
                if event.event_type == event_type:
                    return event
            return None

    def find_events_by_type(self, event_cls: type[T]) -> list[T]:
        """Filter recorded integration events by their concrete dataclass type."""
        with self._lock:
            return [e for e in self._integration_events if isinstance(e, event_cls)]

    def clear(self) -> None:
        """Reset all recorded events."""
        with self._lock:
            self._integration_events.clear()
            self._domain_events.clear()
