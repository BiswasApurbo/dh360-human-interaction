"""In-process synchronous event bus for domain events.

Design properties
-----------------
* **No global singleton** — callers instantiate ``EventBus`` explicitly.
* **Synchronous** — ``publish()`` calls all subscriber callbacks inline, in
  the same thread and call stack as the publisher.
* **No threads, queues, or network** — purely in-process.
* **Subscriber isolation** — an exception in one subscriber is logged and
  swallowed; it does NOT crash the bus or block remaining subscribers.
* **Deterministic ordering** — subscribers are called in the order they were
  registered for each event type.
* **Multiple subscribers per type** — each ``subscribe()`` returns a unique
  integer subscription ID that can be passed to ``unsubscribe()``.
* **Covariant dispatch** — publishing a subclass event notifies subscribers
  registered for that exact type (no automatic parent-type fan-out; keep it
  simple and explicit).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from dh360_hi.contracts.events import BaseEvent

logger = logging.getLogger(__name__)

E = TypeVar("E", bound=BaseEvent)

Callback = Callable[[BaseEvent], None]


class EventBus:
    """Synchronous in-process event bus.

    Usage::

        bus = EventBus()
        sub_id = bus.subscribe(PersonEnteredEvent, on_entered)
        bus.publish(PersonEnteredEvent(...))   # on_entered called inline
        bus.unsubscribe(sub_id)
    """

    def __init__(self) -> None:
        # Maps event type → ordered list of (subscription_id, callback) pairs
        self._subscribers: dict[type[BaseEvent], list[tuple[int, Callback]]] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: type[E],
        callback: Callable[[E], None],
    ) -> int:
        """Register *callback* to be called whenever *event_type* is published.

        Args:
            event_type: The exact event class to subscribe to.
            callback:   A callable that accepts one argument of *event_type*.

        Returns:
            An opaque integer subscription ID for later use with
            :meth:`unsubscribe`.
        """
        sub_id = self._next_id
        self._next_id += 1

        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        # Store as (BaseEvent → None) to keep the dict homogeneous;
        # the TypeVar annotation on the public API gives callers type safety.
        self._subscribers[event_type].append((sub_id, callback))  # type: ignore[arg-type]
        return sub_id

    def unsubscribe(self, subscription_id: int) -> bool:
        """Remove the subscription identified by *subscription_id*.

        Args:
            subscription_id: The ID returned by a previous :meth:`subscribe`
                             call.

        Returns:
            ``True`` if the subscription was found and removed, ``False`` if
            the ID was not registered (idempotent — no exception raised).
        """
        for event_type, subs in self._subscribers.items():
            for i, (sub_id, _) in enumerate(subs):
                if sub_id == subscription_id:
                    self._subscribers[event_type].pop(i)
                    return True
        return False

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: BaseEvent) -> None:
        """Deliver *event* to all subscribers registered for its exact type.

        Subscribers are called in registration order.  An exception raised by
        one subscriber is caught and logged; remaining subscribers still
        receive the event.

        Args:
            event: The event instance to deliver.
        """
        event_type = type(event)
        subs = self._subscribers.get(event_type)
        if not subs:
            return

        for sub_id, callback in list(subs):
            try:
                callback(event)
            except Exception:
                logger.exception(
                    "EventBus: subscriber %d raised an unhandled exception for %s",
                    sub_id,
                    event_type.__name__,
                )

    # ------------------------------------------------------------------
    # Introspection (useful for testing)
    # ------------------------------------------------------------------

    def subscriber_count(self, event_type: type[BaseEvent]) -> int:
        """Return the number of active subscribers for *event_type*."""
        return len(self._subscribers.get(event_type, []))

    def clear(self) -> None:
        """Remove all subscriptions from every event type."""
        self._subscribers.clear()
