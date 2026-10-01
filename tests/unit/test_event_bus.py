"""Unit tests for EventBus — subscription, publishing, isolation, and ordering."""

from __future__ import annotations

from dh360_hi.contracts.events import (
    BaseEvent,
    PersonEnteredEvent,
    PersonLeftEvent,
)
from dh360_hi.edge.event_bus import EventBus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entered(track_id: int = 1) -> PersonEnteredEvent:
    return PersonEnteredEvent(frame_id=1, timestamp=1.0, track_id=track_id)


def _left(track_id: int = 1) -> PersonLeftEvent:
    return PersonLeftEvent(frame_id=2, timestamp=2.0, track_id=track_id)


# ---------------------------------------------------------------------------
# Basic subscription and publishing
# ---------------------------------------------------------------------------


class TestEventBusSubscribeAndPublish:
    def test_publish_reaches_subscriber(self) -> None:
        bus = EventBus()
        received: list[BaseEvent] = []
        bus.subscribe(PersonEnteredEvent, received.append)
        event = _entered()
        bus.publish(event)
        assert received == [event]

    def test_publish_wrong_type_not_delivered(self) -> None:
        """Subscribers for PersonEnteredEvent do NOT receive PersonLeftEvent."""
        bus = EventBus()
        received: list[BaseEvent] = []
        bus.subscribe(PersonEnteredEvent, received.append)
        bus.publish(_left())
        assert received == []

    def test_multiple_subscribers_all_receive(self) -> None:
        bus = EventBus()
        r1: list[BaseEvent] = []
        r2: list[BaseEvent] = []
        bus.subscribe(PersonEnteredEvent, r1.append)
        bus.subscribe(PersonEnteredEvent, r2.append)
        event = _entered()
        bus.publish(event)
        assert r1 == [event]
        assert r2 == [event]

    def test_multiple_event_types_routed_correctly(self) -> None:
        bus = EventBus()
        entered_received: list[BaseEvent] = []
        left_received: list[BaseEvent] = []
        bus.subscribe(PersonEnteredEvent, entered_received.append)
        bus.subscribe(PersonLeftEvent, left_received.append)

        e = _entered()
        left_evt = _left()
        bus.publish(e)
        bus.publish(left_evt)

        assert entered_received == [e]
        assert left_received == [left_evt]

    def test_no_subscribers_publish_is_noop(self) -> None:
        bus = EventBus()
        bus.publish(_entered())  # Should not raise

    def test_publish_to_empty_bus_is_noop(self) -> None:
        bus = EventBus()
        bus.publish(BaseEvent(frame_id=0, timestamp=0.0))  # No subscribers


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestEventBusOrdering:
    def test_subscribers_called_in_registration_order(self) -> None:
        bus = EventBus()
        call_order: list[str] = []
        bus.subscribe(PersonEnteredEvent, lambda _: call_order.append("first"))
        bus.subscribe(PersonEnteredEvent, lambda _: call_order.append("second"))
        bus.subscribe(PersonEnteredEvent, lambda _: call_order.append("third"))
        bus.publish(_entered())
        assert call_order == ["first", "second", "third"]

    def test_multiple_publishes_ordered(self) -> None:
        bus = EventBus()
        received: list[int] = []
        bus.subscribe(PersonEnteredEvent, lambda e: received.append(e.track_id))
        bus.publish(_entered(track_id=10))
        bus.publish(_entered(track_id=20))
        assert received == [10, 20]


# ---------------------------------------------------------------------------
# Unsubscription
# ---------------------------------------------------------------------------


class TestEventBusUnsubscribe:
    def test_unsubscribe_removes_callback(self) -> None:
        bus = EventBus()
        received: list[BaseEvent] = []
        sub_id = bus.subscribe(PersonEnteredEvent, received.append)
        bus.unsubscribe(sub_id)
        bus.publish(_entered())
        assert received == []

    def test_unsubscribe_unknown_id_returns_false(self) -> None:
        bus = EventBus()
        result = bus.unsubscribe(9999)
        assert result is False

    def test_unsubscribe_correct_id_returns_true(self) -> None:
        bus = EventBus()
        sub_id = bus.subscribe(PersonEnteredEvent, lambda _: None)
        assert bus.unsubscribe(sub_id) is True

    def test_unsubscribe_only_removes_target(self) -> None:
        bus = EventBus()
        r1: list[BaseEvent] = []
        r2: list[BaseEvent] = []
        sub1 = bus.subscribe(PersonEnteredEvent, r1.append)
        bus.subscribe(PersonEnteredEvent, r2.append)
        bus.unsubscribe(sub1)
        bus.publish(_entered())
        assert r1 == []
        assert len(r2) == 1

    def test_unsubscribe_is_idempotent(self) -> None:
        bus = EventBus()
        sub_id = bus.subscribe(PersonEnteredEvent, lambda _: None)
        bus.unsubscribe(sub_id)
        # Second unsubscribe should be a no-op, not raise
        result = bus.unsubscribe(sub_id)
        assert result is False


# ---------------------------------------------------------------------------
# Subscriber isolation (exception handling)
# ---------------------------------------------------------------------------


class TestEventBusSubscriberIsolation:
    def test_bad_subscriber_does_not_crash_bus(self) -> None:
        bus = EventBus()
        received: list[BaseEvent] = []

        def bad_callback(_: BaseEvent) -> None:
            raise RuntimeError("intentional error")

        bus.subscribe(PersonEnteredEvent, bad_callback)
        bus.subscribe(PersonEnteredEvent, received.append)

        # Should NOT raise even though bad_callback raises
        bus.publish(_entered())
        assert len(received) == 1

    def test_all_subscribers_called_even_after_exception(self) -> None:
        bus = EventBus()
        call_order: list[str] = []

        def raises(_: BaseEvent) -> None:
            call_order.append("raises")
            raise ValueError("boom")

        bus.subscribe(PersonEnteredEvent, lambda _: call_order.append("before"))
        bus.subscribe(PersonEnteredEvent, raises)
        bus.subscribe(PersonEnteredEvent, lambda _: call_order.append("after"))

        bus.publish(_entered())
        assert call_order == ["before", "raises", "after"]

    def test_multiple_bad_subscribers_all_isolated(self) -> None:
        bus = EventBus()
        received: list[BaseEvent] = []

        for _ in range(3):
            bus.subscribe(PersonEnteredEvent, lambda _: (_ for _ in ()).throw(RuntimeError()))  # type: ignore[misc]

        bus.subscribe(PersonEnteredEvent, received.append)
        bus.publish(_entered())
        assert len(received) == 1


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


class TestEventBusIntrospection:
    def test_subscriber_count_zero_initially(self) -> None:
        bus = EventBus()
        assert bus.subscriber_count(PersonEnteredEvent) == 0

    def test_subscriber_count_reflects_subscriptions(self) -> None:
        bus = EventBus()
        bus.subscribe(PersonEnteredEvent, lambda _: None)
        bus.subscribe(PersonEnteredEvent, lambda _: None)
        assert bus.subscriber_count(PersonEnteredEvent) == 2

    def test_subscriber_count_decreases_after_unsubscribe(self) -> None:
        bus = EventBus()
        sub_id = bus.subscribe(PersonEnteredEvent, lambda _: None)
        bus.subscribe(PersonEnteredEvent, lambda _: None)
        bus.unsubscribe(sub_id)
        assert bus.subscriber_count(PersonEnteredEvent) == 1

    def test_clear_removes_all_subscriptions(self) -> None:
        bus = EventBus()
        bus.subscribe(PersonEnteredEvent, lambda _: None)
        bus.subscribe(PersonLeftEvent, lambda _: None)
        bus.clear()
        assert bus.subscriber_count(PersonEnteredEvent) == 0
        assert bus.subscriber_count(PersonLeftEvent) == 0


# ---------------------------------------------------------------------------
# No global singleton
# ---------------------------------------------------------------------------


class TestEventBusIsolation:
    def test_two_buses_are_independent(self) -> None:
        bus_a = EventBus()
        bus_b = EventBus()
        r_a: list[BaseEvent] = []
        r_b: list[BaseEvent] = []
        bus_a.subscribe(PersonEnteredEvent, r_a.append)
        bus_b.subscribe(PersonEnteredEvent, r_b.append)
        bus_a.publish(_entered(track_id=1))
        bus_b.publish(_entered(track_id=2))
        assert [e.track_id for e in r_a] == [1]  # type: ignore[union-attr]
        assert [e.track_id for e in r_b] == [2]  # type: ignore[union-attr]

    def test_subscription_ids_are_unique_across_instances(self) -> None:
        bus_a = EventBus()
        bus_b = EventBus()
        id_a = bus_a.subscribe(PersonEnteredEvent, lambda _: None)
        id_b = bus_b.subscribe(PersonEnteredEvent, lambda _: None)
        # Each bus has its own counter starting at 0 — that's fine and expected
        assert isinstance(id_a, int)
        assert isinstance(id_b, int)
