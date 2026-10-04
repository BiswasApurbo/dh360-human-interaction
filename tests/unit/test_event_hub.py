"""Unit tests for the synchronous InteractionEventHub core."""

from __future__ import annotations

import json
import threading
from typing import Any

from dh360_hi.api.event_hub import (
    BUFFER_SIZE,
    InteractionEventHub,
    format_context_frame,
    format_event_frame,
    format_heartbeat_frame,
    format_resync_frame,
    format_retry_frame,
)
from dh360_hi.contracts.dto import InteractionContextDTO
from dh360_hi.contracts.integration import (
    InteractionContextUpdatedIntegrationEvent,
    SessionStartedIntegrationEvent,
)


def _event(session_id: str = "session-1") -> SessionStartedIntegrationEvent:
    return SessionStartedIntegrationEvent(session_id=session_id)


class _FakeService:
    """Minimal service double exposing the two listener channels."""

    def __init__(self) -> None:
        self.integration: dict[int, Any] = {}
        self.contexts: dict[int, Any] = {}
        self._next = 1

    def add_integration_event_listener(self, callback: Any) -> int:
        lid = self._next
        self._next += 1
        self.integration[lid] = callback
        return lid

    def remove_integration_event_listener(self, listener_id: int) -> bool:
        return self.integration.pop(listener_id, None) is not None

    def add_context_listener(self, callback: Any) -> int:
        lid = self._next
        self._next += 1
        self.contexts[lid] = callback
        return lid

    def remove_context_listener(self, listener_id: int) -> bool:
        return self.contexts.pop(listener_id, None) is not None


def test_sequence_assignment_starts_at_one_and_increments() -> None:
    hub = InteractionEventHub()
    assert hub.latest_seq == 0
    assert hub.publish(_event()) == 1
    assert hub.publish(_event()) == 2
    assert hub.latest_seq == 2


def test_replay_ring_evicts_oldest_beyond_buffer_size() -> None:
    hub = InteractionEventHub()
    for _ in range(BUFFER_SIZE + 5):
        hub.publish(_event())
    frames, expired = hub.replay_since(5)
    assert expired is False
    assert len(frames) == BUFFER_SIZE
    assert [seq for seq, _ in frames] == list(range(6, BUFFER_SIZE + 6))
    seqs = [seq for seq, _ in frames]
    assert all(a < b for a, b in zip(seqs, seqs[1:]))


def test_replay_within_window_returns_newer_events() -> None:
    hub = InteractionEventHub()
    for _ in range(5):
        hub.publish(_event())
    frames, expired = hub.replay_since(2)
    assert expired is False
    assert [seq for seq, _ in frames] == [3, 4, 5]


def test_replay_too_old_requests_resync() -> None:
    hub = InteractionEventHub(buffer_size=4)
    for _ in range(6):
        hub.publish(_event())
    frames, expired = hub.replay_since(0)
    assert frames == []
    assert expired is True


def test_replay_at_tip_returns_empty_without_resync() -> None:
    hub = InteractionEventHub()
    for _ in range(3):
        hub.publish(_event())
    frames, expired = hub.replay_since(3)
    assert frames == []
    assert expired is False
    frames, expired = hub.replay_since(99)
    assert frames == []
    assert expired is False


def test_replay_without_last_event_id_returns_empty() -> None:
    hub = InteractionEventHub()
    hub.publish(_event())
    frames, expired = hub.replay_since(None)
    assert frames == []
    assert expired is False


def test_replay_on_empty_hub_returns_empty() -> None:
    hub = InteractionEventHub()
    frames, expired = hub.replay_since(0)
    assert frames == []
    assert expired is False


def test_subscriber_receives_live_events_in_order() -> None:
    hub = InteractionEventHub()
    sub_id, buf = hub.subscribe()
    try:
        hub.publish(_event("a"))
        hub.publish(_event("b"))
        got = [buf.get_nowait()[0], buf.get_nowait()[0]]
        assert got == [1, 2]
        assert buf.empty()
    finally:
        assert hub.unsubscribe(sub_id) is True
        assert hub.unsubscribe(sub_id) is False


def test_subscriber_overflow_drops_oldest_and_flags_resync() -> None:
    hub = InteractionEventHub(queue_size=2)
    sub_id, buf = hub.subscribe()
    try:
        hub.publish(_event("a"))
        hub.publish(_event("b"))
        hub.publish(_event("c"))
        assert hub.consume_resync(sub_id) is True
        assert hub.consume_resync(sub_id) is False
        got = [buf.get_nowait()[0], buf.get_nowait()[0]]
        assert got == [2, 3]
    finally:
        hub.unsubscribe(sub_id)


def test_publish_never_blocks_or_raises_on_full_subscriber() -> None:
    hub = InteractionEventHub(queue_size=1)
    sub_id, _ = hub.subscribe()
    try:
        for _ in range(50):
            hub.publish(_event())
        assert hub.latest_seq == 50
    finally:
        hub.unsubscribe(sub_id)


def test_attach_registers_both_channels_and_detach_cleans_up() -> None:
    hub = InteractionEventHub()
    service = _FakeService()
    hub.attach(service)
    assert len(service.integration) == 1
    assert len(service.contexts) == 1
    before_integration = len(service.integration)
    hub.attach(service)
    assert len(service.integration) == before_integration
    hub.detach()
    assert service.integration == {}
    assert service.contexts == {}
    hub.detach()


def test_integration_channel_fans_out_to_subscriber() -> None:
    hub = InteractionEventHub()
    service = _FakeService()
    hub.attach(service)
    sub_id, buf = hub.subscribe()
    try:
        callback = next(iter(service.integration.values()))
        callback(_event("live"))
        seq, received = buf.get_nowait()
        assert seq == 1
        assert received.event_type == "session_started"
    finally:
        hub.unsubscribe(sub_id)
        hub.detach()


def test_context_channel_produces_sanitized_context_updated() -> None:
    hub = InteractionEventHub()
    service = _FakeService()
    hub.attach(service)
    sub_id, buf = hub.subscribe()
    try:
        callback = next(iter(service.contexts.values()))
        callback(InteractionContextDTO(session_id="s-9", state="active"))
        seq, received = buf.get_nowait()
        assert seq == 1
        assert isinstance(received, InteractionContextUpdatedIntegrationEvent)
        assert received.event_type == "context_updated"
        assert received.context.session_id == "s-9"
    finally:
        hub.unsubscribe(sub_id)
        hub.detach()


def test_sse_frames_carry_no_internal_or_biometric_fields() -> None:
    ctx = InteractionContextDTO(session_id="s-1", state="active")
    rendered = (
        format_event_frame(7, _event("s-1"))
        + format_context_frame(ctx)
        + format_resync_frame(ctx)
        + format_retry_frame()
        + format_heartbeat_frame()
    )
    lowered = rendered.lower()
    for forbidden in (
        "track_id",
        "frame_id",
        "embedding",
        "bounding",
        "landmark",
        "crop",
        "bbox",
    ):
        assert forbidden not in lowered
    data_line = [ln for ln in rendered.splitlines() if ln.startswith("data:")][0]
    payload = json.loads(data_line[len("data:") :])
    assert payload["event_type"] == "session_started"
    assert "track_id" not in payload
    assert "frame_id" not in payload


def test_concurrent_publishers_yield_strictly_increasing_sequences() -> None:
    hub = InteractionEventHub()
    sub_id, buf = hub.subscribe()
    threads = [
        threading.Thread(target=lambda: [hub.publish(_event()) for _ in range(25)])
        for _ in range(8)
    ]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert hub.latest_seq == 200
        seqs = [buf.get_nowait()[0] for _ in range(200)]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == 200
    finally:
        hub.unsubscribe(sub_id)


def test_subscriber_count_tracks_connections() -> None:
    hub = InteractionEventHub()
    assert hub.subscriber_count == 0
    first, _ = hub.subscribe()
    second, _ = hub.subscribe()
    assert hub.subscriber_count == 2
    hub.unsubscribe(first)
    assert hub.subscriber_count == 1
    hub.unsubscribe(second)
    assert hub.subscriber_count == 0
