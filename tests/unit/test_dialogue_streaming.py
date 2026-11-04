"""Unit tests for low-latency dialogue streaming (SSE) pipeline."""

import pytest

from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.contracts.dialogue import MockLLMProvider
from dh360_hi.edge.event_bus import EventBus


@pytest.fixture
def streaming_coordinator():
    bus = EventBus()
    ks = KnowledgeService()
    llm = MockLLMProvider()
    coord = DialogueCoordinator(
        knowledge_service=ks,
        llm_provider=llm,
        stt_provider=None,
        tts_provider=None,
        event_bus=bus,
    )
    yield coord
    coord.close()


def test_handle_turn_stream_sync(streaming_coordinator):
    events = list(
        streaming_coordinator.handle_turn_stream(
            user_text="What is the weather today?",
            language="en",
        )
    )
    assert len(events) >= 3

    event_names = [e["event"] for e in events]
    assert event_names[0] == "metadata"
    assert "token" in event_names
    assert event_names[-1] == "done"

    metadata = events[0]["data"]
    assert metadata["user_text"] == "What is the weather today?"
    assert metadata["language"] == "en"

    # Assemble tokens
    tokens = [e["data"]["token"] for e in events if e["event"] == "token"]
    assembled = "".join(tokens).strip()
    done_text = events[-1]["data"]["text"].strip()
    assert assembled == done_text
    assert "weather" in done_text.lower()


def test_handle_turn_stream_direct_inquiry(streaming_coordinator):
    # Set llm_provider to None to verify direct deterministic resolution without external LLM
    streaming_coordinator._llm_provider = None
    events = list(
        streaming_coordinator.handle_turn_stream(
            user_text="Who are you?",
            language="en",
        )
    )
    assert len(events) >= 3
    tokens = [e["data"]["token"] for e in events if e["event"] == "token"]
    assembled = "".join(tokens).strip()
    assert "KBCON" in assembled


@pytest.mark.anyio
async def test_handle_turn_stream_async(streaming_coordinator):
    received_events = []
    async for event in streaming_coordinator.handle_turn_stream_async(
        user_text="কেমন আছেন?",
        language="bn",
    ):
        received_events.append(event)

    assert len(received_events) >= 3
    assert received_events[0]["event"] == "metadata"
    assert received_events[-1]["event"] == "done"
    done_data = received_events[-1]["data"]
    assert done_data["role"] == "model"
    assert len(done_data["text"]) > 0


def test_handle_turn_stream_empty_utterance(streaming_coordinator):
    events = list(
        streaming_coordinator.handle_turn_stream(
            user_text="",
            language="bn",
        )
    )
    assert len(events) >= 2
    assert events[0]["event"] == "error"
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["metadata"]["error"] == "EMPTY_UTTERANCE"
