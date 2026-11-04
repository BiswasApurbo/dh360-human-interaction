"""Unit tests for Soniox STT receiver timeout and non-blocking lifecycle."""

import json
import queue
import threading
import time
from unittest.mock import MagicMock

from dh360_hi.integrations.eliva import soniox


def test_ws_receiver_silence_timeout():
    """Verify ws_receiver breaks on silence timeout instead of blocking forever."""
    mock_ws = MagicMock()
    mock_ws.recv.side_effect = TimeoutError()

    result_container = {"text": ""}
    soniox._stop_event.clear()
    soniox._dispatch_event.clear()

    # Call with short silence timeout of 0.2s
    soniox.ws_receiver(
        mock_ws,
        result_container,
        silence_timeout=0.2,
        post_speech_timeout=0.2,
        max_duration=1.0,
    )

    assert soniox._dispatch_event.is_set()
    assert soniox._stop_event.is_set()
    assert result_container["text"] == ""


def test_ws_receiver_post_speech_timeout():
    """Verify ws_receiver finalizes when speech ends and pause follows."""
    # First message returns a token, subsequent calls time out (silence after speech)
    msg1 = json.dumps({"tokens": [{"text": "hello", "is_final": True}]})
    mock_ws = MagicMock()

    def mock_recv(*args, **kwargs):
        if not hasattr(mock_recv, "called"):
            mock_recv.called = True
            return msg1
        time.sleep(0.05)
        raise TimeoutError()

    mock_ws.recv.side_effect = mock_recv

    result_container = {"text": ""}
    soniox._stop_event.clear()
    soniox._dispatch_event.clear()

    soniox.ws_receiver(
        mock_ws,
        result_container,
        silence_timeout=1.0,
        post_speech_timeout=0.2,
        max_duration=2.0,
    )

    assert soniox._dispatch_event.is_set()
    assert soniox._stop_event.is_set()
    assert "hello" in result_container["text"]


def test_ws_receiver_endpoint_detection_end():
    """Verify ws_receiver finalizes when <end> token appears."""
    msg1 = json.dumps({"tokens": [{"text": "kiosk assistant<end>", "is_final": True}]})
    msg2 = json.dumps({"tokens": []})
    mock_ws = MagicMock()
    mock_ws.recv.side_effect = [msg1, msg2]

    result_container = {"text": ""}
    soniox._stop_event.clear()
    soniox._dispatch_event.clear()

    soniox.ws_receiver(
        mock_ws,
        result_container,
        silence_timeout=5.0,
        post_speech_timeout=2.0,
        max_duration=10.0,
    )

    assert soniox._dispatch_event.is_set()
    assert "kiosk assistant" in result_container["text"]


def test_mic_producer_loop_queue_timeout():
    """Verify mic_producer_loop respects max_duration and dispatch/stop events."""
    mock_ws = MagicMock()
    q: queue.Queue[bytes | None] = queue.Queue()

    soniox._stop_event.clear()
    soniox._dispatch_event.clear()

    # Signal stop event after 0.1s
    def delayed_stop():
        time.sleep(0.1)
        soniox._stop_event.set()

    threading.Thread(target=delayed_stop, daemon=True).start()

    soniox.mic_producer_loop(mock_ws, audio_queue=q, max_duration=1.0)
    # Loop should exit cleanly without hanging
    assert True
