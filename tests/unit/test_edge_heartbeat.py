"""Unit tests for EdgeHeartbeatClient resilience and reconnection."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dh360_hi.edge.heartbeat import EdgeHeartbeatClient


@pytest.mark.anyio
async def test_heartbeat_client_init_and_properties():
    client = EdgeHeartbeatClient(
        robot_id="test-bot",
        central_url="https://example.com/",
        interval_seconds=12.0,
        min_backoff_seconds=2.0,
        max_backoff_seconds=15.0,
    )
    assert client.robot_id == "test-bot"
    assert client.central_url == "https://example.com"
    assert client.interval_seconds == 12.0
    assert client.min_backoff == 2.0
    assert client.max_backoff == 15.0
    assert not client.is_running


@pytest.mark.anyio
async def test_heartbeat_client_start_and_stop():
    client = EdgeHeartbeatClient(
        robot_id="test-bot",
        central_url="https://example.com",
        interval_seconds=1.0,
    )
    with patch.object(client, "_send_heartbeat", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        client.start()
        assert client.is_running
        # Start again should be no-op
        client.start()
        await asyncio.sleep(0.05)
        client.stop()
        assert not client.is_running


@pytest.mark.anyio
async def test_send_heartbeat_success_and_config_sync():
    synced_configs = []

    def on_sync(data):
        synced_configs.append(data)

    client = EdgeHeartbeatClient(
        robot_id="test-bot",
        central_url="https://example.com",
        on_config_synced=on_sync,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "online",
        "tenantName": "Salute",
        "config": {"language": "bn"},
    }

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post.return_value = mock_resp

    with patch("dh360_hi.edge.heartbeat._read_cpu_temperature", return_value=42.5):
        success = await client._send_heartbeat(mock_http)

    assert success is True
    assert len(synced_configs) == 1
    assert synced_configs[0]["config"] == {"language": "bn"}


@pytest.mark.anyio
async def test_send_heartbeat_network_error_handled():
    client = EdgeHeartbeatClient(
        robot_id="test-bot",
        central_url="https://example.com",
    )

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post.side_effect = httpx.ConnectError("Connection refused")

    success = await client._send_heartbeat(mock_http)
    assert success is False
    assert client._was_offline is True


@pytest.mark.anyio
async def test_send_heartbeat_multi_url_broadcast():
    client = EdgeHeartbeatClient(
        robot_id="test-bot",
        central_url="https://site1.com,https://site2.com",
    )
    assert len(client.central_urls) == 2

    mock_resp1 = MagicMock()
    mock_resp1.status_code = 200
    mock_resp1.json.return_value = {"status": "online", "tenant": "Salute"}

    mock_resp2 = MagicMock()
    mock_resp2.status_code = 500
    mock_resp2.text = "Internal Server Error"

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    # site1 succeeds, site2 returns 500
    mock_http.post.side_effect = [mock_resp1, mock_resp2]

    with patch("dh360_hi.edge.heartbeat._read_cpu_temperature", return_value=40.0):
        success = await client._send_heartbeat(mock_http)

    assert success is True
    assert mock_http.post.call_count == 2

