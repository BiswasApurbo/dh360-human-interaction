"""Edge heartbeat and automatic reconnection client for KBCON / DH360.

Periodically reports edge telemetry and online presence to the central
dashboard API, handles temporary network and Wi-Fi drops with exponential
backoff, avoids duplicate processes or resource leaks, and synchronizes
configuration state upon reconnection.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _read_cpu_temperature() -> float:
    """Read the Raspberry Pi CPU temperature in Celsius."""
    try:
        thermal_file = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(thermal_file):
            with open(thermal_file, encoding="utf-8") as f:
                return round(float(f.read().strip()) / 1000.0, 1)
    except Exception:
        pass
    return 45.0


class EdgeHeartbeatClient:
    """Manages resilient edge-to-cloud heartbeats with exponential backoff."""

    def __init__(
        self,
        robot_id: str | None = None,
        central_url: str | None = None,
        interval_seconds: float = 15.0,
        min_backoff_seconds: float = 3.0,
        max_backoff_seconds: float = 20.0,
        on_config_synced: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not robot_id:
            try:
                from dh360_hi.backend.robot_registry import RobotRegistry

                resolved = RobotRegistry.get_instance().resolve_local_robot_id()
            except Exception:
                resolved = (
                    os.environ.get("ROBOT_ID") or os.environ.get("KBCON_ROBOT_ID") or "Kb-001"
                )
        else:
            resolved = robot_id

        self.robot_id = resolved
        self.central_url = (
            central_url
            or os.environ.get("KBCON_CENTRAL_URL")
            or "https://kbcon-dashboard-gamma.vercel.app"
        ).rstrip("/")
        self.interval_seconds = max(5.0, interval_seconds)
        self.min_backoff = min_backoff_seconds
        self.max_backoff = max_backoff_seconds
        self.on_config_synced = on_config_synced

        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._current_backoff = self.min_backoff
        self._was_offline = False
        self._synced_config: dict[str, Any] = {}
        self._consecutive_failures = 0

    @property
    def is_running(self) -> bool:
        """Return whether the heartbeat loop is actively running."""
        return self._running and self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the background heartbeat task, preventing duplicate loops."""
        if self.is_running:
            logger.debug("EdgeHeartbeatClient is already running; skipping duplicate start.")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="kbcon_edge_heartbeat")
        logger.info(
            "EdgeHeartbeatClient started for robot '%s' -> %s (interval: %.1fs)",
            self.robot_id,
            self.central_url,
            self.interval_seconds,
        )

    def stop(self) -> None:
        """Stop the background heartbeat task cleanly."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
        logger.info("EdgeHeartbeatClient stopped.")

    async def _send_heartbeat(self, client: httpx.AsyncClient) -> bool:
        """Send a single heartbeat request to the central dashboard."""
        # Check if local device identity exists and update robot_id dynamically
        headers: dict[str, str] = {}
        try:
            from dh360_hi.device.identity import DeviceIdentity

            dev = DeviceIdentity.load_or_create()
            if dev.is_paired:
                headers["X-Device-ID"] = dev.device_id
                if dev.config.device_token:
                    headers["Authorization"] = f"Bearer {dev.config.device_token}"
                if dev.config.paired_robot_id:
                    self.robot_id = dev.config.paired_robot_id
        except Exception:
            pass

        if self.robot_id == "UNPROVISIONED":
            logger.debug("Device is UNPROVISIONED; awaiting pairing before heartbeating.")
            return False

        url = f"{self.central_url}/api/robot/heartbeat"
        temp = _read_cpu_temperature()
        payload: dict[str, Any] = {
            "robotId": self.robot_id,
            "telemetry": {
                "battery": 100,
                "temperature": temp,
                "activity": "active",
                "errorCode": None,
            },
            "firmwareVersion": "1.0.0",
        }
        if "X-Device-ID" in headers:
            payload["deviceId"] = headers["X-Device-ID"]

        try:
            resp = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
            if resp.status_code == 200:
                data = resp.json()
                if self._was_offline:
                    logger.info("Reconnected to central dashboard after network interruption!")
                    self._was_offline = False

                self._current_backoff = self.min_backoff

                # Synchronize authoritative tenant if provided
                if isinstance(data, dict):
                    remote_tenant = (
                        data.get("tenantName") or data.get("tenant") or data.get("tenantId")
                    )
                    if remote_tenant:
                        try:
                            from dh360_hi.device.identity import DeviceIdentity

                            dev_ident = DeviceIdentity.load_or_create()
                            if dev_ident.config.tenant_id != remote_tenant:
                                dev_ident.config.tenant_id = remote_tenant
                                dev_ident.save()
                                logger.info(
                                    "EdgeHeartbeatClient synchronized authoritative tenant: %s",
                                    remote_tenant,
                                )
                        except Exception:
                            pass

                # Synchronize remote config if provided
                if self.on_config_synced and isinstance(data, dict):
                    remote_config = data.get("config", {})
                    if remote_config != self._synced_config:
                        self._synced_config = remote_config
                        self.on_config_synced(data)

                logger.info(
                    "Heartbeat acknowledged by central dashboard (robot '%s', tenant '%s')",
                    self.robot_id,
                    data.get("tenantName") or data.get("tenant")
                    if isinstance(data, dict)
                    else "unknown",
                )
                return True

            logger.warning(
                "Heartbeat rejected by central server (status %d): %s",
                resp.status_code,
                resp.text[:200],
            )
            return False

        except (httpx.RequestError, httpx.TimeoutException) as exc:
            if not self._was_offline:
                logger.warning(
                    "Network disconnected or central server unreachable (%s). Entering retry loop...",
                    exc,
                )
                self._was_offline = True
            return False
        except Exception as exc:
            logger.error("Unexpected error in heartbeat transmission: %s", exc, exc_info=True)
            return False

    async def _run_loop(self) -> None:
        """Continuous heartbeat and reconnection loop with resilient client refresh and backoff."""
        while self._running:
            try:
                # Open fresh HTTP client session to discard broken sockets on network interruptions
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                    while self._running:
                        if self.robot_id == "UNPROVISIONED":
                            await asyncio.sleep(self.interval_seconds)
                            continue

                        success = await self._send_heartbeat(client)
                        if not self._running:
                            break

                        if success:
                            self._consecutive_failures = 0
                            self._current_backoff = self.min_backoff
                            await asyncio.sleep(self.interval_seconds)
                        else:
                            self._consecutive_failures += 1
                            # Break inner loop so client and its socket pool are completely refreshed
                            logger.info(
                                "Heartbeat attempt failed (streak: %d). Refreshing HTTP client after %.1fs backoff...",
                                self._consecutive_failures,
                                self._current_backoff,
                            )
                            await asyncio.sleep(self._current_backoff)
                            self._current_backoff = min(
                                self._current_backoff * 1.5,
                                self.max_backoff,
                            )
                            break
            except asyncio.CancelledError:
                break
            except Exception as loop_exc:
                logger.error("Error in heartbeat loop; refreshing client in 3s: %s", loop_exc)
                await asyncio.sleep(3.0)
