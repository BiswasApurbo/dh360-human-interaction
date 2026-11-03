"""KBCON Device Provisioning and Pairing Client.

Handles device registration requests, interactive pairing flow, cryptographic
challenge signing, and server-authoritative configuration synchronization.
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import subprocess
import time
from collections.abc import Callable
from typing import Any

import httpx

from dh360_hi.device.identity import DeviceIdentity

logger = logging.getLogger(__name__)

DEFAULT_CENTRAL_URL = os.environ.get(
    "KBCON_CENTRAL_URL", "https://kbcon-dashboard-gamma.vercel.app"
).rstrip("/")


def get_device_telemetry_info() -> dict[str, Any]:
    """Collect hardware and system telemetry metadata for device pairing."""
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "pythonVersion": platform.python_version(),
    }

    # Best-effort primary local IP address resolution
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            info["localIp"] = s.getsockname()[0]
    except Exception:
        info["localIp"] = "127.0.0.1"

    # Best-effort Raspberry Pi model detection
    try:
        model_path = "/proc/device-tree/model"
        if os.path.exists(model_path):
            with open(model_path, encoding="utf-8") as f:
                info["hardwareModel"] = f.read().strip().replace("\x00", "")
        else:
            info["hardwareModel"] = platform.processor() or "Generic Linux / ARM"
    except Exception:
        info["hardwareModel"] = "Unknown"

    return info


class PairingError(Exception):
    """Raised when device pairing fails or is rejected."""


class PairingExpiredError(PairingError):
    """Raised when a pairing session exceeds its time-to-live."""


def request_pairing_session(
    identity: DeviceIdentity,
    server_url: str | None = None,
) -> dict[str, Any]:
    """Request a temporary pairing code and challenge session from central server."""
    base_url = (server_url or DEFAULT_CENTRAL_URL).rstrip("/")
    url = f"{base_url}/api/device/pairing/request"

    payload = {
        "deviceId": identity.device_id,
        "publicKey": identity.get_public_key_pem(),
        "deviceInfo": get_device_telemetry_info(),
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                err_msg = resp.text
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("message") or err_json.get("error") or err_msg
                except Exception:
                    pass
                raise PairingError(f"Server rejected pairing request ({resp.status_code}): {err_msg}")

            data = resp.json()
            if not data.get("code") or not data.get("sessionId"):
                raise PairingError(f"Invalid server response: missing code or sessionId: {data}")
            return data
    except httpx.RequestError as exc:
        raise PairingError(f"Cannot reach central server at {base_url}: {exc}") from exc


def poll_pairing_approval(
    identity: DeviceIdentity,
    session_id: str,
    challenge: str,
    server_url: str | None = None,
    timeout_seconds: int = 600,
    poll_interval: float = 3.0,
    on_tick: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Poll pairing session status until approved, expired, or timed out."""
    base_url = (server_url or DEFAULT_CENTRAL_URL).rstrip("/")
    url = f"{base_url}/api/device/pairing/status"

    # Pre-sign the challenge with device private key
    signature = identity.sign(challenge.encode("utf-8"))
    headers = {
        "X-Device-ID": identity.device_id,
        "X-Device-Signature": signature,
    }
    params = {
        "sessionId": session_id,
        "deviceId": identity.device_id,
    }

    start_time = time.time()
    elapsed = 0

    with httpx.Client(timeout=10.0) as client:
        while elapsed < timeout_seconds:
            try:
                resp = client.get(url, params=params, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status")

                    if status == "approved":
                        robot_id = data.get("robotId")
                        tenant_id = data.get("tenantId")
                        device_token = data.get("deviceToken")
                        robot_config = data.get("robotConfig") or {}

                        if not robot_id or not device_token:
                            raise PairingError(
                                f"Approved session missing robotId or deviceToken: {data}"
                            )

                        identity.save_pairing(
                            robot_id=robot_id,
                            tenant_id=tenant_id,
                            device_token=device_token,
                            robot_config=robot_config,
                        )
                        return data

                    if status in ("expired", "rejected", "revoked"):
                        raise PairingExpiredError(
                            f"Pairing session ended with status: {status}"
                        )

                elif resp.status_code in (404, 410):
                    raise PairingExpiredError("Pairing session expired or not found.")

            except (httpx.RequestError, httpx.TimeoutException) as exc:
                logger.debug("Network error during pairing polling (will retry): %s", exc)

            time.sleep(poll_interval)
            elapsed = int(time.time() - start_time)
            if on_tick:
                on_tick(elapsed)

    raise PairingExpiredError(f"Pairing timed out after {timeout_seconds} seconds.")


def restart_backend_service() -> bool:
    """Attempt to restart dh360-backend systemd service if running on Linux."""
    try:
        res = subprocess.run(
            ["systemctl", "restart", "dh360-backend.service"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if res.returncode == 0:
            logger.info("Successfully restarted dh360-backend.service")
            return True
        logger.warning("Could not restart dh360-backend.service: %s", res.stderr)
    except Exception as exc:
        logger.debug("systemctl restart dh360-backend not available: %s", exc)
    return False
