"""KBCON Device Provisioning and Registration Client.

Handles device registration requests, Ed25519 identity verification,
and server-authoritative configuration synchronization without temporary codes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import socket
import subprocess
from typing import Any

import httpx

from dh360_hi.device.identity import DeviceIdentity

logger = logging.getLogger(__name__)

DEFAULT_CENTRAL_URL = os.environ.get(
    "KBCON_CENTRAL_URL", "https://kbcon-dashboard-client.vercel.app"
).rstrip("/")


def get_key_fingerprint(public_pem: str) -> str:
    """Return SHA256 hexadecimal fingerprint of public key PEM."""
    digest = hashlib.sha256(public_pem.encode("utf-8")).hexdigest()
    return ":".join(digest[i : i + 2] for i in range(0, 32, 2))


def get_device_telemetry_info() -> dict[str, Any]:
    """Collect hardware and system telemetry metadata for device registration."""
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "pythonVersion": platform.python_version(),
    }

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            info["localIp"] = s.getsockname()[0]
    except Exception:
        info["localIp"] = "127.0.0.1"

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


class ProvisioningError(Exception):
    """Raised when device registration or synchronization fails."""


def register_device(
    identity: DeviceIdentity,
    server_url: str | None = None,
) -> dict[str, Any]:
    """Register device's permanent Ed25519 identity with the KBCON Device Registry."""
    base_url = (server_url or DEFAULT_CENTRAL_URL).rstrip("/")
    url = f"{base_url}/api/device/register"

    pub_pem = identity.get_public_key_pem()
    fingerprint = get_key_fingerprint(pub_pem)

    payload = {
        "deviceId": identity.device_id,
        "publicKey": pub_pem,
        "fingerprint": fingerprint,
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
                raise ProvisioningError(
                    f"Server rejected registration request ({resp.status_code}): {err_msg}"
                )

            data = resp.json()
            # If server indicates device is already bound to a robot, synchronize local configuration
            if data.get("bound") and data.get("robotId"):
                robot_id = data["robotId"]
                tenant_name = data.get("tenantName") or data.get("tenant") or "Salute"
                tenant_id = data.get("tenantId") or tenant_name
                robot_config = data.get("config") or {
                    "robotId": robot_id,
                    "name": data.get("robotName") or f"KBCON Robot {robot_id}",
                    "language": data.get("language") or "bn",
                    "mode": data.get("mode") or "education",
                }
                device_token = (
                    data.get("deviceToken")
                    or identity.config.device_token
                    or hashlib.sha256(f"{identity.device_id}:{robot_id}".encode()).hexdigest()
                )
                identity.save_pairing(
                    robot_id=robot_id,
                    tenant_id=tenant_id,
                    device_token=device_token,
                    robot_config=robot_config,
                )

            return data
    except httpx.RequestError as exc:
        raise ProvisioningError(f"Cannot reach central server at {base_url}: {exc}") from exc


def sync_authoritative_config(
    identity: DeviceIdentity,
    server_url: str | None = None,
) -> dict[str, Any]:
    """Query the central server to synchronize authoritative robot binding and tenant configuration."""
    return register_device(identity, server_url=server_url)


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
