"""Unit tests for KBCON Device Identity, Provisioning, and Pairing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from dh360_hi.backend.robot_registry import RobotRegistry
from dh360_hi.device.identity import DeviceIdentity
from dh360_hi.device.provisioning import (
    get_device_telemetry_info,
)


def test_device_identity_generation_and_keys(tmp_path: Path) -> None:
    """DeviceIdentity generates UUID starting with dev_, creates Ed25519 keys, and persists securely."""
    identity = DeviceIdentity.load_or_create(storage_dir=tmp_path)

    assert identity.device_id.startswith("dev_")
    assert len(identity.device_id) > 10
    assert identity.is_paired is False

    # Check files created on disk
    assert (tmp_path / "device_id").is_file()
    assert (tmp_path / "private_key.pem").is_file()
    assert (tmp_path / "public_key.pem").is_file()
    assert (tmp_path / "device_config.json").is_file()

    # Permissions on private key must be 0600
    priv_stat = (tmp_path / "private_key.pem").stat().st_mode & 0o777
    assert priv_stat == 0o600

    # Public key PEM format
    pub_pem = identity.get_public_key_pem()
    assert "-----BEGIN PUBLIC KEY-----" in pub_pem
    assert "-----END PUBLIC KEY-----" in pub_pem

    # Digital signature test
    payload = b"kbcon_challenge_payload_12345"
    signature = identity.sign(payload)
    assert isinstance(signature, str)
    assert len(signature) > 20

    # Verification must succeed with correct payload
    assert identity.verify(payload, signature) is True
    # Verification must fail with tampered payload
    assert identity.verify(b"tampered_payload", signature) is False


def test_device_identity_reload(tmp_path: Path) -> None:
    """Reloading identity from disk reuses the same device ID and keypair."""
    first = DeviceIdentity.load_or_create(storage_dir=tmp_path)
    orig_id = first.device_id
    orig_pem = first.get_public_key_pem()

    second = DeviceIdentity.load_or_create(storage_dir=tmp_path)
    assert second.device_id == orig_id
    assert second.get_public_key_pem() == orig_pem


def test_device_pairing_lifecycle(tmp_path: Path) -> None:
    """Test pairing saving, inspection, and revocation."""
    identity = DeviceIdentity.load_or_create(storage_dir=tmp_path)
    assert identity.is_paired is False

    identity.save_pairing(
        robot_id="KB-002",
        tenant_id="tenant_hospital_12",
        device_token="sec_token_abcdef123456",
        robot_config={"name": "KBCON Hospital Rover", "language": "en", "mode": "healthcare"},
    )

    assert identity.is_paired is True
    assert identity.config.status == "paired"
    assert identity.config.paired_robot_id == "KB-002"
    assert identity.config.tenant_id == "tenant_hospital_12"
    assert identity.config.device_token == "sec_token_abcdef123456"

    # Reload from disk
    reloaded = DeviceIdentity.load_or_create(storage_dir=tmp_path)
    assert reloaded.is_paired is True
    assert reloaded.config.paired_robot_id == "KB-002"
    assert reloaded.config.robot_config["mode"] == "healthcare"

    # Revoke
    identity.revoke_pairing()
    assert identity.is_paired is False
    assert identity.config.status == "unpaired"
    assert identity.config.paired_robot_id is None


def test_robot_registry_unprovisioned_fallback(tmp_path: Path) -> None:
    """An unprovisioned machine without configs must resolve to UNPROVISIONED, never defaulting to Kb-001."""
    with (
        patch("dh360_hi.backend.robot_registry.ROBOT_ID_FILE_CANDIDATES", [tmp_path / "nonexistent.json"]),
        patch("dh360_hi.backend.robot_registry.DEVICE_CONFIG_CANDIDATES", [tmp_path / "no_device.json"]),
        patch("dh360_hi.backend.robot_registry.OFFLINE_CONFIG_CACHE_CANDIDATES", [tmp_path / "no_cache.json"]),
        patch.dict("os.environ", {}, clear=True),
    ):
        registry = RobotRegistry(default_robot_id=None)
        local_id = registry.resolve_local_robot_id()
        assert local_id == "UNPROVISIONED"

        rec = registry.resolve_authoritative_config(local_id)
        assert rec.robot_id == "UNPROVISIONED"
        assert rec.status == "unprovisioned"


def test_robot_registry_detects_paired_device(tmp_path: Path) -> None:
    """When device_config.json has status='paired', RobotRegistry automatically adopts the paired robot ID."""
    cfg_file = tmp_path / "device_config.json"
    cfg_file.write_text(
        json.dumps({
            "device_id": "dev_test_99",
            "status": "paired",
            "paired_robot_id": "KB-042",
        }),
        encoding="utf-8",
    )

    with (
        patch("dh360_hi.backend.robot_registry.DEVICE_CONFIG_CANDIDATES", [cfg_file]),
        patch.dict("os.environ", {}, clear=True),
    ):
        registry = RobotRegistry(default_robot_id=None)
        assert registry.resolve_local_robot_id() == "KB-042"


def test_get_device_telemetry_info() -> None:
    """Telemetry collection returns hostname, architecture, os, and local IP."""
    info = get_device_telemetry_info()
    assert "hostname" in info
    assert "architecture" in info
    assert "os" in info
    assert "localIp" in info
