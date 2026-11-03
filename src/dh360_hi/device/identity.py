"""KBCON Device Identity, Cryptographic Keys, and Secure Configuration Management.

Provides cryptographically random device UUID generation, asymmetric Ed25519
keypair generation, digital signature signing and verification, and secure local
storage with restrictive permissions (0600/0700).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger(__name__)

# Primary system-wide and user-fallback device configuration directories
PRIMARY_DEVICE_DIR = Path("/etc/kbcon/device")
FALLBACK_DEVICE_DIR = Path.home() / ".config" / "dh360" / "device"


def resolve_device_storage_dir(preferred_dir: Path | None = None) -> Path:
    """Determine the writable storage directory for device identity and keys."""
    if preferred_dir is not None:
        return preferred_dir

    # Check if primary system path (/etc/kbcon/device) is usable or can be created
    if PRIMARY_DEVICE_DIR.exists() and os.access(PRIMARY_DEVICE_DIR, os.W_OK):
        return PRIMARY_DEVICE_DIR

    try:
        if os.geteuid() == 0:  # Running as root
            PRIMARY_DEVICE_DIR.mkdir(parents=True, exist_ok=True)
            PRIMARY_DEVICE_DIR.chmod(0o700)
            return PRIMARY_DEVICE_DIR
    except Exception:
        pass

    # Fall back to user-scoped configuration directory
    FALLBACK_DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        FALLBACK_DEVICE_DIR.chmod(0o700)
    except Exception:
        pass
    return FALLBACK_DEVICE_DIR


@dataclass
class DeviceConfig:
    """Local device configuration state and pairing record."""

    device_id: str
    status: str = "unpaired"  # "unpaired" | "pending_pairing" | "paired" | "revoked"
    paired_robot_id: str | None = None
    tenant_id: str | None = None
    device_token: str | None = None
    paired_at: str | None = None
    robot_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceConfig:
        return cls(
            device_id=str(data.get("device_id") or data.get("deviceId") or ""),
            status=str(data.get("status") or "unpaired"),
            paired_robot_id=(
                str(data.get("paired_robot_id") or data.get("robot_id") or data.get("robotId") or "")
                or None
            ),
            tenant_id=(
                str(data.get("tenant_id") or data.get("tenantId") or "") or None
            ),
            device_token=(
                str(data.get("device_token") or data.get("deviceToken") or "") or None
            ),
            paired_at=data.get("paired_at") or data.get("pairedAt"),
            robot_config=data.get("robot_config") or data.get("robotConfig"),
        )


class DeviceIdentity:
    """Manages unique device identifier, Ed25519 keypair, and signed challenge proof."""

    def __init__(
        self,
        device_id: str,
        storage_dir: Path,
        private_key: ed25519.Ed25519PrivateKey | None = None,
        public_key: ed25519.Ed25519PublicKey | None = None,
        config: DeviceConfig | None = None,
    ) -> None:
        self.device_id = device_id
        self.storage_dir = storage_dir
        self._private_key = private_key
        self._public_key = public_key or (private_key.public_key() if private_key else None)
        self.config = config or DeviceConfig(device_id=device_id)

    @property
    def id_file(self) -> Path:
        return self.storage_dir / "device_id"

    @property
    def private_key_file(self) -> Path:
        return self.storage_dir / "private_key.pem"

    @property
    def public_key_file(self) -> Path:
        return self.storage_dir / "public_key.pem"

    @property
    def config_file(self) -> Path:
        return self.storage_dir / "device_config.json"

    @classmethod
    def load_or_create(cls, storage_dir: Path | None = None) -> DeviceIdentity:
        """Load an existing device identity from disk or generate a brand new one."""
        resolved_dir = resolve_device_storage_dir(storage_dir)
        id_path = resolved_dir / "device_id"
        priv_path = resolved_dir / "private_key.pem"
        cfg_path = resolved_dir / "device_config.json"

        # If existing files exist, load them
        if id_path.exists() and priv_path.exists():
            try:
                device_id = id_path.read_text(encoding="utf-8").strip()
                priv_bytes = priv_path.read_bytes()
                priv_key = serialization.load_pem_private_key(priv_bytes, password=None)
                if not isinstance(priv_key, ed25519.Ed25519PrivateKey):
                    raise ValueError("Existing private key is not Ed25519")

                config = None
                if cfg_path.exists():
                    try:
                        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
                        config = DeviceConfig.from_dict(cfg_data)
                    except Exception as err:
                        logger.warning("Failed to parse device_config.json: %s", err)

                return cls(
                    device_id=device_id,
                    storage_dir=resolved_dir,
                    private_key=priv_key,
                    config=config,
                )
            except Exception as exc:
                logger.warning("Could not load existing device identity at %s: %s", resolved_dir, exc)

        # Generate a new unique device identity
        new_id = f"dev_{uuid.uuid4().hex}"
        priv_key = ed25519.Ed25519PrivateKey.generate()
        instance = cls(
            device_id=new_id,
            storage_dir=resolved_dir,
            private_key=priv_key,
            config=DeviceConfig(device_id=new_id, status="unpaired"),
        )
        instance.save()
        return instance

    def save(self) -> None:
        """Persist device ID, asymmetric keypair, and config with secure 0600 permissions."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.storage_dir.chmod(0o700)
        except Exception:
            pass

        # 1. Save device ID
        self.id_file.write_text(self.device_id, encoding="utf-8")
        self.id_file.chmod(0o644)

        # 2. Save private key (0600)
        if self._private_key:
            priv_pem = self._private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            self.private_key_file.write_bytes(priv_pem)
            self.private_key_file.chmod(0o600)

        # 3. Save public key
        if self._public_key:
            pub_pem = self._public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            self.public_key_file.write_bytes(pub_pem)
            self.public_key_file.chmod(0o644)

        # 4. Save device_config.json
        self.config_file.write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        self.config_file.chmod(0o600)

    def get_public_key_pem(self) -> str:
        """Return public key formatted as PEM string."""
        if not self._public_key:
            raise ValueError("No public key available")
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

    def sign(self, data: bytes) -> str:
        """Sign binary data with device private key, returning base64-encoded signature."""
        if not self._private_key:
            raise ValueError("Private key is not loaded; cannot sign challenge")
        signature = self._private_key.sign(data)
        return base64.b64encode(signature).decode("utf-8")

    def verify(self, data: bytes, signature_b64: str) -> bool:
        """Verify signature using device public key."""
        if not self._public_key:
            return False
        try:
            sig_bytes = base64.b64decode(signature_b64)
            self._public_key.verify(sig_bytes, data)
            return True
        except (InvalidSignature, Exception):
            return False

    @property
    def is_paired(self) -> bool:
        """Return True if device is paired with an authoritative robot ID."""
        return (
            self.config.status == "paired"
            and bool(self.config.paired_robot_id)
            and bool(self.config.device_token)
        )

    def save_pairing(
        self,
        robot_id: str,
        tenant_id: str | None,
        device_token: str,
        robot_config: dict[str, Any] | None = None,
        paired_at: str | None = None,
    ) -> None:
        """Record successful pairing credentials and robot ownership."""
        from datetime import UTC, datetime

        self.config.status = "paired"
        self.config.paired_robot_id = robot_id
        self.config.tenant_id = tenant_id
        self.config.device_token = device_token
        self.config.paired_at = paired_at or datetime.now(UTC).isoformat()
        self.config.robot_config = robot_config or {}
        self.save()

    def revoke_pairing(self) -> None:
        """Clear pairing credentials and reset state to unpaired."""
        self.config.status = "unpaired"
        self.config.paired_robot_id = None
        self.config.tenant_id = None
        self.config.device_token = None
        self.config.paired_at = None
        self.config.robot_config = None
        self.save()
