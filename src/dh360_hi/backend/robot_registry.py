"""Robot identity, configuration, and authoritative ownership registry.

Manages the local physical robot identity, resolving authoritative configuration
from Firestore (when online) and persisting a local cache to ensure the robot
starts up cleanly in offline environments.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Standard locations for local robot identification and offline configuration cache
ROBOT_ID_FILE_CANDIDATES = [
    Path("/home/eliva/.config/dh360/robot.json"),
    Path.home() / ".config" / "dh360" / "robot.json",
]

OFFLINE_CONFIG_CACHE_CANDIDATES = [
    Path("/home/eliva/.config/dh360/robot_config.json"),
    Path.home() / ".config" / "dh360" / "robot_config.json",
]


@dataclass(frozen=True)
class RobotRecord:
    """Authoritative representation of a single robot's configuration and ownership."""

    robot_id: str
    name: str = "KBCON Humanoid"
    language: str = "bn"
    mode: str = "education"
    tenant_id: str | None = None
    assigned_admin_name: str | None = None
    status: str = "online"
    type: str = "Humanoid"
    last_synced: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotRecord:
        """Construct record from dictionary with safe field normalizations."""
        raw_lang = str(data.get("language") or "bn").strip().lower()
        norm_lang = "en" if raw_lang in ("en", "english") else "bn"

        raw_mode = str(data.get("mode") or "education").strip().lower()
        norm_mode = (
            "healthcare"
            if raw_mode in ("healthcare", "clinic", "hospital", "medical")
            else "education"
        )

        return cls(
            robot_id=str(data.get("id") or data.get("robot_id") or "Kb-001").strip(),
            name=str(data.get("name") or "KBCON Humanoid").strip(),
            language=norm_lang,
            mode=norm_mode,
            tenant_id=(
                str(data.get("assignedAdminId") or data.get("tenant_id") or "").strip() or None
            ),
            assigned_admin_name=(
                str(data.get("assignedAdminName") or data.get("assigned_admin_name") or "").strip()
                or None
            ),
            status=str(data.get("status") or "online").strip().lower(),
            type=str(data.get("type") or "Humanoid").strip(),
            last_synced=str(
                data.get("last_synced") or data.get("updatedAt") or datetime.now(UTC).isoformat()
            ),
        )


class RobotRegistry:
    """Thread-safe registry resolving and caching the active robot identity and config."""

    _instance: RobotRegistry | None = None

    def __init__(
        self,
        default_robot_id: str | None = None,
        firestore_client: Any | None = None,
    ) -> None:
        self._default_robot_id = default_robot_id
        self._firestore_client = firestore_client
        self._active_robot: RobotRecord | None = None

    @classmethod
    def get_instance(cls) -> RobotRegistry:
        """Singleton accessor for application-wide robot registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_instance(cls, instance: RobotRegistry) -> None:
        """Explicitly override the singleton instance (useful for testing)."""
        cls._instance = instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance to None (useful for testing)."""
        cls._instance = None

    def resolve_local_robot_id(self) -> str:
        """Determine the robot ID assigned to this physical device.

        Precedence:
        1. Explicit default / override
        2. Environment variable KBCON_ROBOT_ID or ROBOT_ID
        3. Local configuration file (~/.config/dh360/robot.json)
        4. Fallback default 'KB001'
        """
        if self._default_robot_id and self._default_robot_id.strip():
            return self._default_robot_id.strip()

        env_id = os.getenv("KBCON_ROBOT_ID") or os.getenv("ROBOT_ID")
        if env_id and env_id.strip():
            return env_id.strip()

        for path in ROBOT_ID_FILE_CANDIDATES:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    cand_id = data.get("robot_id") or data.get("id")
                    if cand_id and str(cand_id).strip():
                        return str(cand_id).strip()
                except Exception as exc:
                    logger.warning("Failed to parse local robot ID file at %s: %s", path, exc)

        for path in OFFLINE_CONFIG_CACHE_CANDIDATES:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    cand_id = data.get("robot_id") or data.get("id")
                    if cand_id and str(cand_id).strip():
                        return str(cand_id).strip()
                except Exception as exc:
                    logger.warning("Failed to parse offline config cache at %s: %s", path, exc)

        return "Kb-001"

    def _get_cache_path(self) -> Path:
        """Return the preferred path for offline configuration cache."""
        for path in OFFLINE_CONFIG_CACHE_CANDIDATES:
            if path.parent.exists():
                return path
        return OFFLINE_CONFIG_CACHE_CANDIDATES[-1]

    def _load_cached_config(self, robot_id: str) -> RobotRecord | None:
        """Attempt to read cached configuration from local disk."""
        for path in OFFLINE_CONFIG_CACHE_CANDIDATES:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if data.get("robot_id") == robot_id or data.get("id") == robot_id:
                        return RobotRecord.from_dict(data)
                except Exception as exc:
                    logger.warning("Could not read offline config cache at %s: %s", path, exc)
        return None

    def _save_cached_config(self, record: RobotRecord) -> None:
        """Persist authoritative configuration to local disk for offline resilience."""
        cache_path = self._get_cache_path()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
            cache_path.chmod(0o600)
            logger.info("Saved authoritative robot config to offline cache at %s", cache_path)
        except Exception as exc:
            logger.warning(
                "Could not persist offline robot config cache to %s: %s", cache_path, exc
            )

    def resolve_authoritative_config(self, robot_id: str | None = None) -> RobotRecord:
        """Resolve authoritative robot configuration from Firestore or offline fallback."""
        target_id = robot_id or self.resolve_local_robot_id()

        # 1. Attempt Firestore live lookup if credentials / client available
        db = self._firestore_client
        if db is None:
            try:
                from dh360_hi.api.routers.admin import get_firebase_firestore

                db = get_firebase_firestore()
            except Exception:
                db = None

        if db is not None:
            try:
                doc = db.collection("robots").document(target_id).get()
                if not doc.exists:
                    clean_target = target_id.replace("-", "").lower()
                    for d in db.collection("robots").stream():
                        if d.id.replace("-", "").lower() == clean_target:
                            doc = d
                            target_id = d.id
                            break

                if doc.exists:
                    raw_data = doc.to_dict() or {}
                    raw_data["id"] = target_id
                    record = RobotRecord.from_dict(raw_data)
                    self._save_cached_config(record)
                    self._active_robot = record
                    logger.info(
                        "Resolved authoritative robot config from Firestore for '%s': %s",
                        target_id,
                        record,
                    )
                    return record
                else:
                    logger.warning(
                        "Robot '%s' not found in Firestore collection 'robots'", target_id
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to query Firestore for robot '%s': %s (falling back to local cache)",
                    target_id,
                    exc,
                )

        # 2. Offline fallback: check local cached configuration
        cached = self._load_cached_config(target_id)
        if cached is not None:
            logger.info(
                "Using cached authoritative robot config for '%s' (offline mode): %s",
                target_id,
                cached,
            )
            self._active_robot = cached
            return cached

        # 3. Safe bootstrap default if brand new deployment
        logger.info("Initializing fallback robot record for '%s'", target_id)
        bootstrap = RobotRecord(
            robot_id=target_id,
            name=f"KBCON {target_id}",
            language="bn",
            mode="education",
            status="online",
            last_synced=datetime.now(UTC).isoformat(),
        )
        self._save_cached_config(bootstrap)
        self._active_robot = bootstrap
        return bootstrap

    def get_active_robot(self) -> RobotRecord:
        """Return the currently cached or resolved active robot record."""
        if self._active_robot is None:
            return self.resolve_authoritative_config()
        return self._active_robot

    def set_active_robot(self, record: RobotRecord) -> None:
        """Directly update the active robot record (for tests or dynamic reconfiguration)."""
        self._active_robot = record
        self._save_cached_config(record)
