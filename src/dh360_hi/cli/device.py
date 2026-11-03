"""KBCON Device Management CLI: kbcon-device.

Provides hardware device provisioning, Ed25519 identity initialization,
interactive pairing with the central dashboard, and pairing revocation.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from dh360_hi.device.identity import DeviceIdentity, resolve_device_storage_dir
from dh360_hi.device.provisioning import (
    DEFAULT_CENTRAL_URL,
    PairingError,
    PairingExpiredError,
    poll_pairing_approval,
    request_pairing_session,
    restart_backend_service,
)

logger = logging.getLogger(__name__)


def _get_key_fingerprint(public_pem: str) -> str:
    """Return SHA256 hexadecimal fingerprint of public key PEM."""
    digest = hashlib.sha256(public_pem.encode("utf-8")).hexdigest()
    return ":".join(digest[i : i + 2] for i in range(0, 32, 2))


def _check_service_status(service_name: str = "dh360-backend.service") -> str:
    """Check systemd service active status."""
    if not shutil.which("systemctl"):
        return "systemctl not available (non-systemd environment)"
    try:
        res = subprocess.run(
            ["systemctl", "is-active", service_name],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return res.stdout.strip() or "unknown"
    except Exception:
        return "error querying service"


def handle_status(args: argparse.Namespace) -> int:
    """Display current device identity, pairing status, and service health."""
    storage_dir = Path(args.dir) if getattr(args, "dir", None) else None
    resolved_dir = resolve_device_storage_dir(storage_dir)

    print("\n" + "=" * 62)
    print("              KBCON DEVICE STATUS & IDENTITY")
    print("=" * 62)
    print(f"Storage Directory: {resolved_dir}")

    id_file = resolved_dir / "device_id"
    if not id_file.exists():
        print("Device Identity:   [NOT INITIALIZED]")
        print("Run 'kbcon-device init' or 'kbcon-device pair' to create device identity.")
        print("=" * 62 + "\n")
        return 0

    identity = DeviceIdentity.load_or_create(resolved_dir)
    cfg = identity.config

    print(f"Device ID:         {identity.device_id}")
    try:
        pub_pem = identity.get_public_key_pem()
        print(f"Public Key SHA256: {_get_key_fingerprint(pub_pem)}")
    except Exception as exc:
        print(f"Public Key:        Error ({exc})")

    if cfg.status == "paired" and cfg.paired_robot_id:
        print(f"Pairing Status:    PAIRED -> Robot [{cfg.paired_robot_id}]")
        print(f"Assigned Tenant:   {cfg.tenant_id or 'None (System Default)'}")
        print(f"Paired At:         {cfg.paired_at or 'Unknown'}")
        if cfg.robot_config:
            name = cfg.robot_config.get("name", "Unknown")
            lang = cfg.robot_config.get("language", "Unknown")
            mode = cfg.robot_config.get("mode", "Unknown")
            print(f"Robot Info:        Name='{name}', Language='{lang}', Mode='{mode}'")
    elif cfg.status == "revoked":
        print("Pairing Status:    REVOKED (Device was unlinked by administrator)")
    else:
        print("Pairing Status:    UNPAIRED (Ready for pairing)")

    svc_status = _check_service_status()
    print(f"Backend Service:   {svc_status}")
    print("=" * 62 + "\n")
    return 0


def handle_init(args: argparse.Namespace) -> int:
    """Initialize or load device identity and cryptographic keys."""
    storage_dir = Path(args.dir) if getattr(args, "dir", None) else None
    identity = DeviceIdentity.load_or_create(storage_dir)
    pub_pem = identity.get_public_key_pem()

    print("\n" + "=" * 62)
    print("            KBCON DEVICE IDENTITY INITIALIZED")
    print("=" * 62)
    print(f"Device ID:         {identity.device_id}")
    print(f"Storage Directory: {identity.storage_dir}")
    print(f"Fingerprint:       {_get_key_fingerprint(pub_pem)}")
    print("Files created/verified:")
    print(f"  - {identity.id_file}")
    print(f"  - {identity.private_key_file} (mode 0600)")
    print(f"  - {identity.public_key_file}")
    print(f"  - {identity.config_file}")
    print("=" * 62 + "\n")
    return 0


def handle_pair(args: argparse.Namespace) -> int:
    """Request a temporary pairing code and wait for Super Admin approval in the dashboard."""
    server_url = getattr(args, "server", None) or DEFAULT_CENTRAL_URL
    storage_dir = Path(args.dir) if getattr(args, "dir", None) else None
    identity = DeviceIdentity.load_or_create(storage_dir)

    if identity.is_paired and not getattr(args, "force", False):
        print(f"\n[!] Device is ALREADY PAIRED to robot '{identity.config.paired_robot_id}'.")
        print("    Use 'kbcon-device pair --force' or 'kbcon-device unpair' first.\n")
        return 1

    print("\nConnecting to KBCON Central Platform...")
    try:
        session_data = request_pairing_session(identity, server_url=server_url)
    except PairingError as err:
        print(f"\n[ERROR] Pairing request failed: {err}\n", file=sys.stderr)
        return 1

    code = session_data["code"]
    session_id = session_data["sessionId"]
    challenge = session_data["challenge"]
    expires_at = session_data.get("expiresAt", "in 10 minutes")

    print("\n" + "=" * 64)
    print("                   KBCON DEVICE PAIRING")
    print("=" * 64)
    print(f" Device ID:     {identity.device_id}")
    print(f" PAIRING CODE:  \033[1;32m{code}\033[0m")
    print(f" Expiration:    {expires_at}")
    print("-" * 64)
    print(" INSTRUCTIONS:")
    print(" 1. Sign in to the KBCON Dashboard as Super Admin:")
    print(f"    {server_url}/robots")
    print(" 2. Click 'Pair Device' in the top right corner.")
    print(f" 3. Enter pairing code: {code}")
    print(" 4. Select the robot you wish to bind to this physical Raspberry Pi.")
    print("=" * 64)
    print("Waiting for Super Admin approval in dashboard... (Press Ctrl+C to cancel)\n")

    def on_tick(elapsed: int) -> None:
        mins, secs = divmod(elapsed, 60)
        sys.stdout.write(f"\r[Polling status...] Elapsed: {mins:02d}:{secs:02d} ")
        sys.stdout.flush()

    try:
        result = poll_pairing_approval(
            identity=identity,
            session_id=session_id,
            challenge=challenge,
            server_url=server_url,
            timeout_seconds=600,
            poll_interval=3.0,
            on_tick=on_tick,
        )
        print("\n\n" + "=" * 64)
        print("              [SUCCESS] DEVICE SUCCESSFULLY PAIRED!")
        print("=" * 64)
        print(f" Paired Robot ID: {result.get('robotId')}")
        print(f" Tenant ID:       {result.get('tenantId') or 'Default'}")
        if result.get("robotConfig"):
            r_cfg = result["robotConfig"]
            print(f" Robot Name:      {r_cfg.get('name')}")
            print(f" Language:        {r_cfg.get('language')}")
            print(f" Mode:            {r_cfg.get('mode')}")
        print("=" * 64)

        # Notify / restart backend service
        restarted = restart_backend_service()
        if restarted:
            print("Restarted dh360-backend.service with new robot identity.")
        else:
            print("Note: Restart 'dh360-backend.service' to reload robot identity immediately.")
        print()
        return 0

    except PairingExpiredError as err:
        print(f"\n\n[EXPIRED] {err}\n", file=sys.stderr)
        return 1
    except PairingError as err:
        print(f"\n\n[ERROR] Pairing failed: {err}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nPairing canceled by user.\n")
        return 130


def handle_unpair(args: argparse.Namespace) -> int:
    """Revoke local pairing configuration and reset device to unpaired."""
    storage_dir = Path(args.dir) if getattr(args, "dir", None) else None
    identity = DeviceIdentity.load_or_create(storage_dir)

    if not identity.is_paired:
        print("\nDevice is not currently paired.\n")
        return 0

    old_robot = identity.config.paired_robot_id
    identity.revoke_pairing()
    print(f"\n[OK] Device pairing for robot '{old_robot}' has been revoked.")
    print("Device is now in UNPAIRED status.\n")

    restart_backend_service()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for kbcon-device CLI."""
    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Custom device configuration directory (default: /etc/kbcon/device or ~/.config/dh360/device)",
    )

    parser = argparse.ArgumentParser(
        prog="kbcon-device",
        description="KBCON Raspberry Pi Device Provisioning and Pairing Utility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[common_parent],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = subparsers.add_parser(
        "status", parents=[common_parent], help="Show device identity and pairing status"
    )
    p_status.set_defaults(func=handle_status)

    # init
    p_init = subparsers.add_parser(
        "init", parents=[common_parent], help="Initialize device identity and cryptographic keys"
    )
    p_init.set_defaults(func=handle_init)

    # pair
    p_pair = subparsers.add_parser(
        "pair", parents=[common_parent], help="Initiate pairing with KBCON Central Dashboard"
    )
    p_pair.add_argument(
        "--server",
        type=str,
        default=DEFAULT_CENTRAL_URL,
        help="KBCON central dashboard URL",
    )
    p_pair.add_argument(
        "--force",
        action="store_true",
        help="Force re-pairing even if already paired",
    )
    p_pair.set_defaults(func=handle_pair)

    # unpair
    p_unpair = subparsers.add_parser(
        "unpair", parents=[common_parent], help="Revoke local device pairing"
    )
    p_unpair.set_defaults(func=handle_unpair)

    return parser


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register 'device' subcommand into root dh360-hi CLI parser."""
    parser = subparsers.add_parser(
        "device",
        help="KBCON device provisioning, cryptographic identity, and robot pairing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dev_sub = parser.add_subparsers(dest="subcommand", required=True)

    p_status = dev_sub.add_parser("status", help="Show device identity and pairing status")
    p_status.add_argument("--dir", type=str, default=None)
    p_status.set_defaults(func=handle_status)

    p_init = dev_sub.add_parser("init", help="Initialize device identity and cryptographic keys")
    p_init.add_argument("--dir", type=str, default=None)
    p_init.set_defaults(func=handle_init)

    p_pair = dev_sub.add_parser("pair", help="Initiate pairing with KBCON Central Dashboard")
    p_pair.add_argument("--dir", type=str, default=None)
    p_pair.add_argument("--server", type=str, default=DEFAULT_CENTRAL_URL)
    p_pair.add_argument("--force", action="store_true")
    p_pair.set_defaults(func=handle_pair)

    p_unpair = dev_sub.add_parser("unpair", help="Revoke local device pairing")
    p_unpair.add_argument("--dir", type=str, default=None)
    p_unpair.set_defaults(func=handle_unpair)


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI execution entry point for standalone kbcon-device."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return args.func(args)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130
    except Exception as exc:
        sys.stderr.write(f"\n[ERROR] {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
