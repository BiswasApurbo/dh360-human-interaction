"""KBCON Device Management CLI: kbcon-device.

Provides hardware device provisioning, Ed25519 identity initialization,
authoritative dashboard registration, and robot binding synchronization.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from dh360_hi.device.identity import DeviceIdentity, resolve_device_storage_dir
from dh360_hi.device.provisioning import (
    DEFAULT_CENTRAL_URL,
    ProvisioningError,
    get_key_fingerprint,
    register_device,
    restart_backend_service,
    sync_authoritative_config,
)

logger = logging.getLogger(__name__)


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
        print("Run 'kbcon-device init' or 'kbcon-device register' to create device identity.")
        print("=" * 62 + "\n")
        return 0

    identity = DeviceIdentity.load_or_create(resolved_dir)
    cfg = identity.config

    print(f"Device ID:         {identity.device_id}")
    try:
        pub_pem = identity.get_public_key_pem()
        print(f"Public Key SHA256: {get_key_fingerprint(pub_pem)}")
    except Exception as exc:
        print(f"Public Key:        Error ({exc})")

    if cfg.status == "paired" and cfg.paired_robot_id:
        print(f"Pairing Status:    PAIRED -> Robot [{cfg.paired_robot_id}]")
        tenant_display = cfg.tenant_id or "Salute"
        print(f"Assigned Tenant:   {tenant_display}")
        print(f"Paired At:         {cfg.paired_at or 'Unknown'}")
        if cfg.robot_config:
            name = cfg.robot_config.get("name", "Unknown")
            lang = cfg.robot_config.get("language", "Unknown")
            mode = cfg.robot_config.get("mode", "Unknown")
            print(f"Robot Info:        Name='{name}', Language='{lang}', Mode='{mode}'")
    elif cfg.status == "revoked":
        print("Pairing Status:    REVOKED (Device was unlinked by administrator)")
    else:
        print("Pairing Status:    UNPAIRED (Registered, awaiting Super Admin binding)")

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
    print(f"Fingerprint:       {get_key_fingerprint(pub_pem)}")
    print("Files created/verified:")
    print(f"  - {identity.id_file}")
    print(f"  - {identity.private_key_file} (mode 0600)")
    print(f"  - {identity.public_key_file}")
    print(f"  - {identity.config_file}")
    print("=" * 62 + "\n")
    return 0


def handle_register(args: argparse.Namespace) -> int:
    """Register permanent cryptographic identity with the central Device Registry."""
    server_url = getattr(args, "server", None) or DEFAULT_CENTRAL_URL
    storage_dir = Path(args.dir) if getattr(args, "dir", None) else None
    identity = DeviceIdentity.load_or_create(storage_dir)

    print(f"\nRegistering device '{identity.device_id}' with {server_url}...")
    try:
        data = register_device(identity, server_url=server_url)
    except ProvisioningError as err:
        print(f"\n[ERROR] Device registration failed: {err}\n", file=sys.stderr)
        return 1

    print("\n" + "=" * 64)
    print("              KBCON DEVICE REGISTRATION SUCCESS")
    print("=" * 64)
    print(f" Device ID:       {identity.device_id}")
    print(f" Fingerprint:     {get_key_fingerprint(identity.get_public_key_pem())}")
    if data.get("bound"):
        print(f" Binding Status:  BOUND -> Robot [{data.get('robotId')}]")
        print(f" Assigned Tenant: {data.get('tenantName') or data.get('tenant') or 'Salute'}")
        print(f" Robot Name:      {data.get('robotName') or data.get('name')}")
        restart_backend_service()
    else:
        print(" Binding Status:  UNBOUND (Registered in KBCON Device Registry)")
        print("\n INSTRUCTIONS:")
        print(" 1. Sign in to the KBCON Dashboard as Super Admin:")
        print(f"    {server_url}/robots")
        print(" 2. Open 'Device Registry' and select the target robot to bind this device.")
    print("=" * 64 + "\n")
    return 0


def handle_pair(args: argparse.Namespace) -> int:
    """Alias for register & synchronize authoritative binding."""
    return handle_register(args)


def handle_sync(args: argparse.Namespace) -> int:
    """Synchronize authoritative robot binding and tenant configuration from central dashboard."""
    server_url = getattr(args, "server", None) or DEFAULT_CENTRAL_URL
    storage_dir = Path(args.dir) if getattr(args, "dir", None) else None
    identity = DeviceIdentity.load_or_create(storage_dir)

    print(f"\nSynchronizing configuration for '{identity.device_id}' from {server_url}...")
    try:
        data = sync_authoritative_config(identity, server_url=server_url)
    except ProvisioningError as err:
        print(f"\n[ERROR] Sync failed: {err}\n", file=sys.stderr)
        return 1

    if data.get("bound"):
        print(f"\n[OK] Device is authoritatively bound to robot '{data.get('robotId')}'.")
        print(f"     Tenant: {data.get('tenantName') or data.get('tenant') or 'Salute'}")
        restart_backend_service()
    else:
        print(f"\n[INFO] Device '{identity.device_id}' is registered but currently unbound.")
    print()
    return 0


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

    # register
    p_reg = subparsers.add_parser(
        "register",
        parents=[common_parent],
        help="Register permanent cryptographic identity in KBCON Device Registry",
    )
    p_reg.add_argument("--server", type=str, default=DEFAULT_CENTRAL_URL)
    p_reg.set_defaults(func=handle_register)

    # pair
    p_pair = subparsers.add_parser(
        "pair",
        parents=[common_parent],
        help="Register and synchronize device binding with KBCON Dashboard",
    )
    p_pair.add_argument("--server", type=str, default=DEFAULT_CENTRAL_URL)
    p_pair.add_argument("--force", action="store_true")
    p_pair.set_defaults(func=handle_pair)

    # sync
    p_sync = subparsers.add_parser(
        "sync",
        parents=[common_parent],
        help="Synchronize authoritative robot and tenant assignment from dashboard",
    )
    p_sync.add_argument("--server", type=str, default=DEFAULT_CENTRAL_URL)
    p_sync.set_defaults(func=handle_sync)

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

    p_reg = dev_sub.add_parser("register", help="Register permanent cryptographic identity")
    p_reg.add_argument("--dir", type=str, default=None)
    p_reg.add_argument("--server", type=str, default=DEFAULT_CENTRAL_URL)
    p_reg.set_defaults(func=handle_register)

    p_pair = dev_sub.add_parser("pair", help="Register and synchronize device binding")
    p_pair.add_argument("--dir", type=str, default=None)
    p_pair.add_argument("--server", type=str, default=DEFAULT_CENTRAL_URL)
    p_pair.add_argument("--force", action="store_true")
    p_pair.set_defaults(func=handle_pair)

    p_sync = dev_sub.add_parser(
        "sync", help="Synchronize authoritative robot and tenant assignment"
    )
    p_sync.add_argument("--dir", type=str, default=None)
    p_sync.add_argument("--server", type=str, default=DEFAULT_CENTRAL_URL)
    p_sync.set_defaults(func=handle_sync)

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
