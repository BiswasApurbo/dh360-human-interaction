"""CLI handler for service health inspection and diagnostics."""

from __future__ import annotations

import argparse
from typing import Any

from dh360_hi.application.factory import create_interaction_service
from dh360_hi.application.health import SubsystemHealthStatus
from dh360_hi.application.loader import load_config
from dh360_hi.cli.formatting import format_json, format_key_value, format_table


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'health' subcommand and its options."""
    parser = subparsers.add_parser(
        "health",
        help="Inspect overall service health and subsystem diagnostics",
        description=(
            "Evaluates operational health across runtime, database, persistence, "
            "recognizer, and enrollment subsystems."
        ),
    )
    parser.add_argument("--preset", help="Configuration preset (development, testing, production)")
    parser.add_argument("--config-file", help="Path to TOML configuration file")
    parser.add_argument("--db-path", help="Override SQLite database path")
    parser.add_argument(
        "--json", action="store_true", help="Output in machine-readable JSON format"
    )


def handle_health(args: argparse.Namespace) -> int:
    """Evaluate and display service health snapshot."""
    overrides: dict[str, Any] = {}
    if getattr(args, "db_path", None):
        overrides["db_path"] = args.db_path

    config = load_config(
        preset=getattr(args, "preset", None),
        config_file=getattr(args, "config_file", None),
        overrides=overrides if overrides else None,
        validate=True,
    )

    enable_perc = bool(config.detector_model_path)
    service = create_interaction_service(config=config, enable_perception=enable_perc)
    try:
        snapshot = service.get_health_snapshot()

        if getattr(args, "json", False):
            print(format_json(snapshot.to_dict()))
        else:
            print(f"=== Service Health Snapshot [{snapshot.status.value.upper()}] ===")
            uptime_str = (
                f"{snapshot.metrics.uptime_seconds:.2f}" if snapshot.metrics is not None else "N/A"
            )
            top_pairs = [
                ("Overall Status", snapshot.status.value.upper()),
                ("Timestamp", snapshot.timestamp_iso),
                ("Uptime (seconds)", uptime_str),
            ]
            print(format_key_value(top_pairs))
            print()

            headers = ["Subsystem", "Status", "Details"]
            rows = []
            for name, sub in snapshot.subsystems.items():
                status_str = sub.status.value.upper()
                details_str = (
                    ", ".join(f"{k}={v}" for k, v in sub.details.items()) if sub.details else "-"
                )
                if sub.message:
                    details_str += f" ({sub.message})"
                rows.append([name, status_str, details_str])

            print(format_table(headers, rows))

        return 1 if snapshot.status == SubsystemHealthStatus.UNHEALTHY else 0
    finally:
        service.stop(timeout=1.0)
