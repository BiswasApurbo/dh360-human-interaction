"""CLI handler for configuration inspection and validation."""

from __future__ import annotations

import argparse
from typing import Any

from dh360_hi.application.loader import load_config
from dh360_hi.cli.formatting import format_json, format_key_value


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'config' subcommand and its options."""
    parser = subparsers.add_parser(
        "config",
        help="Inspect, validate, and manage subsystem configuration",
        description="Configuration management commands for presets, TOML files, and validation.",
    )
    config_sub = parser.add_subparsers(dest="config_action", required=True)

    # 1. config show
    show_p = config_sub.add_parser("show", help="Display active configuration settings")
    show_p.add_argument("--preset", help="Configuration preset (development, testing, production)")
    show_p.add_argument("--config-file", help="Path to TOML configuration file")
    show_p.add_argument("--db-path", help="Override SQLite database path")
    show_p.add_argument(
        "--json", action="store_true", help="Output in machine-readable JSON format"
    )

    # 2. config validate
    val_p = config_sub.add_parser("validate", help="Validate a configuration file or preset")
    val_p.add_argument("--preset", help="Configuration preset to validate")
    val_p.add_argument("--config-file", help="Path to TOML configuration file to validate")
    val_p.add_argument("--db-path", help="Override SQLite database path")

    # 3. config presets
    config_sub.add_parser("presets", help="List available built-in configuration presets")


def handle_config(args: argparse.Namespace) -> int:
    """Dispatch config subcommand execution."""
    action = args.config_action

    if action == "presets":
        print("Available Configuration Presets:")
        presets = [
            ("development", "Local development defaults (db: dh360_dev.db, target_fps: 30)"),
            ("testing", "Headless fast-paced test defaults (db: :memory:, target_fps: 60)"),
            (
                "production",
                "Strict deployment defaults (db: dh360_hi.db, target_fps: 30, max_errors: 5)",
            ),
        ]
        for name, desc in presets:
            print(f"  - {name.ljust(14)}: {desc}")
        return 0

    overrides: dict[str, Any] = {}
    if getattr(args, "db_path", None):
        overrides["db_path"] = args.db_path

    if action == "validate":
        config = load_config(
            preset=getattr(args, "preset", None),
            config_file=getattr(args, "config_file", None),
            overrides=overrides if overrides else None,
            validate=True,
        )
        print("Configuration validation successful: all parameters are valid.")
        return 0

    if action == "show":
        config = load_config(
            preset=getattr(args, "preset", None),
            config_file=getattr(args, "config_file", None),
            overrides=overrides if overrides else None,
            validate=False,
        )
        config_dict = config.to_dict()

        if getattr(args, "json", False):
            print(format_json(config_dict))
            return 0

        print("=== Active Configuration ===")
        top_pairs = [
            ("db_path", config_dict["db_path"]),
            ("auto_init_db", config_dict["auto_init_db"]),
            ("detector_model_path", config_dict["detector_model_path"]),
            ("embedder_model_path", config_dict["embedder_model_path"]),
            ("model_name", config_dict["model_name"]),
            ("model_version", config_dict["model_version"]),
            ("require_active", config_dict["require_active_for_gallery"]),
            ("require_consent", config_dict["require_consent_for_gallery"]),
        ]
        print(format_key_value(top_pairs, title="Storage & Models"))
        print()

        for sec in [
            "runtime",
            "quality",
            "session",
            "recognition",
            "worker",
            "identity",
            "target_selector",
            "presence",
            "enrollment",
        ]:
            if sec in config_dict and isinstance(config_dict[sec], dict):
                sec_pairs = [(k, v) for k, v in config_dict[sec].items()]
                print(format_key_value(sec_pairs, title=sec.capitalize()))
                print()

        return 0

    return 0
