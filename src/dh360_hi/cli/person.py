"""CLI handler for person profile and identity management."""

from __future__ import annotations

import argparse
from typing import Any

from dh360_hi.application.factory import create_interaction_service
from dh360_hi.application.loader import load_config, resolve_cli_db_path
from dh360_hi.cli.formatting import format_json, format_key_value, format_table
from dh360_hi.contracts.dto import CreatePersonCommand, UpdatePersonCommand
from dh360_hi.contracts.errors import EntityNotFoundError


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'person' subcommand and its options."""
    parser = subparsers.add_parser(
        "person",
        help="Manage registered person profiles and consent records",
        description="Person identity and demographic management commands.",
    )
    person_sub = parser.add_subparsers(dest="person_action", required=True)

    # Global config options for all subcommands
    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--preset", help="Configuration preset (development, testing, production)")
        p.add_argument("--config-file", help="Path to TOML configuration file")
        p.add_argument("--db-path", help="Override SQLite database path (default: dh360_hi.db)")

    # 1. person list
    list_p = person_sub.add_parser("list", help="List registered person profiles")
    add_common_args(list_p)
    list_p.add_argument("--status", help="Filter by status (active, inactive, suspended)")
    list_p.add_argument(
        "--consent-status", help="Filter by consent status (granted, withdrawn, unknown)"
    )
    list_p.add_argument(
        "--json", action="store_true", help="Output in machine-readable JSON format"
    )

    # 2. person get <person_id>
    get_p = person_sub.add_parser("get", help="Retrieve details of a single person profile")
    add_common_args(get_p)
    get_p.add_argument("person_id", help="UUID identifier of the person")
    get_p.add_argument("--json", action="store_true", help="Output in machine-readable JSON format")

    # 3. person create
    create_p = person_sub.add_parser("create", help="Create a new person profile")
    add_common_args(create_p)
    create_p.add_argument("--display-name", required=True, help="Full display name")
    create_p.add_argument("--preferred-name", help="Optional preferred name or nickname")
    create_p.add_argument("--status", default="active", help="Person status (default: active)")
    create_p.add_argument(
        "--consent-status", default="granted", help="Consent status (default: granted)"
    )
    create_p.add_argument("--person-id", help="Optional explicit UUID identifier")
    create_p.add_argument("--json", action="store_true", help="Output created profile as JSON")

    # 4. person update <person_id>
    update_p = person_sub.add_parser("update", help="Update an existing person profile")
    add_common_args(update_p)
    update_p.add_argument("person_id", help="UUID identifier of the person")
    update_p.add_argument("--display-name", help="Updated display name")
    update_p.add_argument("--preferred-name", help="Updated preferred name")
    update_p.add_argument("--status", help="Updated status (active, inactive, suspended)")
    update_p.add_argument(
        "--consent-status", help="Updated consent status (granted, withdrawn, unknown)"
    )
    update_p.add_argument("--json", action="store_true", help="Output updated profile as JSON")

    # 5. person delete <person_id>
    delete_p = person_sub.add_parser(
        "delete", help="Delete a person profile and all associated face embeddings"
    )
    add_common_args(delete_p)
    delete_p.add_argument("person_id", help="UUID identifier of the person to delete")

    # 6. person reset-biometrics <person_id>
    reset_p = person_sub.add_parser(
        "reset-biometrics",
        help="Clear all facial biometric embeddings for a person without deleting their profile",
    )
    add_common_args(reset_p)
    reset_p.add_argument("person_id", help="UUID identifier of the target person")


def handle_person(args: argparse.Namespace) -> int:
    """Dispatch person subcommand execution."""
    action = args.person_action

    overrides: dict[str, Any] = {}
    cli_db = resolve_cli_db_path(args)
    if cli_db is not None:
        overrides["db_path"] = cli_db

    config = load_config(
        preset=getattr(args, "preset", None),
        config_file=getattr(args, "config_file", None),
        overrides=overrides if overrides else None,
        validate=True,
    )

    service = create_interaction_service(config=config, enable_perception=False)
    try:
        if action == "list":
            profiles = service.list_persons(
                status=getattr(args, "status", None),
                consent_status=getattr(args, "consent_status", None),
            )
            if getattr(args, "json", False):
                print(format_json([p.to_dict() for p in profiles]))
            else:
                if not profiles:
                    print("No registered person profiles found.")
                else:
                    headers = [
                        "Person ID",
                        "Display Name",
                        "Preferred",
                        "Status",
                        "Consent",
                        "Updated At",
                    ]
                    rows = [
                        [
                            p.person_id[:8] + "...",
                            p.display_name,
                            p.preferred_name or "-",
                            p.status,
                            p.consent_status,
                            p.updated_at[:19] if p.updated_at else "-",
                        ]
                        for p in profiles
                    ]
                    print(format_table(headers, rows))
            return 0

        if action == "get":
            person_id = args.person_id
            profile = service.get_person(person_id)
            if profile is None:
                raise EntityNotFoundError("person", person_id)

            if getattr(args, "json", False):
                print(format_json(profile.to_dict()))
            else:
                pairs: list[tuple[str, Any]] = [
                    ("Person ID", profile.person_id),
                    ("Display Name", profile.display_name),
                    ("Preferred Name", profile.preferred_name or "-"),
                    ("Status", profile.status),
                    ("Consent Status", profile.consent_status),
                    ("Consent Timestamp", profile.consent_timestamp or "-"),
                    ("Created At", profile.created_at),
                    ("Updated At", profile.updated_at or "-"),
                ]
                print(format_key_value(pairs, title=f"Person Profile: {profile.display_name}"))
            return 0

        if action == "create":
            create_cmd = CreatePersonCommand(
                display_name=args.display_name,
                preferred_name=args.preferred_name,
                status=args.status,
                consent_status=args.consent_status,
                notes="",
                person_id=getattr(args, "person_id", None),
            )
            created = service.create_person(create_cmd)
            if getattr(args, "json", False):
                print(format_json(created.to_dict()))
            else:
                print(f"Created person profile '{created.display_name}' (ID: {created.person_id})")
            return 0

        if action == "update":
            person_id = args.person_id
            update_cmd = UpdatePersonCommand(
                display_name=args.display_name,
                preferred_name=args.preferred_name,
                status=args.status,
                consent_status=args.consent_status,
                notes=None,
            )
            updated = service.update_person(person_id, update_cmd)
            if getattr(args, "json", False):
                print(format_json(updated.to_dict()))
            else:
                print(f"Updated person profile '{updated.display_name}' (ID: {updated.person_id})")
            return 0

        if action == "delete":
            person_id = args.person_id
            deleted = service.delete_person(person_id)
            if not deleted:
                raise EntityNotFoundError("person", person_id)
            print(f"Successfully deleted person profile {person_id}")
            return 0

        if action == "reset-biometrics":
            person_id = args.person_id
            count = service.reset_person_biometrics(person_id)
            print(f"Reset biometrics for person {person_id}: {count} embedding vector(s) deleted.")
            return 0

        return 0
    finally:
        service.stop(timeout=1.0)
