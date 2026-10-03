"""CLI handler for historical interaction sessions and engagement analytics."""

from __future__ import annotations

import argparse
from typing import Any

from dh360_hi.application.factory import create_interaction_service
from dh360_hi.application.loader import load_config
from dh360_hi.backend.models import SessionQueryFilter
from dh360_hi.cli.formatting import format_json, format_key_value, format_table
from dh360_hi.contracts.errors import EntityNotFoundError


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'session' subcommand and its options."""
    parser = subparsers.add_parser(
        "session",
        help="Query historical interaction sessions and engagement analytics",
        description="Historical interaction session auditing and aggregate analytics.",
    )
    session_sub = parser.add_subparsers(dest="session_action", required=True)

    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--preset", help="Configuration preset (development, testing, production)")
        p.add_argument("--config-file", help="Path to TOML configuration file")
        p.add_argument("--db-path", help="Override SQLite database path")

    def add_filter_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--person-id", help="Filter by person UUID identifier")
        p.add_argument(
            "--is-identified",
            action="store_true",
            default=None,
            help="Filter for identified sessions only",
        )
        p.add_argument(
            "--is-anonymous",
            action="store_true",
            default=None,
            help="Filter for anonymous sessions only",
        )
        p.add_argument(
            "--started-after", type=float, help="Filter sessions started after epoch timestamp"
        )
        p.add_argument(
            "--started-before", type=float, help="Filter sessions started before epoch timestamp"
        )
        p.add_argument(
            "--ended-after", type=float, help="Filter sessions ended after epoch timestamp"
        )
        p.add_argument(
            "--ended-before", type=float, help="Filter sessions ended before epoch timestamp"
        )
        p.add_argument("--min-duration", type=float, help="Minimum duration in seconds")
        p.add_argument("--max-duration", type=float, help="Maximum duration in seconds")

    # 1. session list
    list_p = session_sub.add_parser("list", help="List recent interaction sessions with pagination")
    add_common_args(list_p)
    list_p.add_argument("--person-id", help="Optional person UUID filter")
    list_p.add_argument(
        "--limit", type=int, default=20, help="Maximum sessions to return (default: 20)"
    )
    list_p.add_argument("--offset", type=int, default=0, help="Pagination offset (default: 0)")
    list_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 2. session get <session_id>
    get_p = session_sub.add_parser("get", help="Retrieve details of a single historical session")
    add_common_args(get_p)
    get_p.add_argument("session_id", help="UUID identifier of the session")
    get_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 3. session query
    query_p = session_sub.add_parser("query", help="Query sessions with rich criteria filtering")
    add_common_args(query_p)
    add_filter_args(query_p)
    query_p.add_argument(
        "--limit", type=int, default=20, help="Maximum records to return (default: 20)"
    )
    query_p.add_argument("--offset", type=int, default=0, help="Offset (default: 0)")
    query_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 4. session analytics
    analytics_p = session_sub.add_parser(
        "analytics", help="Compute SQL-side aggregate session analytics"
    )
    add_common_args(analytics_p)
    add_filter_args(analytics_p)
    analytics_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 5. session stats <person_id>
    stats_p = session_sub.add_parser(
        "stats", help="Retrieve engagement statistics for an individual"
    )
    add_common_args(stats_p)
    stats_p.add_argument("person_id", help="UUID identifier of the person")
    stats_p.add_argument("--json", action="store_true", help="Output as JSON")


def _build_filter(args: argparse.Namespace) -> SessionQueryFilter:
    """Helper to build SessionQueryFilter from parsed CLI args."""
    is_identified: bool | None = None
    if getattr(args, "is_identified", None):
        is_identified = True
    elif getattr(args, "is_anonymous", None):
        is_identified = False

    return SessionQueryFilter(
        person_id=getattr(args, "person_id", None),
        is_identified=is_identified,
        started_after=getattr(args, "started_after", None),
        started_before=getattr(args, "started_before", None),
        ended_after=getattr(args, "ended_after", None),
        ended_before=getattr(args, "ended_before", None),
        min_duration_seconds=getattr(args, "min_duration", None),
        max_duration_seconds=getattr(args, "max_duration", None),
        limit=getattr(args, "limit", 50),
        offset=getattr(args, "offset", 0),
    )


def handle_session(args: argparse.Namespace) -> int:
    """Dispatch session subcommand execution."""
    action = args.session_action

    overrides: dict[str, Any] = {}
    if getattr(args, "db_path", None):
        overrides["db_path"] = args.db_path

    config = load_config(
        preset=getattr(args, "preset", None),
        config_file=getattr(args, "config_file", None),
        overrides=overrides if overrides else None,
        validate=True,
    )

    service = create_interaction_service(config=config, enable_perception=False)
    try:
        if action == "list":
            sessions = service.list_sessions(
                person_id=getattr(args, "person_id", None),
                limit=args.limit,
                offset=args.offset,
            )
            if getattr(args, "json", False):
                print(format_json([s.to_dict() for s in sessions]))
            else:
                if not sessions:
                    print("No historical interaction sessions found.")
                else:
                    headers = [
                        "Session ID",
                        "Person ID",
                        "Display Name",
                        "Started At",
                        "Duration (s)",
                        "Final State",
                    ]
                    rows = [
                        [
                            s.session_id[:8] + "...",
                            (s.person_id[:8] + "...") if s.person_id else "Anonymous",
                            s.display_name or "-",
                            s.started_at_iso[:19] if s.started_at_iso else "-",
                            f"{s.duration_seconds:.1f}",
                            s.final_state,
                        ]
                        for s in sessions
                    ]
                    print(format_table(headers, rows))
            return 0

        if action == "get":
            session_id = args.session_id
            session_dto = service.get_session(session_id)
            if session_dto is None:
                raise EntityNotFoundError("session", session_id)

            if getattr(args, "json", False):
                print(format_json(session_dto.to_dict()))
            else:
                pairs_get: list[tuple[str, Any]] = [
                    ("Session ID", session_dto.session_id),
                    ("Person ID", session_dto.person_id or "Anonymous"),
                    ("Display Name", session_dto.display_name or "-"),
                    ("Started At", session_dto.started_at_iso),
                    ("Ended At", session_dto.ended_at_iso),
                    ("Duration (seconds)", f"{session_dto.duration_seconds:.2f}"),
                    ("Identity Known", str(session_dto.identity_was_known)),
                    ("Identity Transitions", str(session_dto.identity_transitions_count)),
                    ("Final State", session_dto.final_state),
                    ("Created At", session_dto.created_at),
                ]
                print(
                    format_key_value(pairs_get, title=f"Session Record: {session_dto.session_id}")
                )
            return 0

        if action == "query":
            q_filter = _build_filter(args)
            sessions = service.query_sessions(filter=q_filter)
            count = service.count_query_sessions(filter=q_filter)

            if getattr(args, "json", False):
                print(
                    format_json(
                        {"total_matching": count, "sessions": [s.to_dict() for s in sessions]}
                    )
                )
            else:
                print(f"Matched {count} session(s) (showing {len(sessions)}):")
                if not sessions:
                    print("No matching sessions found.")
                else:
                    headers = [
                        "Session ID",
                        "Person ID",
                        "Display Name",
                        "Started At",
                        "Duration (s)",
                        "Final State",
                    ]
                    rows = [
                        [
                            s.session_id[:8] + "...",
                            (s.person_id[:8] + "...") if s.person_id else "Anonymous",
                            s.display_name or "-",
                            s.started_at_iso[:19] if s.started_at_iso else "-",
                            f"{s.duration_seconds:.1f}",
                            s.final_state,
                        ]
                        for s in sessions
                    ]
                    print(format_table(headers, rows))
            return 0

        if action == "analytics":
            q_filter = _build_filter(args)
            analytics = service.get_aggregate_analytics(filter=q_filter)
            if getattr(args, "json", False):
                print(format_json(analytics.to_dict()))
            else:
                pairs_analytics: list[tuple[str, Any]] = [
                    ("Total Sessions", analytics.total_sessions),
                    ("Total Duration (s)", f"{analytics.total_duration_seconds:.1f}"),
                    ("Average Duration (s)", f"{analytics.average_duration_seconds:.1f}"),
                    ("Min Duration (s)", f"{analytics.min_duration_seconds:.1f}"),
                    ("Max Duration (s)", f"{analytics.max_duration_seconds:.1f}"),
                    ("Identified Sessions", analytics.identified_sessions_count),
                    ("Anonymous Sessions", analytics.anonymous_sessions_count),
                    ("Earliest Session Started", analytics.earliest_session_started_at or "-"),
                    ("Latest Session Ended", analytics.latest_session_ended_at or "-"),
                ]
                print(format_key_value(pairs_analytics, title="Session Aggregate Analytics"))
            return 0

        if action == "stats":
            person_id = args.person_id
            stats = service.get_person_engagement_stats(person_id)
            if stats is None:
                raise EntityNotFoundError("person", person_id)

            if getattr(args, "json", False):
                print(format_json(stats.to_dict()))
            else:
                pairs_stats: list[tuple[str, Any]] = [
                    ("Person ID", stats.person_id),
                    ("Display Name", stats.display_name or "-"),
                    ("Total Sessions", stats.total_sessions),
                    ("Total Duration (s)", f"{stats.total_duration_seconds:.1f}"),
                    ("Average Duration (s)", f"{stats.average_duration_seconds:.1f}"),
                    ("First Interaction", stats.first_interaction_at or "-"),
                    ("Last Interaction", stats.last_interaction_at or "-"),
                ]
                print(
                    format_key_value(
                        pairs_stats, title=f"Engagement Stats: {stats.display_name or person_id}"
                    )
                )
            return 0

        return 0
    finally:
        service.stop(timeout=1.0)
