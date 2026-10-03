"""Root CLI entry point and dispatch orchestrator for dh360-hi."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from dh360_hi.cli.config_cmd import handle_config
from dh360_hi.cli.config_cmd import register_subparser as register_config
from dh360_hi.cli.enroll import handle_enroll
from dh360_hi.cli.enroll import register_subparser as register_enroll
from dh360_hi.cli.formatting import print_error
from dh360_hi.cli.health import handle_health
from dh360_hi.cli.health import register_subparser as register_health
from dh360_hi.cli.person import handle_person
from dh360_hi.cli.person import register_subparser as register_person
from dh360_hi.cli.run import handle_run
from dh360_hi.cli.run import register_subparser as register_run
from dh360_hi.cli.session import handle_session
from dh360_hi.cli.session import register_subparser as register_session
from dh360_hi.contracts.errors import ApplicationError


def create_parser() -> argparse.ArgumentParser:
    """Construct the root argument parser and register all subcommands."""
    parser = argparse.ArgumentParser(
        prog="dh360-hi",
        description="Digital Health 360 Human Interaction Subsystem Operator CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and unhandled traceback printing",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    register_run(subparsers)
    register_health(subparsers)
    register_person(subparsers)
    register_enroll(subparsers)
    register_session(subparsers)
    register_config(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Root CLI execution entry point with ApplicationError boundary."""
    parser = create_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # Configure logging level
    log_level = logging.DEBUG if getattr(args, "debug", False) else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dispatch_map = {
        "run": handle_run,
        "health": handle_health,
        "person": handle_person,
        "enroll": handle_enroll,
        "session": handle_session,
        "config": handle_config,
    }

    handler = dispatch_map.get(args.command)
    if not handler:
        parser.print_help(sys.stderr)
        return 1

    try:
        return handler(args)
    except ApplicationError as app_err:
        print_error(app_err.message, code=app_err.code)
        if getattr(args, "debug", False):
            logging.exception("ApplicationError occurred during command execution")
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\nOperation interrupted by user.\n")
        return 130
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print_error(f"Unexpected operational error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
