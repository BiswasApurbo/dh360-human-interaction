"""CLI handler for guided face enrollment workflows."""

from __future__ import annotations

import argparse
import contextlib
from typing import Any

from dh360_hi.application.factory import create_interaction_service
from dh360_hi.application.loader import load_config
from dh360_hi.cli.formatting import format_json, format_key_value
from dh360_hi.contracts.errors import EnrollmentAppError
from dh360_hi.vision.frame import (
    FrameSource,
    SyntheticFrameSource,
    VideoFileFrameSource,
    WebcamFrameSource,
)


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'enroll' subcommand and its options."""
    parser = subparsers.add_parser(
        "enroll",
        help="Execute an interactive camera face enrollment session for a person",
        description=(
            "Captures and registers facial biometric samples for an active consented person."
        ),
    )
    parser.add_argument("person_id", help="UUID identifier of the enrolled person")
    parser.add_argument(
        "--mode",
        default="add_samples",
        choices=["add_samples", "replace"],
        help="Enrollment mode (default: add_samples)",
    )
    parser.add_argument(
        "--source",
        default="webcam",
        choices=["webcam", "video", "synthetic"],
        help="Frame capture source (default: webcam)",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Webcam device index (default: 0)"
    )
    parser.add_argument("--video-file", help="Path to video file (when --source video)")
    parser.add_argument("--video-loop", action="store_true", help="Loop video stream continuously")
    parser.add_argument("--preset", help="Configuration preset (development, testing, production)")
    parser.add_argument("--config-file", help="Path to TOML configuration file")
    parser.add_argument("--db-path", help="Override SQLite database path")
    parser.add_argument(
        "--json", action="store_true", help="Output final enrollment result as JSON"
    )


def _build_frame_source(args: argparse.Namespace) -> FrameSource:
    """Instantiate appropriate FrameSource based on CLI args."""
    source_type = args.source
    if source_type == "synthetic":
        return SyntheticFrameSource(fps=30.0, num_frames=300)
    if source_type == "video":
        if not getattr(args, "video_file", None):
            raise EnrollmentAppError(
                message="--video-file is required when --source video is selected", reason="source"
            )
        return VideoFileFrameSource(
            file_path=args.video_file, loop=getattr(args, "video_loop", False)
        )
    return WebcamFrameSource(camera_index=getattr(args, "camera_index", 0))


def handle_enroll(args: argparse.Namespace) -> int:
    """Execute guided face enrollment workflow."""
    person_id = args.person_id
    mode = args.mode

    overrides: dict[str, Any] = {}
    if getattr(args, "db_path", None):
        overrides["db_path"] = args.db_path

    config = load_config(
        preset=getattr(args, "preset", None),
        config_file=getattr(args, "config_file", None),
        overrides=overrides if overrides else None,
        validate=True,
    )

    if not config.detector_model_path:
        raise EnrollmentAppError(
            message="Cannot start enrollment: 'detector_model_path' is not configured.",
            reason="missing_model_path",
            details={"field": "detector_model_path"},
        )

    frame_source = _build_frame_source(args)
    service = create_interaction_service(
        config=config, frame_source=frame_source, enable_perception=True
    )

    try:
        # Check person exists and has consent
        person = service.get_person(person_id)
        if person is None:
            raise EnrollmentAppError(
                message=f"Target person {person_id} does not exist",
                reason="person_not_found",
                details={"person_id": person_id},
            )

        if person.status != "active":
            raise EnrollmentAppError(
                message=f"Person {person_id} is not active (status: {person.status})",
                reason="person_inactive",
                details={"status": person.status},
            )

        if person.consent_status != "granted":
            cs = person.consent_status
            raise EnrollmentAppError(
                message=f"Biometric consent not granted for person {person_id} (consent: {cs})",
                reason="consent_required",
                details={"consent_status": cs},
            )

        p_name = person.display_name
        print(f"Starting enrollment session for '{p_name}' ({person_id}) [mode: {mode}]...")
        progress = service.start_enrollment(person_id=person_id, mode=mode)
        print(f"Target samples: {progress.target_samples} (minimum: {progress.minimum_samples})")
        print("Capturing frames from source (Press Ctrl+C to cancel)...")

        last_count = 0
        with frame_source:
            for frame in frame_source:
                progress = service.process_enrollment_frame(frame)
                if progress.samples_collected > last_count:
                    last_count = progress.samples_collected
                    print(f"  [+] Captured sample {last_count}/{progress.target_samples}")

                if progress.samples_collected >= progress.target_samples:
                    break

        if progress.samples_collected >= progress.minimum_samples:
            print(f"Collected {progress.samples_collected} samples. Committing to database...")
            result = service.commit_enrollment()
            if getattr(args, "json", False):
                print(format_json(result.to_dict()))
            else:
                pairs: list[tuple[str, Any]] = [
                    ("Success", str(result.is_success)),
                    ("Enrolled Person ID", result.person_id),
                    ("Display Name", result.display_name),
                    ("Committed Samples", result.samples_enrolled),
                    ("Total Persisted Embeddings", result.persisted_embedding_count),
                    ("Active Mode", result.mode),
                    ("State", result.state),
                ]
                print(format_key_value(pairs, title="Face Enrollment Completed"))
            return 0
        else:
            sc = progress.samples_collected
            ms = progress.minimum_samples
            print(f"Insufficient samples collected ({sc}/{ms}). Cancelling session...")
            service.cancel_enrollment("Insufficient samples")
            raise EnrollmentAppError(
                message=f"Enrollment cancelled: collected only {sc}/{ms} required samples",
                reason="insufficient_samples",
                details={"samples_collected": sc},
            )

    except KeyboardInterrupt:
        print("\nEnrollment interrupted by operator. Cancelling session...")
        with contextlib.suppress(Exception):
            service.cancel_enrollment("Operator interrupted")
        return 130
    finally:
        service.stop(timeout=1.0)
