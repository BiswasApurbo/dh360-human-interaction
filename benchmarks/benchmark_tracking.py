#!/usr/bin/env python3
"""Reproducible face detection + tracking benchmark script.

Measures combined pipeline throughput and per-stage latency distributions (p50, p95, p99)
for YuNet face detector and modular FaceTracker.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src and project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_test_video import generate_face_animation_video  # noqa: E402

from dh360_hi.vision.benchmark import PipelineBenchmark, StageTimer  # noqa: E402
from dh360_hi.vision.detector import YuNetFaceDetector  # noqa: E402
from dh360_hi.vision.frame import (  # noqa: E402
    FrameSource,
    SyntheticFrameSource,
    VideoFileFrameSource,
    WebcamFrameSource,
)
from dh360_hi.vision.tracker import FaceTracker  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "weights" / "face_detection_yunet_2023mar.onnx"
DEFAULT_FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_face.jpg"
DEFAULT_ANIMATED_VIDEO = PROJECT_ROOT / "data" / "face_tracking_video.mp4"
MAC_DISCLAIMER = (
    "Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance."
)


def run_tracking_benchmark(
    source: FrameSource,
    detector: YuNetFaceDetector,
    tracker: FaceTracker,
    max_frames: int = 150,
    device_label: str = "MacBook M4 Air",
) -> int:
    """Run combined detection and tracking benchmark loop."""
    bench = PipelineBenchmark()
    total_faces_detected = 0
    all_seen_track_ids: set[int] = set()

    print("\n================================================================================")
    print("DH360 HUMAN INTERACTION — VISION DETECTION + TRACKING BENCHMARK")
    print("================================================================================")

    with source:
        input_resolution = source.resolution
        bench.start()

        for frame in source:
            if bench.frame_count >= max_frames:
                break

            stage_times: dict[str, float] = {}

            # 1. Detection Stage
            with StageTimer() as det_timer:
                det_result = detector.detect(frame)
            stage_times["face_detection"] = det_timer.elapsed_ms

            # 2. Tracking Stage
            with StageTimer() as track_timer:
                track_result = tracker.update(det_result, frame_id=frame.frame_id)
            stage_times["face_tracking"] = track_timer.elapsed_ms

            total_faces_detected += len(det_result.detections)
            for t in track_result.tracks:
                all_seen_track_ids.add(t.track_id)

            total_frame_latency = det_timer.elapsed_ms + track_timer.elapsed_ms
            confidences = [d.confidence for d in det_result.detections]

            bench.record_frame(
                total_latency_ms=total_frame_latency,
                stage_latencies_ms=stage_times,
                detections_count=len(det_result.detections),
                confidences=confidences,
            )

    det_stats = bench.get_stage_stats("face_detection")
    track_stats = bench.get_stage_stats("face_tracking")
    conf_stats = bench.get_confidence_stats()

    notes = (
        f"Input Resolution:          {input_resolution[0]}x{input_resolution[1]}\n"
        f"Detector Input Resolution: {detector.input_size[0]}x{detector.input_size[1]}\n"
        f"Score Threshold:           {detector.confidence_threshold}\n"
        f"Total Faces Detected:      {total_faces_detected}\n"
        f"Unique Track IDs Assigned: {len(all_seen_track_ids)}\n"
        f"Active Tracks at End:      {len(tracker.active_tracks)}"
    )

    report = bench.format_summary(device_label=device_label, notes=notes)
    print("\n" + report)

    print("\n--- Key Metrics Summary ---")
    print(f"  • Frames Processed:          {bench.frame_count}")
    print(f"  • Total Faces Detected:      {total_faces_detected}")
    print(f"  • Unique Track IDs:          {len(all_seen_track_ids)}")
    print(f"  • Active Tracks:             {len(tracker.active_tracks)}")
    print(f"  • Average Combined FPS:      {bench.average_fps:.2f} FPS")
    print(f"  • Detector Latency (p50):    {det_stats.p50_ms:.2f} ms")
    print(f"  • Tracker Latency (p50):     {track_stats.p50_ms:.2f} ms")
    if conf_stats.total_detections > 0:
        print(f"  • Detection Confidence (p50):{conf_stats.p50_confidence:.4f}")
    print(f"  • Performance Note:          {MAC_DISCLAIMER}\n")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run combined YuNet face detection and FaceTracker benchmark"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--video", type=Path, help="Path to video file for benchmarking")
    group.add_argument("--webcam", type=int, help="Camera index for live webcam benchmarking")
    group.add_argument(
        "--fixture",
        action="store_true",
        help="Generate and benchmark animated face fixture video (default)",
    )
    group.add_argument(
        "--synthetic",
        action="store_true",
        help="Benchmark on synthetic geometric stream (no faces)",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to YuNet ONNX model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=150,
        help="Number of frames to process in benchmark (default: 150)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Tracker IoU overlap matching threshold (default: 0.3)",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=15,
        help="Tracker max missed frames before track deletion (default: 15)",
    )
    args = parser.parse_args()

    # Ensure model weights exist
    if not args.model.exists():
        print(f"Model file not found at {args.model}")
        print("Downloading via scripts/download_models.py...")
        from scripts.download_models import REGISTRY, download_resource

        download_resource(REGISTRY["yunet"])

    source: FrameSource
    if args.video:
        source = VideoFileFrameSource(args.video)
    elif args.webcam is not None:
        source = WebcamFrameSource(camera_index=args.webcam)
    elif args.synthetic:
        source = SyntheticFrameSource(width=640, height=480, num_frames=args.frames)
    else:
        # Default: fixture animated face video
        if not DEFAULT_FIXTURE_PATH.exists():
            from scripts.download_models import REGISTRY, download_resource

            download_resource(REGISTRY["sample_face"])

        generate_face_animation_video(
            face_image_path=DEFAULT_FIXTURE_PATH,
            output_path=DEFAULT_ANIMATED_VIDEO,
            target_width=640,
            target_height=480,
            fps=30.0,
            num_frames=args.frames,
        )
        source = VideoFileFrameSource(DEFAULT_ANIMATED_VIDEO)

    detector = YuNetFaceDetector(model_path=args.model, input_size=(640, 480))
    tracker = FaceTracker(
        iou_threshold=args.iou_threshold,
        max_missed_frames=args.max_missed,
    )

    return run_tracking_benchmark(
        source=source,
        detector=detector,
        tracker=tracker,
        max_frames=args.frames,
    )


if __name__ == "__main__":
    sys.exit(main())
