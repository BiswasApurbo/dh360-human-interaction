#!/usr/bin/env python3
"""Reproducible face detection benchmark script.

Measures throughput and latency distribution (p50, p95, p99) of the YuNet face detector
across video files, synthetic streams, or webcam input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dh360_hi.vision.benchmark import PipelineBenchmark, StageTimer  # noqa: E402
from dh360_hi.vision.detector import YuNetFaceDetector  # noqa: E402
from dh360_hi.vision.frame import (  # noqa: E402
    FrameSource,
    SyntheticFrameSource,
    VideoFileFrameSource,
    WebcamFrameSource,
)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "weights" / "face_detection_yunet_2023mar.onnx"
MAC_DISCLAIMER = (
    "Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance."
)


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse 'WIDTHxHEIGHT' string into (width, height) tuple."""
    parts = res_str.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution format '{res_str}'. Expected 'WIDTHxHEIGHT' (e.g. 320x320)"
        )
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Resolution values must be integers: {res_str}") from exc


def run_benchmark(
    source: FrameSource,
    detector: YuNetFaceDetector,
    max_frames: int = 200,
    device_label: str = "MacBook M4 Air",
) -> int:
    """Run benchmark loop over frame source using detector."""
    bench = PipelineBenchmark()
    total_faces_detected = 0
    input_resolution = (0, 0)

    print("\nStarting Vision Detection Benchmark...")
    with source:
        input_resolution = source.resolution
        bench.start()

        for frame in source:
            if bench.frame_count >= max_frames:
                break

            stage_times: dict[str, float] = {}

            # Face Detection Stage
            with StageTimer() as det_timer:
                result = detector.detect(frame)
            stage_times["face_detection"] = det_timer.elapsed_ms

            total_faces_detected += len(result.detections)
            total_frame_latency = det_timer.elapsed_ms
            confidences = [d.confidence for d in result.detections]

            bench.record_frame(
                total_latency_ms=total_frame_latency,
                stage_latencies_ms=stage_times,
                detections_count=len(result.detections),
                confidences=confidences,
            )

            if bench.frame_count % 50 == 0:
                print(
                    f"  Processed {bench.frame_count}/{max_frames} frames "
                    f"(Rolling FPS: {bench.rolling_fps:.1f}, "
                    f"Latest Latency: {det_timer.elapsed_ms:.2f} ms)"
                )

    # Compile and display report
    det_stats = bench.get_stage_stats("face_detection")
    notes = (
        f"Input Resolution:          {input_resolution[0]}x{input_resolution[1]}\n"
        f"Detector Input Resolution: {detector.input_size[0]}x{detector.input_size[1]}\n"
        f"Score Threshold:           {detector.confidence_threshold}\n"
        f"Total Faces Detected:      {total_faces_detected}"
    )

    report = bench.format_summary(device_label=device_label, notes=notes)
    print("\n" + report)

    print("\nKey Metrics Summary:")
    print(f"  • Frames Processed:          {bench.frame_count}")
    print(f"  • Total Faces Detected:      {total_faces_detected}")
    print(f"  • Average Throughput:        {bench.average_fps:.2f} FPS")
    print(f"  • p50 Detector Latency:      {det_stats.p50_ms:.2f} ms")
    print(f"  • p95 Detector Latency:      {det_stats.p95_ms:.2f} ms")
    print(f"  • p99 Detector Latency:      {det_stats.p99_ms:.2f} ms")
    print(f"  • Performance Note:          {MAC_DISCLAIMER}\n")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible YuNet face detector benchmark")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--video", type=Path, help="Path to video file for benchmarking")
    group.add_argument("--webcam", type=int, help="Camera index for live webcam benchmarking")
    group.add_argument(
        "--synthetic",
        action="store_true",
        help="Run benchmark with deterministic synthetic frames (default)",
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
        default=200,
        help="Number of frames to process in benchmark (default: 200)",
    )
    parser.add_argument(
        "--detector-size",
        type=parse_resolution,
        default=(320, 320),
        help="Detector input size WIDTHxHEIGHT (default: 320x320)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.6,
        help="Confidence score threshold for detections (default: 0.6)",
    )
    parser.add_argument(
        "--device-label",
        type=str,
        default="MacBook M4 Air",
        help="Device label for benchmark output",
    )

    args = parser.parse_args()

    # Select Frame Source
    source: FrameSource
    if args.video:
        source = VideoFileFrameSource(args.video)
    elif args.webcam is not None:
        source = WebcamFrameSource(camera_index=args.webcam)
    else:
        source = SyntheticFrameSource(width=640, height=480, num_frames=args.frames)

    detector = YuNetFaceDetector(
        model_path=args.model,
        input_size=args.detector_size,
        score_threshold=args.score_threshold,
    )

    return run_benchmark(
        source=source,
        detector=detector,
        max_frames=args.frames,
        device_label=args.device_label,
    )


if __name__ == "__main__":
    sys.exit(main())
