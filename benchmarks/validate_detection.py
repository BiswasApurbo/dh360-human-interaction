#!/usr/bin/env python3
"""Functional Face Detection Validation Benchmark.

Exercises the YuNet face detector on a standard, non-identifying face fixture
to verify end-to-end detection functionality, bounding box coordinates, landmark
localization, and confidence scores across video and static frames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_test_video import generate_face_animation_video  # noqa: E402

from dh360_hi.vision.benchmark import PipelineBenchmark, StageTimer  # noqa: E402
from dh360_hi.vision.detector import YuNetFaceDetector  # noqa: E402
from dh360_hi.vision.frame import VideoFileFrameSource  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "weights" / "face_detection_yunet_2023mar.onnx"
DEFAULT_FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_face.jpg"
DEFAULT_ANIMATED_VIDEO = PROJECT_ROOT / "data" / "face_validation_video.mp4"
MAC_DISCLAIMER = (
    "Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance."
)


def run_functional_validation(
    video_path: Path,
    detector: YuNetFaceDetector,
    max_frames: int = 100,
    device_label: str = "MacBook M4 Air",
) -> int:
    """Run functional validation loop verifying face detections on every frame."""
    source = VideoFileFrameSource(video_path)
    bench = PipelineBenchmark()
    sample_detections = []

    print("\n================================================================================")
    print("DH360 HUMAN INTERACTION — FUNCTIONAL FACE DETECTION VALIDATION")
    print("================================================================================")
    print(f"Validation Target:        {video_path}")
    print(f"Detector Weights:         {detector.model_path.name}")
    print(f"Confidence Threshold:     {detector.confidence_threshold}")

    with source:
        input_resolution = source.resolution
        bench.start()

        for frame in source:
            if bench.frame_count >= max_frames:
                break

            stage_times: dict[str, float] = {}

            with StageTimer() as det_timer:
                result = detector.detect(frame)
            stage_times["face_detection"] = det_timer.elapsed_ms

            # Extract confidence scores
            confidences = [d.confidence for d in result.detections]
            if result.detections and len(sample_detections) < 3:
                sample_detections.append(result.detections[0])

            bench.record_frame(
                total_latency_ms=det_timer.elapsed_ms,
                stage_latencies_ms=stage_times,
                detections_count=len(result.detections),
                confidences=confidences,
            )

    conf_stats = bench.get_confidence_stats()
    det_stats = bench.get_stage_stats("face_detection")

    notes = (
        f"Input Resolution:          {input_resolution[0]}x{input_resolution[1]}\n"
        f"Detector Input Resolution: {detector.input_size[0]}x{detector.input_size[1]}\n"
        f"Confidence Score Threshold:{detector.confidence_threshold}\n"
        f"Detection Success Rate:    {conf_stats.frames_with_detections}/{bench.frame_count} "
        f"({(conf_stats.frames_with_detections / max(bench.frame_count, 1)) * 100:.1f}%)"
    )

    report = bench.format_summary(device_label=device_label, notes=notes)
    print("\n" + report)

    print("\n--- Detailed Detection Validation Summary ---")
    print(f"  • Frames Processed:          {bench.frame_count}")
    print(f"  • Total Faces Detected:      {conf_stats.total_detections}")
    print(f"  • Frames with >=1 Face:      {conf_stats.frames_with_detections}/{bench.frame_count}")
    print(
        f"  • Confidence (Mean / p50):   {conf_stats.mean_confidence:.4f} / "
        f"{conf_stats.p50_confidence:.4f}"
    )
    print(
        f"  • Confidence (Min / Max):    {conf_stats.min_confidence:.4f} / "
        f"{conf_stats.max_confidence:.4f}"
    )
    print(f"  • Average Throughput:        {bench.average_fps:.2f} FPS")
    print(f"  • p50 Detector Latency:      {det_stats.p50_ms:.2f} ms")
    print(f"  • p95 Detector Latency:      {det_stats.p95_ms:.2f} ms")
    print(f"  • p99 Detector Latency:      {det_stats.p99_ms:.2f} ms")
    print(f"  • Performance Note:          {MAC_DISCLAIMER}\n")

    if sample_detections:
        print("Sample Detected Face Coordinates (First Frame):")
        sample = sample_detections[0]
        box_str = f"({sample.x:.1f}, {sample.y:.1f}, {sample.width:.1f}, {sample.height:.1f})"
        print(f"  Bounding Box (x, y, w, h):   {box_str}")
        print(f"  Confidence Score:            {sample.confidence:.4f}")
        if sample.landmarks:
            print(f"  Right Eye:                   {sample.landmarks.right_eye}")
            print(f"  Left Eye:                    {sample.landmarks.left_eye}")
            print(f"  Nose Tip:                    {sample.landmarks.nose_tip}")
            print(f"  Right Mouth Corner:          {sample.landmarks.right_mouth_corner}")
            print(f"  Left Mouth Corner:           {sample.landmarks.left_mouth_corner}")

    # Functional validation assertions
    if conf_stats.total_detections == 0:
        print("\n❌ VALIDATION FAILED: No faces detected in test fixture!", file=sys.stderr)
        return 1

    print("\n✅ VALIDATION PASSED: YuNet successfully detected faces with high confidence.\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run functional face detection validation benchmark on standard fixture"
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help=f"Path to reference face image fixture (default: {DEFAULT_FIXTURE_PATH})",
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
        default=100,
        help="Number of validation frames to generate and evaluate (default: 100)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.6,
        help="Confidence score threshold for detections (default: 0.6)",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=DEFAULT_ANIMATED_VIDEO,
        help=f"Output path for generated validation video (default: {DEFAULT_ANIMATED_VIDEO})",
    )
    args = parser.parse_args()

    # Ensure model exists
    if not args.model.exists():
        print(f"Model file not found at {args.model}")
        print("Downloading via scripts/download_models.py...")
        from scripts.download_models import REGISTRY, download_resource

        download_resource(REGISTRY["yunet"])

    # Ensure fixture exists
    if not args.fixture.exists():
        print(f"Fixture file not found at {args.fixture}")
        print("Downloading via scripts/download_models.py...")
        from scripts.download_models import REGISTRY, download_resource

        download_resource(REGISTRY["sample_face"])

    # Generate animated multi-frame validation video from standard fixture
    msg = f"Generating animated validation video ({args.frames} frames) from {args.fixture.name}..."
    print(msg)
    generate_face_animation_video(
        face_image_path=args.fixture,
        output_path=args.video_output,
        target_width=640,
        target_height=480,
        fps=30.0,
        num_frames=args.frames,
    )

    detector = YuNetFaceDetector(
        model_path=args.model,
        input_size=(640, 480),
        score_threshold=args.score_threshold,
    )

    return run_functional_validation(
        video_path=args.video_output,
        detector=detector,
        max_frames=args.frames,
    )


if __name__ == "__main__":
    sys.exit(main())
