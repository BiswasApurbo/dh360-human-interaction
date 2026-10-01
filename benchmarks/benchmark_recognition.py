#!/usr/bin/env python3
"""End-to-End Face Recognition & Identity State Machine Benchmark.

Measures per-stage latencies across the full vision chain:
Detection -> Tracking -> Quality Gate -> Alignment -> Embedding -> Recognition -> Identity State.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Add src and project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_models import download_all_resources  # noqa: E402
from scripts.generate_test_video import generate_face_animation_video  # noqa: E402

from dh360_hi.vision.aligner import FaceAligner  # noqa: E402
from dh360_hi.vision.benchmark import PipelineBenchmark, StageTimer  # noqa: E402
from dh360_hi.vision.detector import YuNetFaceDetector  # noqa: E402
from dh360_hi.vision.embedder import SFaceEmbedder  # noqa: E402
from dh360_hi.vision.frame import Frame, VideoFileFrameSource  # noqa: E402
from dh360_hi.vision.identity_state import (  # noqa: E402
    IdentityState,
    IdentityStateManager,
)
from dh360_hi.vision.quality_gate import FaceQualityGate  # noqa: E402
from dh360_hi.vision.recognition import (  # noqa: E402
    FaceRecognizer,
    GalleryEntry,
    InMemoryGallery,
)
from dh360_hi.vision.tracker import FaceTracker  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models" / "weights"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_face.jpg"
ANIMATED_VIDEO_PATH = PROJECT_ROOT / "data" / "face_recognition_benchmark.mp4"

MAC_DISCLAIMER = (
    "Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance."
)


def run_recognition_benchmark(
    num_frames: int = 100,
    device_label: str = "MacBook M4 Air",
) -> int:
    """Run benchmark across the entire end-to-end vision and recognition pipeline."""
    # Ensure dependencies and fixtures exist
    download_all_resources()

    generate_face_animation_video(
        face_image_path=FIXTURE_PATH,
        output_path=ANIMATED_VIDEO_PATH,
        target_width=640,
        target_height=480,
        fps=30.0,
        num_frames=num_frames,
    )

    detector = YuNetFaceDetector(model_path=YUNET_PATH, input_size=(640, 480))
    tracker = FaceTracker()
    quality_gate = FaceQualityGate()
    aligner = FaceAligner(output_size=(112, 112))
    embedder = SFaceEmbedder(model_path=SFACE_PATH)

    # Enroll reference identity in gallery
    ref_img = cv2.imread(str(FIXTURE_PATH))
    assert ref_img is not None
    frame_0 = Frame(image=ref_img, frame_id=0, timestamp=0.0, source_info="ref")
    det_0 = detector.detect(frame_0)
    align_0 = aligner.align(frame_0, det_0.detections[0])
    emb_0 = embedder.embed(align_0)

    gallery = InMemoryGallery(
        [
            GalleryEntry(
                person_id="patient_001",
                display_name="Consented Reference Patient",
                embeddings=[emb_0.embedding],
                model_name="SFace",
                model_version="2021dec",
            )
        ]
    )
    recognizer = FaceRecognizer(gallery=gallery)
    id_manager = IdentityStateManager()

    source = VideoFileFrameSource(ANIMATED_VIDEO_PATH)
    bench = PipelineBenchmark()

    confirmed_identifications = 0
    total_evaluations = 0

    print("\n================================================================================")
    print("DH360 VISION PIPELINE: DETECTION -> TRACKING -> QUALITY -> EMBEDDING -> IDENTITY")
    print("================================================================================")

    with source:
        bench.start()
        for frame in source:
            if bench.frame_count >= num_frames:
                break

            stage_times: dict[str, float] = {}

            # 1. Detection
            with StageTimer() as det_timer:
                det_result = detector.detect(frame)
            stage_times["detection"] = det_timer.elapsed_ms

            # 2. Tracking
            with StageTimer() as track_timer:
                track_result = tracker.update(det_result, frame_id=frame.frame_id)
            stage_times["tracking"] = track_timer.elapsed_ms

            # 3. Quality + Align + Embed + Recognize + State per track
            q_time = 0.0
            align_time = 0.0
            emb_time = 0.0
            rec_time = 0.0
            state_time = 0.0

            for track in track_result.tracks:
                total_evaluations += 1

                # Quality Gate
                with StageTimer() as qt:
                    q_res = quality_gate.evaluate(frame, track)
                q_time += qt.elapsed_ms

                rec_res = None
                if q_res.is_acceptable:
                    # Alignment
                    with StageTimer() as at:
                        align_res = aligner.align(frame, track)
                    align_time += at.elapsed_ms

                    if align_res.is_success:
                        # Embedding
                        with StageTimer() as et:
                            emb_res = embedder.embed(align_res)
                        emb_time += et.elapsed_ms

                        # Recognition Matching
                        with StageTimer() as rt:
                            rec_res = recognizer.recognize(emb_res)
                        rec_time += rt.elapsed_ms

                # Identity State Machine Update
                with StageTimer() as st:
                    id_state = id_manager.update(
                        track_id=track.track_id,
                        rec_result=rec_res,
                        frame_id=frame.frame_id,
                    )
                state_time += st.elapsed_ms

                if id_state.state == IdentityState.IDENTIFIED:
                    confirmed_identifications += 1

            stage_times["quality_gate"] = q_time
            stage_times["alignment"] = align_time
            stage_times["embedding"] = emb_time
            stage_times["gallery_match"] = rec_time
            stage_times["identity_state"] = state_time

            total_frame_ms = (
                det_timer.elapsed_ms
                + track_timer.elapsed_ms
                + q_time
                + align_time
                + emb_time
                + rec_time
                + state_time
            )

            bench.record_frame(
                total_latency_ms=total_frame_ms,
                stage_latencies_ms=stage_times,
                detections_count=len(det_result.detections),
            )

    notes = (
        f"Input Resolution:          {source.resolution[0]}x{source.resolution[1]}\n"
        f"Total Track Evaluations:   {total_evaluations}\n"
        f"Confirmed Identifications: {confirmed_identifications}"
    )

    report = bench.format_summary(device_label=device_label, notes=notes)
    print("\n" + report)

    det_stats = bench.get_stage_stats("detection")
    track_stats = bench.get_stage_stats("tracking")
    q_stats = bench.get_stage_stats("quality_gate")
    align_stats = bench.get_stage_stats("alignment")
    emb_stats = bench.get_stage_stats("embedding")
    rec_stats = bench.get_stage_stats("gallery_match")
    state_stats = bench.get_stage_stats("identity_state")

    print("\n--- Per-Stage Latency Breakdown (p50) ---")
    print(f"  • Face Detection:          {det_stats.p50_ms:.2f} ms")
    print(f"  • Face Tracking:           {track_stats.p50_ms:.2f} ms")
    print(f"  • Quality Gate:            {q_stats.p50_ms:.2f} ms")
    print(f"  • Face Alignment:          {align_stats.p50_ms:.2f} ms")
    print(f"  • SFace Embedding:         {emb_stats.p50_ms:.2f} ms")
    print(f"  • Gallery Recognition:     {rec_stats.p50_ms:.2f} ms")
    print(f"  • Identity State Machine:  {state_stats.p50_ms:.2f} ms")
    print(f"  • Overall Pipeline FPS:    {bench.average_fps:.2f} FPS")
    print(f"  • Performance Note:        {MAC_DISCLAIMER}\n")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end face recognition and identity state tracking"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to process (default: 100)",
    )
    args = parser.parse_args()
    return run_recognition_benchmark(num_frames=args.frames)


if __name__ == "__main__":
    sys.exit(main())
