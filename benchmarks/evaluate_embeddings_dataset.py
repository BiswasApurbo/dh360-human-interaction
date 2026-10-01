#!/usr/bin/env python3
"""Offline Face Embedding & Verification Evaluation Harness.

Constructs paired genuine (same identity) and impostor (different identity)
trials to evaluate similarity distributions, FAR, FRR, EER, and calibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src and project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_models import download_all_resources  # noqa: E402

from dh360_hi.vision.aligner import FaceAligner  # noqa: E402
from dh360_hi.vision.detector import FaceDetection, FaceLandmarks  # noqa: E402
from dh360_hi.vision.embedder import SFaceEmbedder  # noqa: E402
from dh360_hi.vision.evaluation import (  # noqa: E402
    PairComparison,
    compute_cosine_distance,
    compute_cosine_similarity,
    evaluate_pairs,
    format_evaluation_report,
)

MODELS_DIR = PROJECT_ROOT / "models" / "weights"
SFACE_FP32_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
SFACE_INT8_PATH = MODELS_DIR / "face_recognition_sface_2021dec_int8.onnx"
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_face.jpg"

MAC_DISCLAIMER = (
    "Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance."
)


def _generate_identity_variations(base_img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Generate realistic variations of a reference face (illumination, noise, blur, contrast)."""
    variations: list[tuple[str, np.ndarray]] = [("base", base_img.copy())]

    # 1. Illumination / Brightness changes (+30, -30)
    bright = cv2.convertScaleAbs(base_img, alpha=1.0, beta=30)
    dark = cv2.convertScaleAbs(base_img, alpha=1.0, beta=-30)
    variations.append(("bright", bright))
    variations.append(("dark", dark))

    # 2. Contrast variation (alpha=1.2, alpha=0.8)
    high_contrast = cv2.convertScaleAbs(base_img, alpha=1.2, beta=0)
    low_contrast = cv2.convertScaleAbs(base_img, alpha=0.8, beta=0)
    variations.append(("high_contrast", high_contrast))
    variations.append(("low_contrast", low_contrast))

    # 3. Mild Gaussian noise
    noise = np.random.normal(0, 5, base_img.shape).astype(np.int16)  # noqa: NPY002
    noisy = np.clip(base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    variations.append(("mild_noise", noisy))

    # 4. Mild Gaussian blur
    blurred = cv2.GaussianBlur(base_img, (3, 3), 0.8)
    variations.append(("mild_blur", blurred))

    return variations


def _create_synthetic_distinct_subject(idx: int) -> np.ndarray:
    """Generate a distinct synthetic face-like subject pattern for impostor evaluation."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Distinct background hue & pattern
    base_color = ((idx * 60) % 256, (idx * 90 + 50) % 256, (idx * 130 + 100) % 256)
    cv2.rectangle(img, (150, 150), (360, 390), base_color, -1)
    # Eyes & Nose
    eye_offset = (idx * 5) % 20
    cv2.circle(img, (230 - eye_offset, 250), 15, (255, 255, 255), -1)
    cv2.circle(img, (280 + eye_offset, 250), 15, (255, 255, 255), -1)
    cv2.circle(img, (255, 300), 10, (100, 100, 255), -1)
    cv2.ellipse(img, (255, 340), (30, 15), 0, 0, 180, (50, 50, 200), -1)
    return img


def build_evaluation_dataset(
    fixture_img: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    """Construct multi-sample dataset partitioned by identity."""
    dataset: dict[str, list[np.ndarray]] = {}

    # Subject 1 (Reference Fixture + Variations)
    dataset["subject_1"] = [img for _, img in _generate_identity_variations(fixture_img)]

    # Impostor Subjects (Distinct synthetic identities + variations)
    for sub_idx in range(2, 6):
        sub_label = f"subject_{sub_idx}"
        base_synth = _create_synthetic_distinct_subject(sub_idx)
        dataset[sub_label] = [img for _, img in _generate_identity_variations(base_synth)]

    return dataset


def run_offline_evaluation(
    embedder: SFaceEmbedder,
    aligner: FaceAligner,
    fixture_path: Path,
) -> int:
    """Run genuine vs. impostor evaluation across dataset."""
    img = cv2.imread(str(fixture_path))
    if img is None:
        print(f"Error: Could not read fixture from {fixture_path}")
        return 1

    dataset = build_evaluation_dataset(img)
    total_imgs = sum(len(v) for v in dataset.values())
    print(f"Built offline dataset: {len(dataset)} identities ({total_imgs} total images).")

    # Extract aligned embeddings for all samples
    # Standard detection landmarks for alignment
    lm = FaceLandmarks(
        right_eye=(252.0, 259.0),
        left_eye=(323.0, 267.0),
        nose_tip=(284.0, 313.0),
        right_mouth_corner=(254.0, 351.0),
        left_mouth_corner=(309.0, 357.0),
    )
    det = FaceDetection(bbox=(207.8, 182.5, 145.9, 206.9), confidence=0.91, landmarks=lm)

    subject_embeddings: dict[str, list[np.ndarray]] = {}
    for sub_id, images in dataset.items():
        embs = []
        for im in images:
            aligned_res = aligner.align(im, det)
            if aligned_res.is_success:
                emb_res = embedder.embed(aligned_res)
                if emb_res.is_normalized:
                    embs.append(emb_res.embedding)
        subject_embeddings[sub_id] = embs

    # Form Genuine and Impostor Pairs
    comparisons: list[PairComparison] = []
    pair_counter = 0

    subjects = list(subject_embeddings.keys())
    # 1. Genuine pairs (within each subject)
    for sub_id in subjects:
        embs = subject_embeddings[sub_id]
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                pair_counter += 1
                sim = compute_cosine_similarity(embs[i], embs[j])
                dist = compute_cosine_distance(embs[i], embs[j])
                comparisons.append(
                    PairComparison(
                        pair_id=f"pair_{pair_counter}",
                        label1=f"{sub_id}_sample_{i}",
                        label2=f"{sub_id}_sample_{j}",
                        is_genuine=True,
                        cosine_similarity=sim,
                        cosine_distance=dist,
                    )
                )

    # 2. Impostor pairs (between different subjects)
    for s1_idx in range(len(subjects)):
        for s2_idx in range(s1_idx + 1, len(subjects)):
            sub1 = subjects[s1_idx]
            sub2 = subjects[s2_idx]
            for i, e1 in enumerate(subject_embeddings[sub1]):
                for j, e2 in enumerate(subject_embeddings[sub2]):
                    pair_counter += 1
                    sim = compute_cosine_similarity(e1, e2)
                    dist = compute_cosine_distance(e1, e2)
                    comparisons.append(
                        PairComparison(
                            pair_id=f"pair_{pair_counter}",
                            label1=f"{sub1}_sample_{i}",
                            label2=f"{sub2}_sample_{j}",
                            is_genuine=False,
                            cosine_similarity=sim,
                            cosine_distance=dist,
                        )
                    )

    report = evaluate_pairs(
        comparisons,
        model_name=embedder.model_name,
        model_version=embedder.model_version,
    )

    formatted = format_evaluation_report(report)
    print("\n" + formatted)
    print(f"Performance Note: {MAC_DISCLAIMER}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run offline face verification and threshold calibration"
    )
    parser.add_argument(
        "--model",
        choices=["sface", "sface_int8"],
        default="sface",
        help="Model to evaluate (default: sface)",
    )
    args = parser.parse_args()

    download_all_resources()

    model_path = SFACE_FP32_PATH if args.model == "sface" else SFACE_INT8_PATH
    model_name = "SFace-FP32" if args.model == "sface" else "SFace-INT8"
    embedder = SFaceEmbedder(model_path=model_path, model_name=model_name)
    aligner = FaceAligner(output_size=(112, 112))

    return run_offline_evaluation(embedder, aligner, FIXTURE_PATH)


if __name__ == "__main__":
    sys.exit(main())
